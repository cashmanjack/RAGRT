"""
RT scene abstraction for RTRAG.

This module defines the host‑side representation of the static ray‑tracing
scene built from the codebook centroids.  In the actual system this scene
would be instantiated on the GPU through NVIDIA OptiX: each centroid
becomes a sphere with attached inverted‑list metadata, and each query token
is launched as a ray.  The Python class here is used for prototyping,
unit testing, and as a natural control structure for the OptiX handles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Iterator, Optional

import numpy as np


@dataclass
class Ray:
    """A query token represented as a ray in the RT scene."""

    origin: np.ndarray          # (3,) float; z is used to isolate subspace
    direction: np.ndarray       # (3,) float; usually (0,0,1)
    t_max: float = 1.0
    visibility_mask: int = 0xFF
    query_id: int = -1
    token_id: int = -1


@dataclass
class Hit:
    """A ray‑sphere hit (centroid intersection)."""

    centroid_id: int
    t_hit: float
    metric_value: float
    query_id: int = -1
    token_id: int = -1


@dataclass
class RTScene:
    """
    Host‑side scene of centroid spheres.

    Attributes:
        metric: "l2" for Euclidean distance or "ip" for inner product (MIPS).
        sphere_radius: base radius used for all spheres (L2 mode) or
            the base radius used to derive per‑centroid radii (IP mode).
        positions: centroid_id -> (x, y, z) position in the scene.
        inverted_lists: centroid_id -> list of document IDs in the inverted list.
        per_centroid_radius: optional per‑centroid radius for the MIPS
            adaptation from JUNO++ (used when metric == "ip").
    """

    metric: str = "l2"
    sphere_radius: float = 1.0

    positions: Dict[int, Tuple[float, float, float]] = field(default_factory=dict)
    inverted_lists: Dict[int, List[int]] = field(default_factory=dict)
    per_centroid_radius: Dict[int, float] = field(default_factory=dict)

    # Actual OptiX backend handle (unused in the host‑side simulator)
    optix_scene_handle: Optional[int] = None

    def build_from_centroids(
        self,
        centroids: np.ndarray,
        doc_lists: List[List[int]],
        metric: str = "l2",
        radius: float = 1.0,
    ) -> None:
        """
        Populate the scene from a set of centroid vectors.

        Args:
            centroids: array of shape (num_centroids, dim) -- only the first
                two dimensions are used in the projective 2D scene.
            doc_lists: parallel list of document IDs for each centroid.
            metric: similarity metric; "l2" or "ip".
            radius: nominal sphere radius.
        """
        self.metric = metric.lower()
        self.sphere_radius = radius

        for i, (centroid, docs) in enumerate(zip(centroids, doc_lists)):
            # Place all spheres at z=1; rays will originate below this plane.
            self.positions[i] = (float(centroid[0]), float(centroid[1]), 1.0)
            self.inverted_lists[i] = docs

            # For IP metric, we use the per‑centroid radius expansion from
            # JUNO++: R' = sqrt(R^2 + x^2 + y^2) so that the inner product
            # can be recovered from t_hit without extra memory accesses.
            if self.metric == "ip":
                r2 = radius**2 + float(centroid[0])**2 + float(centroid[1])**2
                self.per_centroid_radius[i] = np.sqrt(r2)
            else:
                self.per_centroid_radius[i] = radius

    def any_hit(
        self,
        ray_xy: Tuple[float, float],
        z0: float,
        z1: float,
        t_max: float = 1.0,
        visibility_mask: int = 0xFF,
    ) -> Iterator[Hit]:
        """
        Simulate an OptiX any‑hit traversal for a vertical ray.

        In the real system this would be a CUDA/PTX shader triggered by
        ``trace``.  The simulator iterates through all centroid spheres and
        yields hits that satisfy the ray's travel constraints.  The returned
        `metric_value` is the actual similarity/distance metric.

        Args:
            ray_xy: (x, y) coordinate of the ray origin in the projective plane.
            z0: starting z coordinate of the ray.
            z1: ending z coordinate (dictated by the scene region).
            t_max: maximum ray travel time; used to implement dynamic radius.
            visibility_mask: optional scene‑level filter (not yet enforced).

        Yields:
            Hit objects for every sphere intercepted by the ray.
        """
        for cid, (x, y, z) in self.positions.items():
            # Ignore centroids outside the visible region when using masks.
            if not (visibility_mask >> (cid % 8)) & 1:
                continue

            dx = ray_xy[0] - x
            dy = ray_xy[1] - y
            dist2 = dx * dx + dy * dy

            radius = self.per_centroid_radius.get(cid, self.sphere_radius)

            # A vertical ray intersects the sphere centered at (x,y,z) only
            # if the perpendicular distance in xy is within the sphere radius.
            if dist2 > radius * radius:
                continue

            # Compute pixel‑space projection along the ray:
            # t_hit = R - sqrt(R^2 - dist^2)  (see JUNO ray‑time derivation)
            t_hit = radius - np.sqrt(max(radius * radius - dist2, 0.0))

            if t_hit > t_max:
                continue

            if self.metric == "l2":
                # For L2 distance we actually care about the xy distance,
                # which is the value we need to rank centroids.
                metric_value = np.sqrt(dist2)
            else:
                # For inner product, the t_hit trick from JUNO++ lets us
                # compute the original MIPS score.  Here we use a simplified
                # proxy for testing; the actual kernel uses the radius
                # adjustment formula.
                metric_value = (1.0 - t_hit) ** 2 - (radius ** 2 - (x**2 + y**2))

            yield Hit(
                centroid_id=cid,
                t_hit=t_hit,
                metric_value=float(metric_value),
            )
