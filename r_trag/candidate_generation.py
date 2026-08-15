"""
RT‑accelerated candidate generation for late‑interaction retrieval.

Given a query (a set of token embeddings) and the centroid RT scene, this
module first casts one ray per query token into the scene, then uses the
centroid hit information to identify candidate documents and produces
approximate per‑document relevance scores.

The host‑side simulator provided here mirrors the logic that would run in
an actual OptiX hit shader.  In production the ray hits are accumulated
directly on the GPU using atomics without ever materializing the full list
of hits in HBM.
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

import numpy as np

from .rt_scene import RTScene


def generate_candidates(
    query_embeddings: np.ndarray,      # (num_query_tokens, embedding_dim)
    scene: RTScene,
    num_probe: int = 4,
    t_max_scale: float = 1.0,
) -> Tuple[Set[int], Dict[int, float]]:
    """
    Simulate RT‑based centroid interaction and approximate scoring.

    Args:
        query_embeddings: token‑wise query embeddings.
        scene: RT scene built from the codebook centroids.
        num_probe: number of proximate centroids to consider per query token.
        t_max_scale: optional scaling factor for the ray travel time (allows
            aggressive pruning; set to 1.0 for the full radius).

    Returns:
        candidate_doc_ids: unique document IDs collected from the inverted lists.
        approx_scores: doc_id -> approximate total score (sum of per‑token max).
    """
    candidate_docs: Set[int] = set()
    doc_to_token_max: Dict[int, Dict[int, float]] = {}

    # The real RT pipeline would launch one ray per (query_token, IVF_cluster)
    # combination.  Here we treat each query token as a single ray.
    for token_id, vec in enumerate(query_embeddings):
        # The query residual is projected to the 2D xOy plane.
        # The z‑coordinate is chosen below the sphere plane (z=0).
        xy = (float(vec[0]), float(vec[1]))
        ray_t_max = t_max_scale * 1.0  # In a real system, t_max encodes dynamic radius

        hits = list(scene.any_hit(xy, z0=0.0, z1=1.0, t_max=ray_t_max))

        # Sort by metric value (lower L2 distance or higher inner product).
        if scene.metric == "l2":
            hits_sorted = sorted(hits, key=lambda hit: hit.metric_value)
        else:
            hits_sorted = sorted(hits, key=lambda hit: hit.metric_value, reverse=True)

        # Consider the closest `num_probe` centroids.
        for hit in hits_sorted[:num_probe]:
            centroid_id = hit.centroid_id
            for doc_id in scene.inverted_lists.get(centroid_id, []):
                candidate_docs.add(doc_id)
                token_max = doc_to_token_max.setdefault(doc_id, {})
                # Keep the per‑token maximum; for L2, lower distance is better,
                # so we store the negative distance for score maximisation.
                if scene.metric == "l2":
                    contribution = -hit.metric_value
                else:
                    contribution = hit.metric_value
                token_max[token_id] = max(token_max.get(token_id, float("-inf")), contribution)

    # Accumulate the sum of the per‑token maxima to produce total score.
    approx_scores: Dict[int, float] = {
        doc_id: sum(token_max.values())
        for doc_id, token_max in doc_to_token_max.items()
    }

    return candidate_docs, approx_scores
