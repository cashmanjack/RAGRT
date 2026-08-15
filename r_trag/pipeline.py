"""
End‑to‑end RTRAG pipeline.

The pipeline follows the design in `../NOTES.md`:

1. Offline: train a codebook from document token embeddings, build
   centroid→document inverted lists, and construct the RT scene.
2. Online:
   a. Encode the query using a late‑interaction encoder (e.g., ColBERT).
   b. Use the RT engine for centroid interaction: cast rays from each
      query token, collect hits, and obtain initial candidate documents.
   c. Fuse centroid‑based approximate scoring with candidate pruning:
      keep only the top `num_candidates` documents according to the
      approximate per‑document score.
   d. Run exact MaxSim scoring (TileMaxSim) only on the pruned set.
   e. Rank and return the top‑k documents.

TileMaxSim is external; any object exposing ``score(query, docs)`` can be
plugged in as `scoring_backend`.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .rt_scene import RTScene
from .candidate_generation import generate_candidates


class RTRAG:
    """Main entry point for the ray‑tracing accelerated late‑interaction system."""

    def __init__(
        self,
        metric: str = "ip",
        radius: float = 1.0,
        t_max_scale: float = 0.5,
        num_probe: int = 4,
        keep_ratio: float = 0.25,
        scoring_backend: Optional[object] = None,
    ) -> None:
        """
        Args:
            metric: similarity metric, "l2" or "ip" (inner product / MIPS).
            radius: base radius of centroid spheres.
            t_max_scale: scaling factor for dynamic radius (1.0 = full radius).
            num_probe: number of closest centroids to probe per query token.
            keep_ratio: fraction of initial candidates kept after pruning.
            scoring_backend: object with a ``score(query, docs)`` method
                (typically the TileMaxSim kernel).
        """
        self.metric = metric.lower()
        self.radius = radius
        self.t_max_scale = t_max_scale
        self.num_probe = num_probe
        self.keep_ratio = keep_ratio
        self.scoring = scoring_backend
        self.scene: Optional[RTScene] = None

    def build_index(
        self,
        centroids: np.ndarray,
        doc_ids_per_centroid: List[List[int]],
    ) -> None:
        """
        Build the RT scene and inverted lists from pre‑trained centroids.

        Args:
            centroids: shape (num_centroids, embedding_dim) array.
            doc_ids_per_centroid: parallel list of document IDs.
        """
        self.scene = RTScene(metric=self.metric, sphere_radius=self.radius)
        self.scene.build_from_centroids(centroids, doc_ids_per_centroid,
                                        metric=self.metric, radius=self.radius)

    def retrieve(
        self,
        query_embeddings: np.ndarray,
        top_k: int,
        documents: Optional[Sequence[object]] = None,
        doc_ids: Optional[Sequence[int]] = None,
        num_candidates: Optional[int] = None,
    ) -> List[int]:
        """
        Run the end‑to‑end retrieval pipeline.

        Args:
            query_embeddings: shape (num_query_tokens, emb_dim).
            top_k: number of final documents to return.
            documents: source documents (only used with the scoring backend).
            doc_ids: explicit list of document IDs if `documents` is not
                easily indexable.
            num_candidates: number of surviving candidates after approximate
                pruning.  If None, computed from `keep_ratio`.

        Returns:
            List of document IDs sorted by relevance (descending).
        """
        if self.scene is None:
            raise ValueError("Must build an index before calling retrieve().")

        # Step 1 + 2: RT‑based candidate generation and centroid interaction
        candidate_set, approx_scores = self._generate_candidates(query_embeddings)

        # Step 3: prune to a manageable candidate set using approximate scores.
        if num_candidates is None:
            num_candidates = max(top_k, int(len(candidate_set) * self.keep_ratio))

        sorted_candidates = sorted(
            candidate_set,
            key=lambda doc_id: approx_scores[doc_id],
            reverse=True,
        )
        pruned_docs = sorted_candidates[:num_candidates]

        # Step 4: exact MaxSim scoring on the pruned set (TileMaxSim).
        if self.scoring is not None:
            scores = self.scoring.score(query_embeddings, pruned_docs, documents)
        else:
            # Fallback placeholder exact scoring (for integration tests only).
            scores = self._placeholder_exact_scores(query_embeddings, pruned_docs)

        # Step 5: rank and return.
        ranked = sorted(zip(pruned_docs, scores), key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in ranked[:top_k]]

    def _generate_candidates(
        self, query_embeddings: np.ndarray
    ) -> Tuple[set, Dict[int, float]]:
        return generate_candidates(
            query_embeddings,
            self.scene,
            num_probe=self.num_probe,
            t_max_scale=self.t_max_scale,
        )

    def _placeholder_exact_scores(
        self, query_embeddings: np.ndarray, doc_ids: List[int]
    ) -> np.ndarray:
        """
        Compute exact L2‑distance‑style scores only for testing.

        This method is *not* the real TileMaxSim; it is intended to allow
        smoke tests before the actual scoring backend is integrated.
        """
        scores = np.zeros(len(doc_ids))
        for i, doc_id in enumerate(doc_ids):
            # In a real system we would access the document's token vectors
            # and compute MaxSim.  Here we use a dummy value to keep the
            # pipeline runnable.
            for qv in query_embeddings:
                # Approximate `max_dot` by the magnitude of `qv`; obviously
                # this is not a real relevance score.
                scores[i] += np.sum(qv) * (doc_id + 1)
        return scores
