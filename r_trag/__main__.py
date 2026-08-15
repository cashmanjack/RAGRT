"""Example usage of the RTRAG package."""

from __future__ import annotations

import numpy as np

from .pipeline import RTRAG


def main() -> None:
    # Toy data: 4 centroids, 2D embeddings (for illustration only).
    centroids = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )
    # Each centroid points to one or two fictitious documents.
    doc_lists = [
        [0, 1],
        [1, 2],
        [2, 3],
        [0, 3],
    ]

    rag = RTRAG(metric="l2", radius=1.0, num_probe=4, keep_ratio=1.0)
    rag.build_index(centroids, doc_lists)

    query = np.array(
        [
            [0.2, 0.2],
            [0.8, 0.8],
        ]
    )

    result = rag.retrieve(query, top_k=2, documents=None, doc_ids=[0, 1, 2, 3])
    print("Top‑2 documents:", result)


if __name__ == "__main__":
    main()
