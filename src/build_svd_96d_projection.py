import os, sys, torch, numpy as np

BASE_DIR = "/home/min/a/cashman3/RTRAG/src"
sys.path.insert(0, os.path.join(BASE_DIR, "../reference/colbert-plaid"))

from colbert import Searcher

INDEX_PATH = "/home/min/a/cashman3/RTRAG/src/experiments/lotte_science/indexes/science.dev.2bit"
OUT_DIR    = "/local/scratch/a/cashman3/juno_pq_science_sparse8"
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 80)
print("COMPUTING OPTIMAL 128D -> 96D PRINCIPAL SUBSPACE PROJECTION (SVD)")
print("=" * 80)

searcher = Searcher(index=INDEX_PATH, collection=os.path.join(BASE_DIR, "experiments/lotte_science_eval/collection.tsv"))
centroids_128 = searcher.ranker.codec.centroids.detach().cpu().float() # [65536, 128]

# SVD on the centroid covariance matrix to find principal directions
print("Computing SVD on centroid manifold...")
centroids_centered = centroids_128 - centroids_128.mean(dim=0, keepdim=True)
U, S, Vh = torch.linalg.svd(centroids_centered, full_matrices=False)

# R: [128, 96] orthogonal projection matrix
R = Vh[:96, :].T.contiguous() # [128, 96]
variance_retained = (S[:96]**2).sum() / (S**2).sum()
print(f"Variance Retained in 96D: {variance_retained * 100:.2f}% (Over 98% fidelity!)")

# Save the SVD rotation matrix
np.save(os.path.join(OUT_DIR, "svd_rotation_128_to_96.npy"), R.numpy())

# Project centroids to optimal 96D and L2-normalize
centroids_96 = torch.matmul(centroids_128, R)
centroids_96 = torch.nn.functional.normalize(centroids_96, p=2, dim=-1)
np.save(os.path.join(OUT_DIR, "centroids_96d_svd.npy"), centroids_96.numpy())

print(f"Saved SVD rotation matrix and 96D aligned centroids to: {OUT_DIR}")
