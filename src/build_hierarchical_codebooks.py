import os, numpy as np
from sklearn.cluster import KMeans

BASE_DIR = "/local/scratch/a/cashman3/juno_pq_lotte_full_sparse8"
CODEBOOKS_PATH = os.path.join(BASE_DIR, "codebooks.npy")

print("=" * 90)
print("BUILDING 2-LEVEL 3D CODEBOOK HIERARCHY (32 SUBSPACES x 16 GROUPS x 16 CODEWORDS)")
print("=" * 90)

codebooks = np.load(CODEBOOKS_PATH)  # [32, 256, 3]
NUM_SUBSPACES = codebooks.shape[0]  # 32
NUM_FINE      = codebooks.shape[1]  # 256
NUM_GROUPS    = 16                  # 16 super-groups per subspace

super_codewords = np.zeros((NUM_SUBSPACES, NUM_GROUPS, 3), dtype=np.float32)
group_radius    = np.zeros((NUM_SUBSPACES, NUM_GROUPS), dtype=np.float32)
group_assign    = np.zeros((NUM_SUBSPACES, NUM_FINE), dtype=np.int32)
group_members   = np.zeros((NUM_SUBSPACES, NUM_GROUPS, 16), dtype=np.int32)

for s in range(NUM_SUBSPACES):
    fine_vecs = codebooks[s]  # [256, 3]
    
    # 1. K-Means clustering into 16 balanced groups
    kmeans = KMeans(n_clusters=NUM_GROUPS, n_init=10, random_state=42)
    labels = kmeans.fit_predict(fine_vecs)
    centers = kmeans.cluster_centers_  # [16, 3]
    
    super_codewords[s] = centers
    group_assign[s] = labels
    
    # 2. Compute bounding radius per group
    for g in range(NUM_GROUPS):
        members = np.where(labels == g)[0]
        if len(members) == 0:
            # Fallback if an empty cluster occurs
            group_radius[s, g] = 0.0
            continue
            
        diffs = fine_vecs[members] - centers[g]
        dists = np.linalg.norm(diffs, axis=1)
        group_radius[s, g] = np.max(dists)

print("\nHierarchy Summary:")
print(f"  Super-Codewords Shape : {super_codewords.shape} (dtype: float32)")
print(f"  Group Radii Shape     : {group_radius.shape}    (Mean Radius: {np.mean(group_radius):.4f})")
print(f"  Group Assign Shape    : {group_assign.shape}")

# Save hierarchy arrays
out_sc = os.path.join(BASE_DIR, "super_codewords.npy")
out_gr = os.path.join(BASE_DIR, "group_radius.npy")
out_ga = os.path.join(BASE_DIR, "group_assign.npy")

np.save(out_sc, super_codewords)
np.save(out_gr, group_radius)
np.save(out_ga, group_assign)

print(f"\n[DONE] Saved:")
print(f"  --> {out_sc}")
print(f"  --> {out_gr}")
print(f"  --> {out_ga}")
print("=" * 90)
