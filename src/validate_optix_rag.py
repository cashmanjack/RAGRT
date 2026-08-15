"""
Brute-force reference check for optix_rag.cu's hardcoded test scene.
Run this, run the compiled optix_rag binary, and diff the outputs by hand
(hit_count per subspace, and the set of reconstructed distances) before
trusting any RT-core result. Values here MUST match h_spheres and
h_query_subspaces in optix_rag.cu exactly -- if you change the .cu test
data, update this file too.
"""

import math

RADIUS = 1.0

spheres = {
    0: [(0.0, 0.0), (0.5, 0.5), (-0.5, 0.5), (0.5, -0.5), (1.5, 0.0)],
    1: [(0.0, 0.0), (-0.5, -0.5), (0.5, -0.5), (-0.5, 0.5), (1.5, 0.0)],
}

queries = {
    0: (0.2, 0.1),
    1: (-0.2, -0.2),
}


def l2(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


for subspace, centroids in spheres.items():
    qx, qy = queries[subspace]
    hits = []
    for cx, cy in centroids:
        d = l2((qx, qy), (cx, cy))
        if d <= RADIUS:
            hits.append(d)
    hits.sort()
    print(f"subspace {subspace}: hit_count={len(hits)}")
    for i, d in enumerate(hits):
        print(f"  hit {i} distance = {d:.6f}")
