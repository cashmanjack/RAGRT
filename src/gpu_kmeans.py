"""
Torch k-means and nearest-codeword assignment for the per-subspace codebooks.
Runs on the GPU when available (CPU works, for tests). Large E (up to 65536)
is handled by chunking the points so a chunk x E distance block stays ~256 MB.
Same algorithm as ragrt_index_lib.kmeans: k-means++ init on a subsample, Lloyd
iterations, empty clusters reseeded from the worst-fit points.
"""
import numpy as np
import torch


def _chunk(E, budget=1 << 26):
    return max(1024, budget // max(1, E))


@torch.no_grad()
def assign(x, centers):
    """x [n, d], centers [E, d] (torch, same device) -> (ids int64 [n], squared error [n])."""
    c_sq = (centers * centers).sum(1)
    n = x.shape[0]
    ids = torch.empty(n, dtype=torch.int64, device=x.device)
    err = torch.empty(n, dtype=torch.float32, device=x.device)
    step = _chunk(centers.shape[0])
    for a in range(0, n, step):
        xb = x[a:a + step]
        d = (xb * xb).sum(1, keepdim=True) - 2.0 * (xb @ centers.T) + c_sq[None, :]
        v, j = d.min(1)
        ids[a:a + step] = j
        err[a:a + step] = v.clamp_min(0)
    return ids, err


def assign_np(x, centers, device=None):
    """numpy in/out wrapper: (ids int32, err float32)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    xt = torch.as_tensor(np.ascontiguousarray(x, dtype=np.float32), device=device)
    ct = torch.as_tensor(np.ascontiguousarray(centers, dtype=np.float32), device=device)
    ids, err = assign(xt, ct)
    return ids.to(torch.int32).cpu().numpy(), err.cpu().numpy()


@torch.no_grad()
def kmeans(x, k, iters=25, seed=0, init_sample=200_000, device=None):
    """x numpy [n, d] -> centers numpy float32 [k, d]."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    x = np.asarray(x, dtype=np.float32)
    assert len(x) >= k, f"need at least k={k} points, got {len(x)}"
    rng = np.random.default_rng(seed)
    X = torch.as_tensor(x, device=device)

    pool = torch.as_tensor(x[rng.choice(len(x), size=min(max(init_sample, 4 * k), len(x)), replace=False)],
                           device=device)
    g = torch.Generator(device="cpu").manual_seed(seed)
    centers = torch.empty((k, x.shape[1]), dtype=torch.float32, device=device)
    centers[0] = pool[int(torch.randint(len(pool), (1,), generator=g))]
    d2 = ((pool - centers[0]) ** 2).sum(1)
    for j in range(1, k):
        tot = d2.sum()
        if float(tot) > 0:
            idx = int(torch.multinomial((d2 / tot).cpu(), 1, generator=g))
        else:
            idx = int(torch.randint(len(pool), (1,), generator=g))
        centers[j] = pool[idx]
        d2 = torch.minimum(d2, ((pool - centers[j]) ** 2).sum(1))

    for _ in range(iters):
        ids, err = assign(X, centers)
        counts = torch.bincount(ids, minlength=k)
        sums = torch.zeros((k, x.shape[1]), dtype=torch.float64, device=device)
        sums.index_add_(0, ids, X.double())
        nonempty = counts > 0
        centers[nonempty] = (sums[nonempty] / counts[nonempty, None]).float()
        empty = torch.nonzero(~nonempty).squeeze(1)
        if len(empty):
            worst = torch.topk(err, len(empty)).indices
            centers[empty] = X[worst]
    return centers.cpu().numpy()
