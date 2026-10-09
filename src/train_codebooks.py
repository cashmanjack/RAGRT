"""
Train the per-subspace fine codebooks for ONE ColBERT index.

For each of the 32 three-dimensional subspaces, runs k-means on that index's own
residual slices, in that index's own 128->96 rotation, using only the slices the
CSR builder will actually index (same top_m rule, ragrt_index_lib.top_m_active_mask).

Outputs in --outdir:
  svd_rotation_128_to_96.npy   created if missing (reused if present)
  codebooks.npy                float32 [32, E, 3]
  codebook_meta.json           rotation fingerprint, settings, quantization error

Quantization error is reported on a held-out split of the sampled tokens:
  mse      mean squared error of a slice vs its nearest codeword
  rel_err  mse / mean squared slice norm  (fraction of slice energy lost)

Usage:
  python3 train_codebooks.py --index <colbert index> --collection <tsv> --outdir <dir>
"""
import os, sys, json, time, argparse
import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "../reference/colbert-plaid"))

from colbert import Searcher
from colbert.indexing.codecs.residual_embeddings import ResidualEmbeddings

import ragrt_index_lib as L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--collection", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--num_entries", type=int, default=256, help="codewords per subspace (E)")
    ap.add_argument("--top_m", type=int, default=10, help="must match the CSR builder")
    ap.add_argument("--sample_tokens", type=int, default=4_000_000)
    ap.add_argument("--holdout_frac", type=float, default=0.1)
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    t0 = time.time()

    searcher = Searcher(index=args.index, collection=args.collection)
    centroids_128 = searcher.ranker.codec.centroids.detach().cpu().float().numpy()

    rot_path = os.path.join(args.outdir, "svd_rotation_128_to_96.npy")
    if os.path.exists(rot_path):
        R = np.load(rot_path).astype(np.float32)
        print(f"Reusing rotation {rot_path}")
    else:
        R = L.compute_rotation(centroids_128)
        np.save(rot_path, R)
        print(f"Computed rotation from {centroids_128.shape[0]:,} centroids -> {rot_path}")
    centroids_96 = L.l2_normalize(centroids_128 @ R)

    codes = searcher.ranker.embeddings.codes.numpy()
    residuals_packed = searcher.ranker.embeddings.residuals.numpy()
    total = codes.shape[0]
    rng = np.random.default_rng(args.seed)
    n = min(args.sample_tokens, total)
    idx = np.sort(rng.choice(total, size=n, replace=False))
    print(f"Sampling {n:,} of {total:,} tokens")

    R_cuda = torch.from_numpy(R).cuda()
    slices = [[] for _ in range(L.NUM_SUBSPACES)]
    for a in range(0, n, 1_000_000):
        b = idx[a:a + 1_000_000]
        obj = ResidualEmbeddings(torch.from_numpy(codes[b]), torch.from_numpy(residuals_packed[b]))
        emb_128 = searcher.ranker.codec.decompress(obj).cuda().float()
        emb_96 = torch.nn.functional.normalize(emb_128 @ R_cuda, p=2, dim=-1).cpu().numpy()
        mask = L.top_m_active_mask(emb_96, args.top_m)
        resid = L.residual_96(emb_96, centroids_96, codes[b])
        for s in range(L.NUM_SUBSPACES):
            slices[s].append(L.subspace_slice(resid[mask[:, s]], s))

    E = args.num_entries
    codebooks = np.zeros((L.NUM_SUBSPACES, E, L.SUBSPACE_DIM), dtype=np.float32)
    per_sub = []
    for s in range(L.NUM_SUBSPACES):
        x = np.concatenate(slices[s]); slices[s] = None
        perm = rng.permutation(len(x))
        n_hold = int(len(x) * args.holdout_frac)
        hold, train = x[perm[:n_hold]], x[perm[n_hold:]]
        codebooks[s] = L.kmeans(train, E, iters=args.iters, seed=args.seed + s)
        _, err = L.assign_nearest(hold, codebooks[s])
        energy = float(np.square(hold).sum(axis=1).mean())
        mse = float(err.mean())
        norms = np.linalg.norm(codebooks[s], axis=1)
        per_sub.append({"subspace": s, "train_slices": int(len(train)), "mse": mse,
                        "rel_err": mse / energy if energy > 0 else 0.0,
                        "codeword_norm_median": float(np.median(norms)),
                        "codeword_norm_max": float(norms.max())})
        print(f"  s={s:2d}  slices={len(train):>10,}  mse={mse:.3e}  rel_err={per_sub[-1]['rel_err']:.3f}  "
              f"|c| median={per_sub[-1]['codeword_norm_median']:.3f}")

    np.save(os.path.join(args.outdir, "codebooks.npy"), codebooks)
    meta = {
        "index": os.path.abspath(args.index),
        "rotation_sha256": L.rotation_fingerprint(R),
        "num_entries": E, "top_m": args.top_m,
        "sample_tokens": int(n), "holdout_frac": args.holdout_frac,
        "iters": args.iters, "seed": args.seed,
        "mean_mse": float(np.mean([p["mse"] for p in per_sub])),
        "mean_rel_err": float(np.mean([p["rel_err"] for p in per_sub])),
        "per_subspace": per_sub,
        "seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(args.outdir, "codebook_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[DONE] codebooks [{L.NUM_SUBSPACES}, {E}, 3]  mean mse={meta['mean_mse']:.3e}  "
          f"mean rel_err={meta['mean_rel_err']:.3f}  -> {args.outdir}")


if __name__ == "__main__":
    main()
