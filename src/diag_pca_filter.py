"""
Is a low-dimensional PCA projection a usable Stage-1 filter? (Offline diagnostic, GPU.)

Idea being tested: project 128D ColBERT vectors to m dims (m = 3 fits an RT scene
directly) and do the nearest-neighbour search there, as a cheap first filter.
Inner products survive projection only partially, so the question is purely
empirical: how big a shortlist in m dims is needed to keep the true neighbours?

Two filters, each with two PCA bases (fit on the centroids, or on a doc-token sample):
  A. centroid probing: per query token, recall of the exact top-nc centroids
     (nc = 8, 32) inside the top (alpha * nc) centroids by m-dim score.
  B. token kNN: per query token, recall of the exact top-K doc tokens (K = 10, 100,
     from a random sample of doc tokens) inside the top (alpha * K) by m-dim score.
Scores use q.c = q.mu + (P q).(P (c - mu)), so centering does not bias the ranking.
Also reports the fraction of doc-token variance each m keeps.

For reference, RAGRT already uses 96 dims split into 32 subspaces of 3: a single
global 3D projection is a much stronger reduction than anything in the current system.

  python3 diag_pca_filter.py --dataset lotte [--n_queries 300 --doc_tokens 2000000]
Writes <results>/<dataset>/diag_pca_filter.json
"""
import os, sys, json, argparse
import numpy as np
import torch

import eval_config as C
import eval_lib as E
from engines import Engines
from colbert.indexing.codecs.residual_embeddings import ResidualEmbeddings

DIMS = [3, 8, 16, 32, 64, 128]
ALPHAS = [1, 4, 16, 64]


def pca_basis(X):
    mu = X.mean(dim=0, keepdim=True)
    _, S, Vh = torch.linalg.svd((X - mu).double(), full_matrices=False)
    var = (S ** 2) / (S ** 2).sum()
    return mu.float(), Vh.float(), var.float()          # rows of Vh are components


@torch.no_grad()
def shortlist_recall(Q, X, mu, V, m, ks, alphas, chunk=256):
    """mean recall of exact top-k (by Q.X) inside approx top-(alpha k), for each k and alpha."""
    P = V[:m].T                                           # [128, m]
    Xp = (X - mu) @ P
    kmax = max(ks)
    amax = max(alphas) * kmax
    hits = {(k, a): 0.0 for k in ks for a in alphas}
    for i in range(0, len(Q), chunk):
        q = Q[i:i + chunk]
        exact = torch.topk(q @ X.T, kmax, dim=1).indices                    # [b, kmax]
        approx = torch.topk((q @ P) @ Xp.T, min(amax, X.shape[0]), dim=1).indices
        for k in ks:
            for a in alphas:
                short = approx[:, :a * k]
                # membership of each exact neighbour in the shortlist
                found = (exact[:, :k, None] == short[:, None, :]).any(dim=2).float().mean(dim=1)
                hits[(k, a)] += float(found.sum())
    return {f"k={k} alpha={a}": hits[(k, a)] / len(Q) for k in ks for a in alphas}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(C.DATASETS))
    ap.add_argument("--n_queries", type=int, default=300)
    ap.add_argument("--doc_tokens", type=int, default=2_000_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    ds = C.dataset(args.dataset)
    eng = Engines(ds, load_ragrt=False)

    qrels = E.load_qrels(ds["qrels_path"])
    qs = [(q, t) for q, t in E.load_questions(ds["questions_path"]) if q in qrels][:args.n_queries]
    toks = []
    for _, text in qs:
        Qf, n = eng.encode(text)
        toks.append(Qf[:n].float())
    Q = torch.nn.functional.normalize(torch.cat(toks), dim=-1)
    print(f"{len(qs)} queries -> {len(Q):,} query tokens")

    Cn = eng.ranker.codec.centroids.cuda().float()
    total = int(eng.doc_offsets[-1])
    rng = np.random.default_rng(args.seed)
    idx = torch.from_numpy(np.sort(rng.choice(total, size=min(args.doc_tokens, total), replace=False)))
    emb = eng.ranker.embeddings
    D = []
    for a in range(0, len(idx), 500_000):
        b = idx[a:a + 500_000]
        obj = ResidualEmbeddings(emb.codes[b], emb.residuals[b])
        D.append(eng.ranker.codec.decompress(obj).cuda().float())
    D = torch.nn.functional.normalize(torch.cat(D), dim=-1)
    print(f"sampled {len(D):,} doc tokens of {total:,}; {len(Cn):,} centroids")

    out = {"n_query_tokens": len(Q), "n_doc_tokens": len(D), "n_centroids": len(Cn), "results": {}}
    bases = {"centroid_pca": pca_basis(Cn), "token_pca": pca_basis(D)}
    for bname, (mu, V, _) in bases.items():
        Dc = (D - mu) @ V.T
        tot = float((Dc ** 2).sum())
        for m in DIMS:
            r = {"doc_variance_kept": float((Dc[:, :m] ** 2).sum()) / tot}
            r["centroid_probe"] = shortlist_recall(Q, Cn, mu, V, m, [8, 32], ALPHAS)
            r["token_knn"] = shortlist_recall(Q, D, mu, V, m, [10, 100], ALPHAS)
            out["results"][f"{bname} m={m}"] = r
            print(f"{bname:<13} m={m:<3} var {r['doc_variance_kept']:.2f} | centroid top-8: "
                  + " ".join(f"a{a}={r['centroid_probe'][f'k=8 alpha={a}']:.2f}" for a in ALPHAS)
                  + " | token top-10: "
                  + " ".join(f"a{a}={r['token_knn'][f'k=10 alpha={a}']:.2f}" for a in ALPHAS), flush=True)
    os.makedirs(ds["results_dir"], exist_ok=True)
    path = os.path.join(ds["results_dir"], "diag_pca_filter.json")
    E.dump_json(out, path)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
