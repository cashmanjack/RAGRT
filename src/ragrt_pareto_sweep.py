"""
Final Standardized Pareto Sweep for 256 Codewords with Adaptive Silicon Pruning (tau=0.10).
"""
import os, sys, time, csv
import numpy as np
import torch

BASE_DIR         = "/home/min/a/cashman3/RTRAG/src"
INDEX_PATH       = "/home/min/a/cashman3/RTRAG/src/experiments/unified_lotte/indexes/unified.dev.2bit"
COLLECTION_PATH  = "/local/scratch/a/cashman3/lotte/unified/collection.tsv"
SPARSE_CSR_DIR   = "/local/scratch/a/cashman3/juno_pq_lotte_full_sparse8"
PTX_PATH         = os.path.join(BASE_DIR, "optix_corr_3d.ptx")
QUESTIONS_PATH   = "/home/min/a/cashman3/RTRAG/src/experiments/lotte_science_eval/questions.tsv"
QRELS_PATH       = "/home/min/a/cashman3/RTRAG/src/experiments/lotte_science_eval/qrels.tsv"
TOP_K            = 100
N_COARSE         = 64

CONFIGS = [
    (4096,  16),
    (16384, 16),
    (32768, 16),
    (65536, 16),
]

sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "../reference/colbert-plaid"))

from colbert import Searcher
from fast_tilemaxsim_scorer import FastTileMaxSimScorer
import rtrag_corr_3d

def load_questions(path, limit=500):
    qs = []
    with open(path) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2: qs.append((p[0], p[1]))
            if len(qs) >= limit: break
    return qs

def load_qrels(path):
    qrels = {}
    with open(path) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 4 and float(p[3]) > 0:
                qrels.setdefault(p[0], set()).add(int(p[2]))
    return qrels

def recall_at_k(retrieved, relevant, k=10):
    if not relevant: return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)

def mrr_at_k(retrieved, relevant, k=10):
    for i, pid in enumerate(retrieved[:k]):
        if pid in relevant: return 1.0 / (i + 1)
    return 0.0

def median_lat(fn, warmup=30, timed=100):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(timed):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    return float(np.median(ts))

def main():
    print("=" * 110)
    print("RAGRT FINAL PARETO SWEEP (256 CODEWORDS + ADAPTIVE PRUNING tau=0.10, N=2.43M PASSAGES)")
    print("=" * 110)

    searcher = Searcher(index=INDEX_PATH, collection=COLLECTION_PATH)
    scorer = FastTileMaxSimScorer(searcher)
    N = len(searcher.ranker.doclens)

    cb = np.load(os.path.join(SPARSE_CSR_DIR, "codebooks.npy"))
    sc = np.load(os.path.join(SPARSE_CSR_DIR, "super_codewords.npy"))
    gr = np.load(os.path.join(SPARSE_CSR_DIR, "group_radius.npy"))
    ga = np.load(os.path.join(SPARSE_CSR_DIR, "group_assign.npy"))

    print("[setup] Building OptiX 256 BVHs...", flush=True)
    unified_idx = rtrag_corr_3d.CorrIndex3D(PTX_PATH)
    unified_idx.build(torch.tensor(cb, dtype=torch.float32).contiguous(), 0.95, 4)
    unified_idx.bind_hierarchy(torch.tensor(sc).float(), torch.tensor(gr).float(), torch.tensor(ga).int())

    csr_row_ptrs = torch.from_numpy(np.load(os.path.join(SPARSE_CSR_DIR, "csr_row_ptrs.npy")).astype(np.int32)).cuda()
    csr_col_eids = torch.from_numpy(np.load(os.path.join(SPARSE_CSR_DIR, "csr_col_eids.npy"))).cuda()
    csr_offsets  = torch.from_numpy(np.load(os.path.join(SPARSE_CSR_DIR, "csr_offsets.npy")).astype(np.int64)).cuda()
    csr_lengths  = torch.from_numpy(np.load(os.path.join(SPARSE_CSR_DIR, "csr_lengths.npy"))).cuda()
    map_packed   = torch.from_numpy(np.load(os.path.join(SPARSE_CSR_DIR, "csr_packed_24.npy"))).cuda()

    centroids_128  = searcher.ranker.codec.centroids.detach().cuda().to(torch.float16).contiguous()
    centroids_96   = torch.from_numpy(np.load(os.path.join(SPARSE_CSR_DIR, "centroids_96d_svd.npy"))).cuda().float()
    R_proj         = torch.from_numpy(np.load(os.path.join(SPARSE_CSR_DIR, "svd_rotation_128_to_96.npy"))).cuda().float()
    doc_predicates = torch.from_numpy(np.load(os.path.join(SPARSE_CSR_DIR, "doc_predicates.npy"))).cuda()

    unified_idx.bind_index(
        csr_row_ptrs, csr_col_eids, csr_offsets, csr_lengths, map_packed,
        scorer.doc_offsets[:N+1], scorer.doclens[:N], scorer.codes, scorer.residuals,
        centroids_128, scorer.bucket_weights, scorer.reversed_bit_map, scorer.decomp_table,
        doc_predicates[:N], N, centroids_96.shape[0])

    questions = load_questions(QUESTIONS_PATH, 600)
    qrels = load_qrels(QRELS_PATH)
    q_with_rel = [(qid, q) for (qid, q) in questions if qid in qrels][:500]
    qual_Q = [searcher.encode(q).squeeze(0).cuda() for _, q in q_with_rel]
    nc = min(N_COARSE, int(centroids_96.shape[0]))

    Q_full_128_list, Q_full_fp16_list, Q_sub3d_list, topc_list, scores_list = [], [], [], [], []
    for (qid, q), Qf in zip(q_with_rel, qual_Q):
        ntok = min(len(q.split()) + 4, 32)
        Q_act = Qf[:ntok, :].contiguous()
        Q_96 = torch.nn.functional.normalize(Q_act @ R_proj, p=2, dim=-1)
        Q_sub = Q_96.view(ntok, 32, 3).contiguous()
        scores = Q_96 @ centroids_96.T
        topc = scores.topk(k=nc, dim=-1).indices.contiguous().to(torch.int32)

        Q_full_128_list.append(Q_act.float().contiguous())
        Q_full_fp16_list.append(Qf.to(torch.float16).contiguous())
        Q_sub3d_list.append(Q_sub)
        topc_list.append(topc)
        scores_list.append(scores)

    print(f"\n{'Config':<25} | {'Seq Latency':<12} | {'Pipe Latency':<13} | {'Speedup':<9} | {'Recall@10':<10} | {'MRR@10':<8}")
    print("-" * 88)

    for k_pool, k_eids in CONFIGS:
        tag = f"pool={k_pool} eids={k_eids}"

        # 1. Sequential Latency
        def run_seq():
            return unified_idx.search_single_query_native(
                Q_full_128_list[0], Q_full_fp16_list[0], Q_sub3d_list[0], topc_list[0], scores_list[0],
                nc, k_pool, TOP_K, 0, 0, k_eids, 0, 0.10)
        seq_lat = median_lat(run_seq)

        # 2. Pipelined Latency (Batch size = 20)
        bs = min(20, len(q_with_rel))
        def run_pipe():
            return unified_idx.search_batch_pipelined(
                Q_full_128_list[:bs], Q_full_fp16_list[:bs], Q_sub3d_list[:bs],
                topc_list[:bs], scores_list[:bs],
                nc, k_pool, TOP_K, 0, 0, k_eids, 0, 0.10)
        pipe_lat = median_lat(run_pipe, warmup=10, timed=30) / float(bs)

        # 3. Quality Evaluation across 500 Queries
        ranked_all = unified_idx.search_batch_pipelined(
            Q_full_128_list, Q_full_fp16_list, Q_sub3d_list, topc_list, scores_list,
            nc, k_pool, TOP_K, 0, 0, k_eids, 0, 0.10)

        recalls, mrrs = [], []
        for i, (qid, _) in enumerate(q_with_rel):
            pids = ranked_all[i].to(torch.int64).cpu().tolist()
            rel = qrels[qid]
            recalls.append(recall_at_k(pids, rel, 10))
            mrrs.append(mrr_at_k(pids, rel, 10))

        r10, m10 = float(np.mean(recalls)), float(np.mean(mrrs))
        pipe_sp = 5.13 / pipe_lat
        print(f"{tag:<25} | {seq_lat:6.2f} ms     | {pipe_lat:6.2f} ms      | {pipe_sp:5.2f}x    | {r10:5.4f}     | {m10:.4f}")

    print("-" * 88)

if __name__ == "__main__":
    main()
