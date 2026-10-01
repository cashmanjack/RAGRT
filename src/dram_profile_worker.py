"""
DRAM profile worker: Runs N queries for a single engine with CUDA profiler markers.
Called by measure_dram.py via: nsys profile --capture-range=cudaProfilerApi python3 dram_profile_worker.py <engine>
Engines: plaid, plaid_tms, ragrt
"""
import os, sys, time
import numpy as np
import torch
from math import ceil

BASE_DIR         = "/home/min/a/cashman3/RTRAG/src"
INDEX_PATH       = "/home/min/a/cashman3/RTRAG/src/experiments/unified_lotte/indexes/unified.dev.2bit"
COLLECTION_PATH  = "/local/scratch/a/cashman3/lotte/unified/collection.tsv"
SPARSE_CSR_DIR   = "/local/scratch/a/cashman3/juno_pq_lotte_full_sparse8"
PTX_PATH         = os.path.join(BASE_DIR, "optix_corr_3d.ptx")
QUESTIONS_PATH   = "/home/min/a/cashman3/RTRAG/src/experiments/lotte_science_eval/questions.tsv"

TOP_K = 100
K_CANDIDATES = 4096
N_COARSE = 32
K_EIDS = 16
NUM_QUERIES = 20

sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "../reference/colbert-plaid"))

from colbert import Searcher
from fast_tilemaxsim_scorer import FastTileMaxSimScorer
import rtrag_corr_3d


def get_stage3_pids(ranker, config, Q):
    """Native PLAID Stage 1-3 candidate generation."""
    with torch.inference_mode():
        pids, _ = ranker.retrieve(config, Q)
        if isinstance(pids, list):
            pids = torch.tensor(pids, dtype=torch.int32, device='cuda')
        return pids


def load_questions(path, limit=30):
    qs = []
    with open(path) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2: qs.append((p[0], p[1]))
            if len(qs) >= limit: break
    return qs


def main():
    engine = sys.argv[1] if len(sys.argv) > 1 else 'ragrt'
    print(f"[worker] Engine: {engine}", flush=True)

    searcher = Searcher(index=INDEX_PATH, collection=COLLECTION_PATH)
    scorer = FastTileMaxSimScorer(searcher)
    N = len(searcher.ranker.doclens)
    searcher.config.ncells = 2
    searcher.config.centroid_score_threshold = 0.45
    searcher.config.ndocs = K_CANDIDATES

    cb = np.load(os.path.join(SPARSE_CSR_DIR, "codebooks.npy"))
    unified_idx = rtrag_corr_3d.CorrIndex3D(PTX_PATH)
    unified_idx.build(torch.tensor(cb, dtype=torch.float32).contiguous(), 0.95, 4)

    csr_row_ptrs = torch.from_numpy(np.load(os.path.join(SPARSE_CSR_DIR, "csr_row_ptrs.npy")).astype(np.int32)).cuda()
    csr_col_eids = torch.from_numpy(np.load(os.path.join(SPARSE_CSR_DIR, "csr_col_eids.npy"))).cuda()
    csr_offsets  = torch.from_numpy(np.load(os.path.join(SPARSE_CSR_DIR, "csr_offsets.npy")).astype(np.int64)).cuda()
    csr_lengths  = torch.from_numpy(np.load(os.path.join(SPARSE_CSR_DIR, "csr_lengths.npy"))).cuda()
    map_packed   = torch.from_numpy(np.load(os.path.join(SPARSE_CSR_DIR, "csr_packed_24.npy"))).cuda()
    centroids_128  = searcher.ranker.codec.centroids.detach().cuda().to(torch.float16).contiguous()
    centroids_96   = torch.from_numpy(np.load(os.path.join(SPARSE_CSR_DIR, "centroids_96d_svd.npy"))).cuda().float()
    R_proj         = torch.from_numpy(np.load(os.path.join(SPARSE_CSR_DIR, "svd_rotation_128_to_96.npy"))).cuda().float()
    doc_predicates = torch.from_numpy(np.load(os.path.join(SPARSE_CSR_DIR, "doc_predicates.npy"))).cuda().to(torch.int64)

    unified_idx.bind_index(
        csr_row_ptrs, csr_col_eids, csr_offsets, csr_lengths, map_packed,
        scorer.doc_offsets[:N+1], scorer.doclens[:N], scorer.codes, scorer.residuals,
        centroids_128, scorer.bucket_weights, scorer.reversed_bit_map, scorer.decomp_table,
        doc_predicates.to(torch.uint32), N, centroids_96.shape[0])

    questions = load_questions(QUESTIONS_PATH, NUM_QUERIES + 5)
    Q_list = []
    for qid, q in questions[:NUM_QUERIES]:
        Qf = searcher.encode(q).squeeze(0).cuda()
        ntok = min(len(q.split()) + 4, 32)
        Q_act = Qf[:ntok, :].contiguous()
        Q_96 = torch.nn.functional.normalize(Q_act @ R_proj, p=2, dim=-1)
        Q_sub = Q_96.view(ntok, 32, 3).contiguous()
        scores = Q_96 @ centroids_96.T
        topc = scores.topk(k=N_COARSE, dim=-1).indices.contiguous().to(torch.int32)
        Q_list.append((Qf.unsqueeze(0), Qf.float().contiguous(), Qf.to(torch.float16).contiguous(),
                       Q_sub, topc, scores))

    # Warmup
    for _ in range(3):
        if engine == 'plaid':
            _ = searcher.ranker.rank(searcher.config, Q_list[0][0])
        elif engine == 'plaid_tms':
            pids_s3 = get_stage3_pids(searcher.ranker, searcher.config, Q_list[0][0])
            _ = scorer.score(Q_list[0][1], pids_s3)
        else:
            _ = unified_idx.search_single_query_native(
                Q_list[0][1], Q_list[0][2], Q_list[0][3], Q_list[0][4], Q_list[0][5],
                N_COARSE, K_CANDIDATES, TOP_K, 0, 0, K_EIDS)
    torch.cuda.synchronize()

    torch.cuda.profiler.start()
    for i in range(NUM_QUERIES):
        Qb, Qf128, Qfp16, Qsub, topc, scores = Q_list[i]
        if engine == 'plaid':
            _ = searcher.ranker.rank(searcher.config, Qb)
        elif engine == 'plaid_tms':
            pids_s3 = get_stage3_pids(searcher.ranker, searcher.config, Qb)
            _ = scorer.score(Qf128, pids_s3)
        else:
            _ = unified_idx.search_single_query_native(
                Qf128, Qfp16, Qsub, topc, scores,
                N_COARSE, K_CANDIDATES, TOP_K, 0, 0, K_EIDS)

    torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    print(f"[worker] Done.", flush=True)


if __name__ == '__main__':
    main()
