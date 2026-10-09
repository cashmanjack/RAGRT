"""
DRAM profile worker: Runs N queries for a single engine with CUDA profiler markers.
Automatically reads the Pareto-selected configurations from selected_configs_{dataset}.json.
"""
import os, sys, time, json
import numpy as np
import torch
from math import ceil

BASE_DIR        = "/home/min/a/cashman3/RTRAG/src"
PTX_PATH        = os.path.join(BASE_DIR, "optix_corr_3d.ptx")
TOP_K           = 100
NUM_QUERIES     = 20

# Determine dataset from environment or argument (default: lotte)
DATASET = os.environ.get("DATASET", "lotte")
if "--dataset" in sys.argv:
    DATASET = sys.argv[sys.argv.index("--dataset") + 1]
elif len(sys.argv) > 2 and sys.argv[2] in ['lotte', 'msmarco']:
    DATASET = sys.argv[2]
assert DATASET in ('lotte', 'msmarco'), f"bad dataset: {DATASET}"

sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "../reference/colbert-plaid"))

from colbert import Searcher
from fast_tilemaxsim_scorer import FastTileMaxSimScorer
from colbert.search.strided_tensor import StridedTensor
from colbert.modeling.colbert import colbert_score_reduce
import rtrag_corr_3d
import ragrt_index_lib as L

# Load dataset paths
if DATASET == 'msmarco':
    INDEX_PATH      = "/local/scratch/a/cashman3/msmarco/indexes/msmarco.dev.2bit"
    COLLECTION_PATH = "/local/scratch/a/cashman3/msmarco/collection.tsv"
    SPARSE_CSR_DIR  = "/local/scratch/a/cashman3/msmarco/msmarco_sparse8"
    QUESTIONS_PATH  = "/local/scratch/a/cashman3/msmarco/queries.dev.small.tsv"
else:
    INDEX_PATH      = "/home/min/a/cashman3/RTRAG/src/experiments/unified_lotte/indexes/unified.dev.2bit"
    COLLECTION_PATH = "/local/scratch/a/cashman3/lotte/unified/collection.tsv"
    SPARSE_CSR_DIR  = "/local/scratch/a/cashman3/juno_pq_lotte_full_sparse8"
    QUESTIONS_PATH  = "/home/min/a/cashman3/RTRAG/src/experiments/lotte_science_eval/questions.tsv"

# Load auto-selected operating points if available
CONFIG_FILE = os.path.join(BASE_DIR, DATASET, f"selected_configs_{DATASET}.json")
selected_params = {
    'plaid': {'ncells': 2, 'ndocs': 4096},
    'ptms':  {'ncells': 2, 'ndocs': 4096},
    'ragrt': {'nc': 32, 'ndocs': 4096, 'eids': 16}
}

if not os.path.exists(CONFIG_FILE):
    raise FileNotFoundError(
        f"[worker] FATAL: {CONFIG_FILE} not found. "
        "Run generate_figures.py (pareto selection) for this dataset first; "
        "refusing to profile non-selected fallback configs.")
with open(CONFIG_FILE) as f:
    data = json.load(f)
    for k in ['plaid', 'ptms', 'ragrt']:
        if k in data:
            selected_params[k] = data[k]['params']
print(f"[worker] Loaded Pareto-selected configs from {CONFIG_FILE}")


def get_stage3_pids(ranker, config, Q):
    """Native single-pass PLAID Stage 1-3 candidate generation."""
    with torch.inference_mode():
        pids, centroid_scores = ranker.retrieve(config, Q)
        if isinstance(pids, list):
            pids = torch.tensor(pids, dtype=torch.int32, device='cuda')
        batch_size = 2 ** 20
        if centroid_scores is not None and ranker.use_gpu:
            centroid_scores = centroid_scores.cuda()
            idx = centroid_scores.max(-1).values >= config.centroid_score_threshold
            approx_scores = []
            for i in range(0, ceil(len(pids) / batch_size)):
                pids_ = pids[i * batch_size : (i+1) * batch_size]
                codes_packed, codes_lengths = ranker.embeddings_strided.lookup_codes(pids_)
                idx_ = idx[codes_packed.long()]
                pruned_codes_strided = StridedTensor(idx_, codes_lengths, use_gpu=ranker.use_gpu)
                pruned_codes_padded, pruned_codes_mask = pruned_codes_strided.as_padded_tensor()
                pruned_codes_lengths = (pruned_codes_padded * pruned_codes_mask).sum(dim=1)
                codes_packed_ = codes_packed[idx_]
                approx_scores_ = centroid_scores[codes_packed_.long()]
                if approx_scores_.shape[0] == 0:
                    approx_scores.append(torch.zeros((len(pids_),), dtype=approx_scores_.dtype).cuda())
                    continue
                approx_scores_strided = StridedTensor(approx_scores_, pruned_codes_lengths, use_gpu=ranker.use_gpu)
                approx_scores_padded, approx_scores_mask = approx_scores_strided.as_padded_tensor()
                approx_scores_ = colbert_score_reduce(approx_scores_padded, approx_scores_mask, config)
                approx_scores.append(approx_scores_)
            approx_scores = torch.cat(approx_scores, dim=0)
            if config.ndocs // 4 < len(approx_scores):
                pids = pids[torch.topk(approx_scores, k=(config.ndocs // 4)).indices]
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
    print(f"[worker] Engine: {engine} | Dataset: {DATASET}", flush=True)

    searcher = Searcher(index=INDEX_PATH, collection=COLLECTION_PATH)
    scorer = FastTileMaxSimScorer(searcher)
    N = len(searcher.ranker.doclens)

    cb = np.load(os.path.join(SPARSE_CSR_DIR, "codebooks.npy"))
    unified_idx = rtrag_corr_3d.CorrIndex3D(PTX_PATH)
    unified_idx.build(torch.tensor(cb, dtype=torch.float32).contiguous(), 0.95, 4)

    csr_row_ptrs   = torch.from_numpy(np.load(os.path.join(SPARSE_CSR_DIR, "csr_row_ptrs.npy")).astype(np.int32)).cuda()
    csr_col_eids   = torch.from_numpy(np.load(os.path.join(SPARSE_CSR_DIR, "csr_col_eids.npy"))).cuda()
    csr_block_sums = torch.from_numpy(np.load(os.path.join(SPARSE_CSR_DIR, "csr_block_sums_128.npy")).astype(np.int64)).cuda()
    csr_lengths    = torch.from_numpy(np.load(os.path.join(SPARSE_CSR_DIR, "csr_lengths.npy"))).cuda()
    map_packed     = torch.from_numpy(np.load(os.path.join(SPARSE_CSR_DIR, "csr_packed_24.npy"))).cuda()

    centroids_128  = searcher.ranker.codec.centroids.detach().cuda().to(torch.float16).contiguous()
    centroids_96   = torch.from_numpy(np.load(os.path.join(SPARSE_CSR_DIR, "centroids_96d_svd.npy"))).cuda().float()
    R_proj         = torch.from_numpy(np.load(os.path.join(SPARSE_CSR_DIR, "svd_rotation_128_to_96.npy"))).cuda().float()

    native_pred_path = os.path.join(SPARSE_CSR_DIR, "synthetic_predicates.npy" if DATASET == "msmarco" else "doc_predicates.npy")
    doc_predicates = torch.from_numpy(np.load(native_pred_path).astype(np.int64)).cuda()

    unified_idx.bind_index(
        csr_row_ptrs, csr_col_eids, csr_block_sums, csr_lengths, map_packed,
        scorer.doc_offsets[:N+1], scorer.doclens[:N], scorer.codes, scorer.residuals,
        centroids_128, scorer.bucket_weights, scorer.reversed_bit_map, scorer.decomp_table,
        doc_predicates.to(torch.uint32), N, centroids_96.shape[0])

    questions = load_questions(QUESTIONS_PATH, NUM_QUERIES + 5)
    
    # Pre-encode queries using RAGRT's selected nc
    nc_ragrt = selected_params['ragrt']['nc']
    Q_list = []
    for qid, q in questions[:NUM_QUERIES]:
        Qf = searcher.encode(q).squeeze(0).cuda()
        ntok = L.query_ntok(q)
        Q_act = Qf[:ntok, :].contiguous()
        Q_96 = torch.nn.functional.normalize(Q_act @ R_proj, p=2, dim=-1)
        Q_sub = Q_96.view(ntok, 32, 3).contiguous()
        scores = Q_96 @ centroids_96.T
        topc = scores.topk(k=nc_ragrt, dim=-1).indices.contiguous().to(torch.int32)
        Q_list.append((Qf.unsqueeze(0), Qf.float().contiguous(), Qf.to(torch.float16).contiguous(),
                       Q_sub, topc, scores))

    # Configure searcher for PLAID variants
    if engine in ['plaid', 'plaid_tms']:
        cfg_key = 'ptms' if engine == 'plaid_tms' else engine
        searcher.config.ncells = selected_params[cfg_key]['ncells']
        searcher.config.ndocs  = selected_params[cfg_key]['ndocs']
        searcher.config.centroid_score_threshold = 0.45

    # Warmup
    for _ in range(3):
        if engine == 'plaid':
            _ = searcher.ranker.rank(searcher.config, Q_list[0][0])
        elif engine == 'plaid_tms':
            pids_s3 = get_stage3_pids(searcher.ranker, searcher.config, Q_list[0][0])
            _ = scorer.score(Q_list[0][1], pids_s3)
        else:
            p = selected_params['ragrt']
            _ = unified_idx.search_single_query_native(
                Q_list[0][1], Q_list[0][2], Q_list[0][3], Q_list[0][4], Q_list[0][5],
                p['nc'], p['ndocs'], TOP_K, query_mask=0, k_eids=p['eids'])
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
            p = selected_params['ragrt']
            _ = unified_idx.search_single_query_native(
                Qf128, Qfp16, Qsub, topc, scores,
                p['nc'], p['ndocs'], TOP_K, query_mask=0, k_eids=p['eids'])

    torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    print(f"[worker] Done.", flush=True)


if __name__ == '__main__':
    main()
