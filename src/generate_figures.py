"""
Unified RAGRT Benchmark Suite - Pareto-First Execution Architecture
1. Combined Pareto Sweeps run FIRST across all three engines (all 100 queries).
2. Auto-selects operating points: PLAID highest R@10 target, RAGRT/TMS fastest with R@10 >= target.
3. Figures 1, 2, 5, 6, 7 evaluate the exact auto-selected configurations.
4. Figure 3 records CUDA event timings inside Figure 1's run directly (exact sums by construction).

Usage:
  python3 generate_figures.py --dataset lotte
  python3 generate_figures.py --dataset msmarco
  python3 generate_figures.py --dataset lotte --plot-only
"""
import os, sys, time, csv, argparse, json
from math import ceil
from collections import defaultdict
import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import LogLocator

BASE_DIR = "/home/min/a/cashman3/RTRAG/src"
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "../reference/colbert-plaid"))

plt.rcParams['font.size'] = 10

# Hardware Manufacturer Spec Peak Bandwidth (NVIDIA RTX 6000 Ada Generation: 384-bit * 20 Gbps / 8)
RTX_6000_ADA_SPEC_PEAK_BW_GBS = 960.0

DATASETS = {
    'lotte': {
        'index': "/home/min/a/cashman3/RTRAG/src/experiments/unified_lotte/indexes/unified.dev.2bit",
        'collection': "/local/scratch/a/cashman3/lotte/unified/collection.tsv",
        'sparse_csr_dir': "/local/scratch/a/cashman3/juno_pq_lotte_full_sparse8",
        'questions': "/home/min/a/cashman3/RTRAG/src/experiments/lotte_science_eval/questions.tsv",
        'qrels': "/home/min/a/cashman3/RTRAG/src/experiments/lotte_science_eval/qrels.tsv",
        'filter_masks': [(1, "science (14%)")],
        'num_eval_queries': 100,
        'predicates_file': "doc_predicates.npy",
    },
    'msmarco': {
        'index': "/local/scratch/a/cashman3/msmarco/indexes/msmarco.dev.2bit",
        'collection': "/local/scratch/a/cashman3/msmarco/collection.tsv",
        'sparse_csr_dir': "/local/scratch/a/cashman3/msmarco/msmarco_sparse8",
        'questions': "/local/scratch/a/cashman3/msmarco/queries.dev.small.tsv",
        'qrels': "/local/scratch/a/cashman3/msmarco/qrels.dev.small.tsv",
        'filter_masks': [
            (1 << 0, "5%"),
            (1 << 1, "14%"),
            (1 << 2, "30%"),
            (1 << 3, "60%"),
        ],
        'num_eval_queries': 100,
        'predicates_file': "synthetic_predicates.npy",
    }
}

TOP_K            = 100
NUM_WARMUP       = 3
NUM_TIMED_RUNS   = 11
ACTIVE_DATASET   = "lotte"

INDEX_PATH       = ""
COLLECTION_PATH  = ""
SPARSE_CSR_DIR   = ""
PTX_PATH         = os.path.join(BASE_DIR, "optix_corr_3d.ptx")
QUESTIONS_PATH   = ""
QRELS_PATH       = ""
DRAM_CSV_PATH    = ""
PARETO_COMB_CSV  = ""
STD_BENCH_CSV    = ""
LAT_DIST_NPZ     = ""
CONCURRENCY_CSV  = ""
SELECTIVITY_CSV  = ""
SELECTED_CFG_JSON= ""
OUT_DIR            = ""
NUM_EVAL_QUERIES = 100

from colbert import Searcher
from fast_tilemaxsim_scorer import FastTileMaxSimScorer
from colbert.search.strided_tensor import StridedTensor
from colbert.modeling.colbert import colbert_score_reduce
import rtrag_corr_3d


def median_lat_seq(fn, warmup=5, timed=15):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(timed):
        t0 = time.perf_counter(); fn(); torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    return float(np.median(ts))

def recall_at_k(retrieved, relevant, k=10):
    if not relevant: return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)

def mrr_at_k(retrieved, relevant, k=10):
    for i, pid in enumerate(retrieved[:k]):
        if pid in relevant: return 1.0 / (i + 1)
    return 0.0

def load_questions(path, limit=100):
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
            elif len(p) == 3 and float(p[2]) > 0:
                qrels.setdefault(p[0], set()).add(int(p[1]))
    return qrels

def load_dram_results_strict():
    target_path = DRAM_CSV_PATH
    if not os.path.exists(target_path) and ACTIVE_DATASET == 'lotte':
        legacy = os.path.join(BASE_DIR, "dram_results.csv")
        if os.path.exists(legacy): target_path = legacy

    if not os.path.exists(target_path):
        raise FileNotFoundError(
            f"FATAL: Missing DRAM results at {target_path}!\n"
            f"Run 'python3 measure_dram.py' for {ACTIVE_DATASET} first under nsys."
        )
    dram = {}
    with open(target_path) as f:
        r = csv.reader(f); next(r, None)
        for row in r:
            if len(row) >= 2: dram[row[0].strip()] = float(row[1])
    for required in ['plaid', 'plaid_tms', 'ragrt']:
        if required not in dram:
            raise KeyError(f"FATAL: Missing '{required}' entry in {target_path}")
    return dram

def get_stage3_pids(ranker, config, Q):
    """Native single-pass PLAID Stage 1-3 candidate filtering."""
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


def setup_engines():
    print("=" * 80)
    print(f"INITIALIZING ENGINES [DATASET: {ACTIVE_DATASET.upper()}]")
    print("=" * 80)
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

    native_pred_path = os.path.join(SPARSE_CSR_DIR, "doc_predicates.npy")
    doc_predicates = torch.from_numpy(np.load(native_pred_path).astype(np.int64)).cuda()

    synth_pred_path = os.path.join(SPARSE_CSR_DIR, "synthetic_predicates.npy")
    if os.path.exists(synth_pred_path):
        synthetic_predicates = torch.from_numpy(np.load(synth_pred_path).astype(np.int64)).cuda()
    else:
        synthetic_predicates = doc_predicates

    unified_idx.bind_index(
        csr_row_ptrs, csr_col_eids, csr_block_sums, csr_lengths, map_packed,
        scorer.doc_offsets[:N+1], scorer.doclens[:N], scorer.codes, scorer.residuals,
        centroids_128, scorer.bucket_weights, scorer.reversed_bit_map, scorer.decomp_table,
        doc_predicates.to(torch.uint32), N, centroids_96.shape[0])

    questions = load_questions(QUESTIONS_PATH, NUM_EVAL_QUERIES * 2)
    qrels = load_qrels(QRELS_PATH)
    eval_qs = [(qid, q) for (qid, q) in questions if qid in qrels][:NUM_EVAL_QUERIES]

    # Pre-encode queries using maximum nc=64 (subsets can slice)
    Q_batches, Q_full_128_list, Q_full_fp16_list, Q_sub3d_list, topc_list, scores_list, q_ntoks = [], [], [], [], [], [], []
    for qid, q in eval_qs:
        Qf = searcher.encode(q).squeeze(0).cuda()
        ntok = min(len(q.split()) + 4, 32)
        q_ntoks.append(ntok)
        Q_act = Qf[:ntok, :].contiguous()
        Q_96 = torch.nn.functional.normalize(Q_act @ R_proj, p=2, dim=-1)
        Q_sub = Q_96.view(ntok, 32, 3).contiguous()
        scores = Q_96 @ centroids_96.T
        topc = scores.topk(k=64, dim=-1).indices.contiguous().to(torch.int32)
        Q_batches.append(Qf.unsqueeze(0))
        Q_full_128_list.append(Qf.float().contiguous())
        Q_full_fp16_list.append(Qf.to(torch.float16).contiguous())
        Q_sub3d_list.append(Q_sub)
        topc_list.append(topc)
        scores_list.append(scores)

    print(f"Setup complete ({ACTIVE_DATASET}: N={N:,}, {len(eval_qs)} queries).\n", flush=True)
    return {
        'searcher': searcher, 'scorer': scorer, 'N': N,
        'unified_idx': unified_idx, 'doc_predicates': doc_predicates,
        'synthetic_predicates': synthetic_predicates,
        'eval_qs': eval_qs, 'qrels': qrels,
        'Q_batches': Q_batches, 'Q_full_128_list': Q_full_128_list,
        'Q_full_fp16_list': Q_full_fp16_list, 'Q_sub3d_list': Q_sub3d_list,
        'topc_list': topc_list, 'scores_list': scores_list,
        'q_ntoks': q_ntoks, 'R_proj': R_proj, 'centroids_96': centroids_96,
        'csr_row_ptrs': csr_row_ptrs, 'csr_col_eids': csr_col_eids,
        'csr_block_sums': csr_block_sums, 'csr_lengths': csr_lengths, 'map_packed': map_packed,
        'centroids_128': centroids_128
    }


# ==============================================================================
# PARETO SWEEPS & AUTOMATIC OPERATING POINT SELECTION
# ==============================================================================
def run_combined_pareto_frontier(env):
    print("=" * 80)
    print(f"RUNNING PARETO SWEEPS [{ACTIVE_DATASET.upper()}] (ALL 100 QUERIES, LIVE)")
    print("=" * 80)
    searcher = env['searcher']; scorer = env['scorer']; unified_idx = env['unified_idx']
    eval_qs = env['eval_qs']; qrels = env['qrels']
    Q_batches = env['Q_batches']; Q_full_128_list = env['Q_full_128_list']
    Q_full_fp16_list = env['Q_full_fp16_list']; Q_sub3d_list = env['Q_sub3d_list']
    topc_list = env['topc_list']; scores_list = env['scores_list']
    num_eval = len(eval_qs)

    PLAID_NDOCS_GRID  = [256, 512, 1024, 2048, 4096, 8192, 16384]
    PLAID_NCELLS_GRID = [1, 2, 4]

    all_engine_points = {'plaid': [], 'ptms': [], 'ragrt': []}
    orig_ncells = searcher.config.ncells
    orig_ndocs  = searcher.config.ndocs

    try:
        print("  [1/3] Sweeping Stock PLAID (21 configs x 100 queries)...", flush=True)
        for ncells in PLAID_NCELLS_GRID:
            for ndocs in PLAID_NDOCS_GRID:
                searcher.config.ncells = ncells
                searcher.config.ndocs  = ndocs
                searcher.config.centroid_score_threshold = 0.45
                q_lats, q_recs, q_mrrs = [], [], []
                for i in range(num_eval):
                    torch.cuda.synchronize(); t0 = time.perf_counter()
                    ranked_raw, _ = searcher.ranker.rank(searcher.config, Q_batches[i])
                    torch.cuda.synchronize()
                    q_lats.append((time.perf_counter() - t0) * 1000)
                    q_recs.append(recall_at_k(ranked_raw[:TOP_K], qrels[eval_qs[i][0]], 10))
                    q_mrrs.append(mrr_at_k(ranked_raw[:TOP_K], qrels[eval_qs[i][0]], 10))

                all_engine_points['plaid'].append({
                    'cfg': f"nc={ncells} ndocs={ndocs}",
                    'params': {'ncells': ncells, 'ndocs': ndocs},
                    'lat': float(np.median(q_lats)),
                    'mrr': float(np.mean(q_mrrs)),
                    'rec': float(np.mean(q_recs))
                })

        print("  [2/3] Sweeping PLAID+TMS (21 configs x 100 queries)...", flush=True)
        for ncells in PLAID_NCELLS_GRID:
            for ndocs in PLAID_NDOCS_GRID:
                searcher.config.ncells = ncells
                searcher.config.ndocs  = ndocs
                searcher.config.centroid_score_threshold = 0.45
                q_lats, q_recs, q_mrrs = [], [], []
                for i in range(num_eval):
                    torch.cuda.synchronize(); t0 = time.perf_counter()
                    pids_s3 = get_stage3_pids(searcher.ranker, searcher.config, Q_batches[i])
                    torch.cuda.synchronize()
                    _ = scorer.score(Q_full_128_list[i], pids_s3)
                    torch.cuda.synchronize()
                    q_lats.append((time.perf_counter() - t0) * 1000)
                    ranked = scorer.score(Q_full_128_list[i], pids_s3)
                    ranked = ranked.cpu().tolist() if isinstance(ranked, torch.Tensor) else ranked
                    q_recs.append(recall_at_k(ranked[:TOP_K], qrels[eval_qs[i][0]], 10))
                    q_mrrs.append(mrr_at_k(ranked[:TOP_K], qrels[eval_qs[i][0]], 10))

                all_engine_points['ptms'].append({
                    'cfg': f"nc={ncells} ndocs={ndocs}",
                    'params': {'ncells': ncells, 'ndocs': ndocs},
                    'lat': float(np.median(q_lats)),
                    'mrr': float(np.mean(q_mrrs)),
                    'rec': float(np.mean(q_recs))
                })

    finally:
        searcher.config.ncells = orig_ncells
        searcher.config.ndocs  = orig_ndocs

    print("  [3/3] Sweeping RAGRT Pipelined (126 configs x 100 queries)...", flush=True)
    NDOCS_RAGRT   = [512, 1024, 2048, 4096, 8192, 16384, 32768]
    EIDS_RAGRT    = [1, 2, 4, 8, 16, 32]
    NCOARSE_RAGRT = [16, 32, 64]

    for ndocs in NDOCS_RAGRT:
        for eids in EIDS_RAGRT:
            for nc in NCOARSE_RAGRT:
                bs = min(20, num_eval)
                topc_bs = [t[:, :nc].contiguous() for t in topc_list[:bs]]
                def run_pipe():
                    return unified_idx.search_batch_pipelined(
                        Q_full_128_list[:bs], Q_full_fp16_list[:bs], Q_sub3d_list[:bs],
                        topc_bs, scores_list[:bs], nc, ndocs, TOP_K, 0, 0, eids)
                for _ in range(2): run_pipe()
                torch.cuda.synchronize()
                t0 = time.perf_counter(); run_pipe(); torch.cuda.synchronize()
                lat = (time.perf_counter() - t0) * 1000 / bs

                topc_full = [t[:, :nc].contiguous() for t in topc_list]
                ranked_all = unified_idx.search_batch_pipelined(
                    Q_full_128_list, Q_full_fp16_list, Q_sub3d_list,
                    topc_full, scores_list, nc, ndocs, TOP_K, 0, 0, eids)
                recs, mrrs = [], []
                for i in range(num_eval):
                    ranked = ranked_all[i].cpu().tolist()
                    recs.append(recall_at_k(ranked, qrels[eval_qs[i][0]], 10))
                    mrrs.append(mrr_at_k(ranked, qrels[eval_qs[i][0]], 10))

                all_engine_points['ragrt'].append({
                    'cfg': f"ndocs={ndocs} eids={eids} nc={nc}",
                    'params': {'ndocs': ndocs, 'eids': eids, 'nc': nc},
                    'lat': lat,
                    'mrr': float(np.mean(mrrs)),
                    'rec': float(np.mean(recs))
                })

    # Save to CSV
    with open(PARETO_COMB_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['engine', 'config', 'latency_ms', 'mrr10', 'recall10'])
        for eng in ['plaid', 'ptms', 'ragrt']:
            for pt in all_engine_points[eng]:
                writer.writerow([eng, pt['cfg'], f"{pt['lat']:.3f}", f"{pt['mrr']:.4f}", f"{pt['rec']:.4f}"])

    def get_frontier(points):
        sorted_pts = sorted(points, key=lambda x: x['lat'])
        front = []
        max_m = -1.0
        for p in sorted_pts:
            if p['mrr'] > max_m:
                max_m = p['mrr']
                front.append(p)
        return front

    frontiers = {
        'plaid': get_frontier(all_engine_points['plaid']),
        'ptms':  get_frontier(all_engine_points['ptms']),
        'ragrt': get_frontier(all_engine_points['ragrt'])
    }

    return frontiers, all_engine_points


def select_operating_points(frontiers):
    """
    Automated Selection of Operating Points:
    1. PLAID Reference: Frontier point with highest Recall@10 (tiebreak: highest MRR, lowest latency).
    2. RAGRT & TMS: Fastest config on respective frontier achieving Recall@10 >= PLAID target.
    """
    print("=" * 80)
    print("AUTOMATIC OPERATING POINT SELECTION (ISO-ACCURACY BY CONSTRUCTION)")
    print("=" * 80)

    # 1. PLAID Reference
    plaid_front = frontiers['plaid']
    plaid_selected = sorted(plaid_front, key=lambda x: (-x['rec'], -x['mrr'], x['lat']))[0]
    target_recall = plaid_selected['rec']

    # 2. RAGRT Selection
    ragrt_front = frontiers['ragrt']
    ragrt_eligible = [p for p in ragrt_front if p['rec'] >= target_recall]
    if ragrt_eligible:
        ragrt_selected = sorted(ragrt_eligible, key=lambda x: x['lat'])[0]
    else:
        ragrt_selected = sorted(ragrt_front, key=lambda x: (-x['rec'], -x['mrr'], x['lat']))[0]
        print(f"  WARNING: RAGRT frontier did not reach PLAID target recall {target_recall:.4f}. Gap: {target_recall - ragrt_selected['rec']:.4f}")

    # 3. PLAID+TMS Selection
    ptms_front = frontiers['ptms']
    ptms_eligible = [p for p in ptms_front if p['rec'] >= target_recall]
    if ptms_eligible:
        ptms_selected = sorted(ptms_eligible, key=lambda x: x['lat'])[0]
    else:
        ptms_selected = sorted(ptms_front, key=lambda x: (-x['rec'], -x['mrr'], x['lat']))[0]
        print(f"  WARNING: PLAID+TMS frontier did not reach PLAID target recall {target_recall:.4f}. Gap: {target_recall - ptms_selected['rec']:.4f}")

    selected = {
        'plaid': plaid_selected,
        'ptms':  ptms_selected,
        'ragrt': ragrt_selected
    }

    # Print Selection Table
    print(f"\n{'Engine':<16} | {'Selected Configuration':<28} | {'Latency':<9} | {'Recall@10':<9} | {'MRR@10':<8}")
    print("-" * 75)
    for eng, name in [('plaid', 'Stock PLAID'), ('ptms', 'PLAID+TMS'), ('ragrt', 'RAGRT')]:
        pt = selected[eng]
        print(f"{name:<16} | {pt['cfg']:<28} | {pt['lat']:5.2f} ms | {pt['rec']:.4f}    | {pt['mrr']:.4f}")
    print("=" * 80 + "\n")

    # Export to JSON for dram_profile_worker.py to read
    with open(SELECTED_CFG_JSON, 'w') as f:
        json.dump(selected, f, indent=2)
    print(f"Saved selected configurations to {SELECTED_CFG_JSON}\n")

    return selected


# ==============================================================================
# FIGURE 1 & FIGURE 3: STANDARD BENCHMARK & LIVE STAGE INSTRUMENTATION
# ==============================================================================
def run_standard_benchmark(env, selected_configs):
    print("=" * 80)
    print(f"FIGURE 1: STANDARD RETRIEVAL BENCHMARK [{ACTIVE_DATASET.upper()}]")
    print("=" * 80)
    searcher = env['searcher']; scorer = env['scorer']; unified_idx = env['unified_idx']
    eval_qs = env['eval_qs']; qrels = env['qrels']
    Q_batches = env['Q_batches']; Q_full_128_list = env['Q_full_128_list']
    Q_full_fp16_list = env['Q_full_fp16_list']; Q_sub3d_list = env['Q_sub3d_list']
    topc_list = env['topc_list']; scores_list = env['scores_list']

    cfg_p  = selected_configs['plaid']['params']
    cfg_pt = selected_configs['ptms']['params']
    cfg_r  = selected_configs['ragrt']['params']

    # --- 1. Stock PLAID ---
    print("  Benchmarking Stock PLAID (selected config)...", flush=True)
    searcher.config.ncells = cfg_p['ncells']
    searcher.config.ndocs  = cfg_p['ndocs']
    searcher.config.centroid_score_threshold = 0.45

    plaid_tot, plaid_s13, plaid_s4, plaid_recs, plaid_mrrs = [], [], [], [], []
    for i, (qid, _) in enumerate(eval_qs):
        for _ in range(NUM_WARMUP): _ = searcher.ranker.rank(searcher.config, Q_batches[i])
        torch.cuda.synchronize()
        q_tot, q_s13, q_s4 = [], [], []
        for _ in range(NUM_TIMED_RUNS):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            pids_s3 = get_stage3_pids(searcher.ranker, searcher.config, Q_batches[i])
            torch.cuda.synchronize()
            t_s13 = (time.perf_counter() - t0) * 1000

            torch.cuda.synchronize(); t0 = time.perf_counter()
            ranked_raw, _ = searcher.ranker.rank(searcher.config, Q_batches[i])
            torch.cuda.synchronize()
            t_tot = (time.perf_counter() - t0) * 1000

            q_tot.append(t_tot)
            q_s13.append(t_s13)
            q_s4.append(max(0.0, t_tot - t_s13))

        plaid_tot.append(float(np.median(q_tot)))
        plaid_s13.append(float(np.median(q_s13)))
        plaid_s4.append(float(np.median(q_s4)))
        plaid_recs.append(recall_at_k(ranked_raw[:TOP_K], qrels[qid], 10))
        plaid_mrrs.append(mrr_at_k(ranked_raw[:TOP_K], qrels[qid], 10))

    # --- 2. PLAID+TMS ---
    print("  Benchmarking PLAID+TMS (selected config)...", flush=True)
    searcher.config.ncells = cfg_pt['ncells']
    searcher.config.ndocs  = cfg_pt['ndocs']
    searcher.config.centroid_score_threshold = 0.45

    ptms_tot, ptms_s13, ptms_s4, ptms_recs, ptms_mrrs = [], [], [], [], []
    for i, (qid, _) in enumerate(eval_qs):
        for _ in range(NUM_WARMUP):
            pids_s3 = get_stage3_pids(searcher.ranker, searcher.config, Q_batches[i])
            _ = scorer.score(Q_full_128_list[i], pids_s3)
        torch.cuda.synchronize()
        q_tot, q_s13, q_s4 = [], [], []
        for _ in range(NUM_TIMED_RUNS):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            pids_s3 = get_stage3_pids(searcher.ranker, searcher.config, Q_batches[i])
            torch.cuda.synchronize()
            t_s13 = (time.perf_counter() - t0) * 1000

            torch.cuda.synchronize(); t0 = time.perf_counter()
            ranked_pids = scorer.score(Q_full_128_list[i], pids_s3)
            torch.cuda.synchronize()
            t_s4 = (time.perf_counter() - t0) * 1000

            q_s13.append(t_s13); q_s4.append(t_s4); q_tot.append(t_s13 + t_s4)

        ptms_s13.append(float(np.median(q_s13)))
        ptms_s4.append(float(np.median(q_s4)))
        ptms_tot.append(float(np.median(q_tot)))
        ranked = ranked_pids.cpu().tolist() if isinstance(ranked_pids, torch.Tensor) else ranked_pids
        ptms_recs.append(recall_at_k(ranked[:TOP_K], qrels[qid], 10))
        ptms_mrrs.append(mrr_at_k(ranked[:TOP_K], qrels[qid], 10))

    # --- 3. RAGRT (Instrumented with Hardware CUDA Events inside Figure 1) ---
    print("  Benchmarking RAGRT (selected config + live CUDA events)...", flush=True)
    nc_r = cfg_r['nc']; ndocs_r = cfg_r['ndocs']; eids_r = cfg_r['eids']
    topc_r = [t[:, :nc_r].contiguous() for t in topc_list]

    ragrt_tot, ragrt_recs, ragrt_mrrs = [], [], []
    r_s1_l, r_s2_l, r_s3_l, r_s4_l = [], [], [], []

    for i, (qid, _) in enumerate(eval_qs):
        for _ in range(NUM_WARMUP):
            _ = unified_idx.search_single_query_profiled(
                Q_full_128_list[i], Q_full_fp16_list[i], Q_sub3d_list[i], topc_r[i], scores_list[i],
                nc_r, ndocs_r, TOP_K, 0, 0, eids_r)
        torch.cuda.synchronize()
        q_lats = []
        for _ in range(NUM_TIMED_RUNS):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            ranked_tensor, stages = unified_idx.search_single_query_profiled(
                Q_full_128_list[i], Q_full_fp16_list[i], Q_sub3d_list[i], topc_r[i], scores_list[i],
                nc_r, ndocs_r, TOP_K, 0, 0, eids_r)
            torch.cuda.synchronize()
            q_lats.append((time.perf_counter() - t0) * 1000)
            r_s1_l.append(stages[0]); r_s2_l.append(stages[1]); r_s3_l.append(stages[2]); r_s4_l.append(stages[3])

        ragrt_tot.append(float(np.median(q_lats)))
        ranked = ranked_tensor.cpu().tolist()
        ragrt_recs.append(recall_at_k(ranked, qrels[qid], 10))
        ragrt_mrrs.append(mrr_at_k(ranked, qrels[qid], 10))

    dram = load_dram_results_strict()
    results = {
        'plaid': (float(np.median(plaid_tot)), float(np.mean(plaid_recs)), float(np.mean(plaid_mrrs)), dram['plaid']),
        'ptms':  (float(np.median(ptms_tot)),  float(np.mean(ptms_recs)),  float(np.mean(ptms_mrrs)),  dram['plaid_tms']),
        'ragrt': (float(np.median(ragrt_tot)), float(np.mean(ragrt_recs)), float(np.mean(ragrt_mrrs)), dram['ragrt']),
    }

    # Save results to CSV for plot-only mode
    with open(STD_BENCH_CSV, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['engine', 'latency_ms', 'recall10', 'mrr10', 'dram_mb'])
        for eng in ['plaid', 'ptms', 'ragrt']:
            w.writerow([eng, f"{results[eng][0]:.3f}", f"{results[eng][1]:.4f}", f"{results[eng][2]:.4f}", f"{results[eng][3]:.1f}"])

    per_query_data = {
        'plaid_lats': plaid_tot,
        'ptms_lats': ptms_tot,
        'ragrt_lats': ragrt_tot,
        'q_ntoks': env['q_ntoks']
    }
    np.savez(LAT_DIST_NPZ, **per_query_data)

    stage_breakdowns = {
        'plaid': (float(np.median(plaid_s13)), float(np.median(plaid_s4)), results['plaid'][0]),
        'ptms':  (float(np.median(ptms_s13)),  float(np.median(ptms_s4)),  results['ptms'][0]),
        'ragrt': (float(np.median(r_s1_l)), float(np.median(r_s2_l)), float(np.median(r_s3_l)), float(np.median(r_s4_l)), results['ragrt'][0])
    }

    plot_standard_benchmark(results, selected_configs)
    return results, per_query_data, stage_breakdowns


def plot_standard_benchmark(results, selected_configs):
    sp_ptms = results['plaid'][0] / results['ptms'][0]
    sp_ragrt = results['plaid'][0] / results['ragrt'][0]
    dr_ptms = results['plaid'][3] / results['ptms'][3]
    dr_ragrt = results['plaid'][3] / results['ragrt'][3]

    cfg_p  = selected_configs['plaid']['cfg']
    cfg_pt = selected_configs['ptms']['cfg']
    cfg_r  = selected_configs['ragrt']['cfg']

    print("\n" + "=" * 108)
    print(f"FIGURE 1: STANDARD BENCHMARK RESULTS [{ACTIVE_DATASET.upper()}]")
    print("=" * 108)
    print(f"{'Engine':<12} | {'Selected Parameters':<26} | {'Latency':<9} | {'Speedup':<8} | {'DRAM/Query':<11} | {'Data Reduc':<10} | {'Recall@10':<9} | {'MRR@10':<8}")
    print("-" * 108)
    print(f"{'Stock PLAID':<12} | {cfg_p:<26} | {results['plaid'][0]:6.2f} ms | {'1.0x':<8} | {results['plaid'][3]:6.1f} MB  | {'1.0x':<10} | {results['plaid'][1]:.4f}    | {results['plaid'][2]:.4f}")
    print(f"{'PLAID+TMS':<12} | {cfg_pt:<26} | {results['ptms'][0]:6.2f} ms | {sp_ptms:5.2f}x  | {results['ptms'][3]:6.1f} MB  | {dr_ptms:5.2f}x    | {results['ptms'][1]:.4f}    | {results['ptms'][2]:.4f}")
    print(f"{'RAGRT':<12} | {cfg_r:<26} | {results['ragrt'][0]:6.2f} ms | {sp_ragrt:5.2f}x  | {results['ragrt'][3]:6.1f} MB  | {dr_ragrt:5.2f}x    | {results['ragrt'][1]:.4f}    | {results['ragrt'][2]:.4f}")
    print("=" * 108 + "\n")

    engines = [f'Stock PLAID\n({cfg_p})', f'PLAID+TMS\n({cfg_pt})', f'RAGRT\n({cfg_r})']
    colors = ['#c0392b', '#e67e22', '#2980b9']
    latencies = [results['plaid'][0], results['ptms'][0], results['ragrt'][0]]
    speedups  = [1.0, sp_ptms, sp_ragrt]
    dram_vals = [results['plaid'][3], results['ptms'][3], results['ragrt'][3]]
    dram_red  = [1.0, dr_ptms, dr_ragrt]
    recalls   = [results['plaid'][1], results['ptms'][1], results['ragrt'][1]]
    mrrs      = [results['plaid'][2], results['ptms'][2], results['ragrt'][2]]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 5.5))
    bars1 = ax1.bar(engines, latencies, color=colors, edgecolor='black', width=0.52)
    ax1.set_ylabel('Latency (ms)')
    ax1.set_title('End-to-End Latency', pad=15)
    ax1.set_ylim(0, max(latencies) * 1.35)
    ax1.tick_params(axis='x', labelsize=8.5)
    for b, l, s in zip(bars1, latencies, speedups):
        ax1.text(b.get_x() + b.get_width()/2, b.get_height() + max(latencies)*0.02,
                 f'{l:.2f} ms ({s:.1f}x)', ha='center', va='bottom', fontsize=9.5, fontweight='bold')

    bars2 = ax2.bar(engines, dram_vals, color=colors, edgecolor='black', width=0.52)
    ax2.set_ylabel('DRAM Traffic (MB / query)')
    ax2.set_title('Memory Movement per Query', pad=15)
    ax2.set_ylim(0, max(dram_vals) * 1.35)
    ax2.tick_params(axis='x', labelsize=8.5)
    for b, d, r in zip(bars2, dram_vals, dram_red):
        ax2.text(b.get_x() + b.get_width()/2, b.get_height() + max(dram_vals)*0.02,
                 f'{d:.1f} MB ({r:.1f}x)', ha='center', va='bottom', fontsize=9.5, fontweight='bold')

    x = np.arange(len(engines)); w = 0.32
    b_r = ax3.bar(x - w/2, recalls, w, label='Recall@10', color='#27ae60', edgecolor='black')
    b_m = ax3.bar(x + w/2, mrrs, w, label='MRR@10', color='#8e44ad', edgecolor='black')
    ax3.set_ylabel('Accuracy Score')
    ax3.set_title('Retrieval Quality', pad=15)
    ax3.set_xticks(x); ax3.set_xticklabels(engines, fontsize=8.5)
    ax3.set_ylim(0.0, max(max(recalls), max(mrrs)) * 1.35)
    ax3.legend(loc='lower left', fontsize=9.5)
    for bar in list(b_r) + list(b_m):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                 f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    fig_name = f'fig1_standard_benchmark_{ACTIVE_DATASET}.png'
    plt.savefig(os.path.join(OUT_DIR, fig_name), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {fig_name}")


def run_latency_breakdown(stage_breakdowns, std_results):
    """Figure 3: Plots Stage Times Recorded Directly Inside Figure 1 (Exact Sums by Construction)."""
    print("=" * 80)
    print(f"FIGURE 3: LATENCY BREAKDOWN [{ACTIVE_DATASET.upper()}]")
    print("=" * 80)

    p_s13, p_s4, plaid_tot = stage_breakdowns['plaid']
    pt_s13, pt_s4, ptms_tot = stage_breakdowns['ptms']
    r_s1, r_s2, r_s3, r_s4, ragrt_tot = stage_breakdowns['ragrt']

    r_gpu_tot = r_s1 + r_s2 + r_s3 + r_s4
    r_dispatch = max(0.0, ragrt_tot - r_gpu_tot)

    print("\n" + "=" * 90)
    print(f"FIGURE 3: LATENCY BREAKDOWN [{ACTIVE_DATASET.upper()}]")
    print("=" * 90)
    print(f"{'Engine':<16} | {'Stage 1-3 (Candidate Gen)':<28} | {'Stage 4 (MaxSim / TMS)':<24} | {'Total Latency':<12}")
    print("-" * 90)
    print(f"{'Stock PLAID':<16} | {p_s13:5.2f} ms ({p_s13/plaid_tot*100:4.1f}%)             | {p_s4:5.2f} ms ({p_s4/plaid_tot*100:4.1f}%)        | {plaid_tot:5.2f} ms")
    print(f"{'PLAID+TMS':<16} | {pt_s13:5.2f} ms ({pt_s13/ptms_tot*100:4.1f}%)             | {pt_s4:5.2f} ms ({pt_s4/ptms_tot*100:4.1f}%)         | {ptms_tot:5.2f} ms")
    print(f"{'RAGRT':<16} | S1:{r_s1:.2f}ms S2:{r_s2:.2f}ms S3:{r_s3:.2f}ms  | S4(TMS):{r_s4:.2f}ms ({r_s4/r_gpu_tot*100:4.1f}%)    | {ragrt_tot:5.2f} ms*")
    print("-" * 90)
    print(f"*Note: RAGRT active GPU time is {r_gpu_tot:.2f} ms ({r_dispatch:.2f} ms dispatch overhead).")
    print("=" * 90 + "\n")

    fig, ax = plt.subplots(figsize=(13, 7))
    x_pos = [0, 1, 2]
    width = 0.52

    c_p_s13 = '#e67e22'; c_p_s4 = '#c0392b'
    ax.bar(x_pos[0], p_s13, color=c_p_s13, edgecolor='black', width=width)
    ax.bar(x_pos[0], p_s4, bottom=p_s13, color=c_p_s4, edgecolor='black', width=width)
    ax.text(x_pos[0], p_s13/2, f'{p_s13:.2f} ms ({p_s13/plaid_tot*100:.0f}%)',
            ha='center', va='center', color='white', fontweight='bold', fontsize=9.5)
    ax.text(x_pos[0], p_s13 + p_s4/2, f'{p_s4:.2f} ms ({p_s4/plaid_tot*100:.0f}%)',
            ha='center', va='center', color='white', fontweight='bold', fontsize=9.5)

    c_pt_s4 = '#d35400'
    ax.bar(x_pos[1], pt_s13, color=c_p_s13, edgecolor='black', width=width)
    ax.bar(x_pos[1], pt_s4, bottom=pt_s13, color=c_pt_s4, edgecolor='black', width=width)
    ax.text(x_pos[1], pt_s13/2, f'{pt_s13:.2f} ms ({pt_s13/ptms_tot*100:.0f}%)',
            ha='center', va='center', color='white', fontweight='bold', fontsize=9.5)
    ax.text(x_pos[1], pt_s13 + pt_s4/2, f'{pt_s4:.2f} ms ({pt_s4/ptms_tot*100:.0f}%)',
            ha='center', va='center', color='white', fontweight='bold', fontsize=9.5)

    c_r_s1 = '#5dade2'; c_r_s2 = '#2980b9'; c_r_s3 = '#1f618d'; c_r_s4 = '#154360'
    bot = 0
    for val, col, name in [(r_s1, c_r_s1, 'S1 RT'), (r_s2, c_r_s2, 'S2 Gather'), (r_s3, c_r_s3, 'S3 Score'), (r_s4, c_r_s4, 'S4 TMS')]:
        ax.bar(x_pos[2], val, bottom=bot, color=col, edgecolor='black', width=width)
        if val >= 0.08:
            ax.text(x_pos[2], bot + val/2, f'{name}: {val:.2f} ms ({val/r_gpu_tot*100:.0f}%)',
                    ha='center', va='center', color='white', fontsize=8.5, fontweight='bold')
        bot += val

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'Stock PLAID\n({plaid_tot:.2f} ms)', f'PLAID+TMS\n({ptms_tot:.2f} ms)', f'RAGRT\n({ragrt_tot:.2f} ms)'], fontsize=9)
    ax.set_ylabel('Latency (ms)')
    ax.set_title(f'Per-Stage Latency Breakdown [{ACTIVE_DATASET.upper()}]', pad=20)
    ax.set_ylim(0, max(plaid_tot, ptms_tot) * 1.25)

    legend_elements = [
        mpatches.Patch(color=c_p_s13, label='Stages 1-3 Candidate Generation (PLAID IVF + StridedTensor)'),
        mpatches.Patch(color=c_p_s4,  label='Stage 4 Stock Scoring (Residual Decompression + GEMM)'),
        mpatches.Patch(color=c_pt_s4, label='Stage 4 Fused TileMaxSim Scoring'),
        mpatches.Patch(color=c_r_s1,  label='RAGRT Stage 1 (OptiX Silicon Ray Tracing)'),
        mpatches.Patch(color=c_r_s2,  label='RAGRT Stage 2 (CSR Gather Tasks)'),
        mpatches.Patch(color=c_r_s3,  label='RAGRT Stage 3 (Cooperative Scoring)'),
        mpatches.Patch(color=c_r_s4,  label='RAGRT Stage 4 (Fused TileMaxSim)'),
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.22), ncol=2, fontsize=8.5, frameon=True)

    plt.tight_layout()
    fig_name = f'fig3_latency_breakdown_{ACTIVE_DATASET}.png'
    plt.savefig(os.path.join(OUT_DIR, fig_name), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {fig_name}")


def run_filtered_benchmark(env, selected_configs):
    print("=" * 80)
    print(f"FIGURE 2: FILTERED RETRIEVAL BENCHMARK [{ACTIVE_DATASET.upper()}]")
    print("=" * 80)
    searcher = env['searcher']; scorer = env['scorer']; unified_idx = env['unified_idx']
    eval_qs = env['eval_qs']; qrels = env['qrels']
    Q_batches = env['Q_batches']; Q_full_128_list = env['Q_full_128_list']
    Q_full_fp16_list = env['Q_full_fp16_list']; Q_sub3d_list = env['Q_sub3d_list']
    topc_list = env['topc_list']; scores_list = env['scores_list']
    doc_predicates = env['doc_predicates']

    cfg_p  = selected_configs['plaid']['params']
    cfg_pt = selected_configs['ptms']['params']
    cfg_r  = selected_configs['ragrt']['params']

    searcher.config.ncells = cfg_p['ncells']
    searcher.config.ndocs  = cfg_p['ndocs']
    searcher.config.centroid_score_threshold = 0.45

    filter_masks = DATASETS[ACTIVE_DATASET]['filter_masks']
    nc_r = cfg_r['nc']; ndocs_r = cfg_r['ndocs']; eids_r = cfg_r['eids']

    for mask_val, mask_lbl in filter_masks:
        print(f"\n--- Evaluating Filter Target: {mask_lbl} ---")
        pf_lats, pf_recs, pf_mrrs = [], [], []
        for i, (qid, _) in enumerate(eval_qs):
            q_lats = []
            for _ in range(NUM_TIMED_RUNS):
                torch.cuda.synchronize(); t0 = time.perf_counter()
                ranked_raw, _ = searcher.ranker.rank(searcher.config, Q_batches[i])
                torch.cuda.synchronize()
                q_lats.append((time.perf_counter() - t0) * 1000)
            pf_lats.append(float(np.median(q_lats)))
            filt = [p for p in ranked_raw if (doc_predicates[p].item() & mask_val) != 0][:TOP_K]
            pf_recs.append(recall_at_k(filt, qrels[qid], 10))
            pf_mrrs.append(mrr_at_k(filt, qrels[qid], 10))

        # PLAID+TMS
        searcher.config.ncells = cfg_pt['ncells']
        searcher.config.ndocs  = cfg_pt['ndocs']
        ptf_lats, ptf_recs, ptf_mrrs = [], [], []
        for i, (qid, _) in enumerate(eval_qs):
            q_lats = []
            for _ in range(NUM_TIMED_RUNS):
                torch.cuda.synchronize(); t0 = time.perf_counter()
                pids_s3 = get_stage3_pids(searcher.ranker, searcher.config, Q_batches[i])
                mask = (doc_predicates[pids_s3.long()] & mask_val) != 0
                pids_f = pids_s3[mask][:cfg_pt['ndocs']]
                torch.cuda.synchronize()
                _ = scorer.score(Q_full_128_list[i], pids_f) if len(pids_f) > 0 else []
                torch.cuda.synchronize()
                q_lats.append((time.perf_counter() - t0) * 1000)
            ptf_lats.append(float(np.median(q_lats)))
            ranked = scorer.score(Q_full_128_list[i], pids_f) if len(pids_f) > 0 else []
            ranked = ranked.cpu().tolist()[:TOP_K] if isinstance(ranked, torch.Tensor) else ranked[:TOP_K]
            ptf_recs.append(recall_at_k(ranked, qrels[qid], 10))
            ptf_mrrs.append(mrr_at_k(ranked, qrels[qid], 10))

        # RAGRT
        topc_r = [t[:, :nc_r].contiguous() for t in topc_list]
        rf_lats, rf_recs, rf_mrrs = [], [], []
        for i, (qid, _) in enumerate(eval_qs):
            q_lats = []
            for _ in range(NUM_TIMED_RUNS):
                torch.cuda.synchronize(); t0 = time.perf_counter()
                ranked_tensor = unified_idx.search_single_query_native(
                    Q_full_128_list[i], Q_full_fp16_list[i], Q_sub3d_list[i], topc_r[i], scores_list[i],
                    nc_r, ndocs_r, TOP_K, 0, mask_val, eids_r)
                torch.cuda.synchronize()
                q_lats.append((time.perf_counter() - t0) * 1000)
            rf_lats.append(float(np.median(q_lats)))
            ranked = ranked_tensor.cpu().tolist()
            rf_recs.append(recall_at_k(ranked, qrels[qid], 10))
            rf_mrrs.append(mrr_at_k(ranked, qrels[qid], 10))

        med_p = float(np.median(pf_lats)); rec_p = float(np.mean(pf_recs)); mrr_p = float(np.mean(pf_mrrs))
        med_pt = float(np.median(ptf_lats)); rec_pt = float(np.mean(ptf_recs)); mrr_pt = float(np.mean(ptf_mrrs))
        med_r = float(np.median(rf_lats)); rec_r = float(np.mean(rf_recs)); mrr_r = float(np.mean(rf_mrrs))

        sp_pt = med_p / med_pt
        sp_r = med_p / med_r

        print(f"{'Stock PLAID':<16} | {med_p:6.2f} ms | {'1.0x':<8} | {rec_p:.4f}    | {mrr_p:.4f}")
        print(f"{'PLAID+TMS':<16} | {med_pt:6.2f} ms | {sp_pt:5.2f}x  | {rec_pt:.4f}    | {mrr_pt:.4f}")
        print(f"{'RAGRT':<16} | {med_r:6.2f} ms | {sp_r:5.2f}x  | {rec_r:.4f}    | {mrr_r:.4f}")

    engines = ['Stock PLAID\n(post-filter)', 'PLAID+TMS\n(post-filter)', 'RAGRT\n(in-engine)']
    colors = ['#c0392b', '#e67e22', '#2980b9']
    latencies = [med_p, med_pt, med_r]
    speedups  = [1.0, sp_pt, sp_r]
    recalls   = [rec_p, rec_pt, rec_r]
    mrrs      = [mrr_p, mrr_pt, mrr_r]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    bars1 = ax1.bar(engines, latencies, color=colors, edgecolor='black', width=0.52)
    ax1.set_ylabel('Latency (ms)')
    ax1.set_title(f'Filtered Latency ({filter_masks[0][1]} Selectivity)', pad=15)
    ax1.set_ylim(0, max(latencies) * 1.35)
    ax1.tick_params(axis='x', labelsize=8.5)
    for b, l, s in zip(bars1, latencies, speedups):
        ax1.text(b.get_x() + b.get_width()/2, b.get_height() + max(latencies)*0.02,
                 f'{l:.2f} ms ({s:.1f}x)', ha='center', va='bottom', fontsize=9, fontweight='bold')

    x = np.arange(len(engines)); w = 0.32
    b_r = ax2.bar(x - w/2, recalls, w, label='Recall@10', color='#27ae60', edgecolor='black')
    b_m = ax2.bar(x + w/2, mrrs, w, label='MRR@10', color='#8e44ad', edgecolor='black')
    ax2.set_ylabel('Accuracy Score')
    ax2.set_title('Filtered Quality', pad=15)
    ax2.set_xticks(x); ax2.set_xticklabels(engines, fontsize=8.5)
    ax2.set_ylim(0.0, max(max(recalls), max(mrrs)) * 1.35)
    ax2.legend(loc='lower left', fontsize=9.5)
    for bar in list(b_r) + list(b_m):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                 f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    fig_name = f'fig2_filtered_benchmark_{ACTIVE_DATASET}.png'
    plt.savefig(os.path.join(OUT_DIR, fig_name), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {fig_name}")
    return {'plaid_f': (med_p, rec_p, mrr_p), 'ptms_f': (med_pt, rec_pt, mrr_pt), 'ragrt_f': (med_r, rec_r, mrr_r)}


def plot_combined_pareto_frontier(frontiers, all_engine_points, selected_configs):
    fig, ax = plt.subplots(figsize=(11, 6.5))

    plot_configs = [
        ('Stock PLAID', all_engine_points['plaid'], frontiers['plaid'], '#c0392b', 's', selected_configs['plaid']['lat'], selected_configs['plaid']['mrr']),
        ('PLAID+TMS',   all_engine_points['ptms'],  frontiers['ptms'],  '#e67e22', 'D', selected_configs['ptms']['lat'],  selected_configs['ptms']['mrr']),
        ('RAGRT (Pipelined)', all_engine_points['ragrt'], frontiers['ragrt'], '#2980b9', 'o', selected_configs['ragrt']['lat'], selected_configs['ragrt']['mrr']),
    ]

    for name, pts, front, col, mark, star_lat, star_mrr in plot_configs:
        xs = [p['lat'] for p in pts]
        ys = [p['mrr'] for p in pts]
        ax.scatter(xs, ys, color=col, marker=mark, alpha=0.20, s=28, edgecolors='none')

        fx = [p['lat'] for p in front]
        fy = [p['mrr'] for p in front]
        ax.plot(fx, fy, color=col, linewidth=2.4, label=f'{name} Frontier')
        ax.scatter(fx, fy, color=col, marker=mark, s=65, edgecolors='black', linewidth=0.8, zorder=4)

        # Star placed programmatically on the exact measured curve coordinate
        ax.scatter([star_lat], [star_mrr], color=col, marker='*', s=240, edgecolors='black', linewidth=1.2, zorder=6)

    ax.set_xscale('log')
    ax.set_xlabel('End-to-End Latency (ms, log scale)')
    ax.set_ylabel('Retrieval Accuracy (MRR@10)')
    ax.set_title(f'Combined Pareto Frontier: Quality vs. Latency Trade-Off [{ACTIVE_DATASET.upper()}]', pad=15)
    ax.grid(True, which='both', linestyle=':', alpha=0.5)
    ax.legend(loc='lower right', fontsize=10)

    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=10))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=15))
    ax.tick_params(which='major', length=6)
    ax.tick_params(which='minor', length=3)

    plt.tight_layout()
    fig_name = f'fig4_pareto_combined_{ACTIVE_DATASET}.png'
    plt.savefig(os.path.join(OUT_DIR, fig_name), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {fig_name}")


def run_concurrency_scaling(env, selected_configs):
    print("=" * 80)
    print(f"FIGURE 6: CONCURRENCY & SERVING THROUGHPUT SCALING [{ACTIVE_DATASET.upper()}]")
    print("=" * 80)
    searcher = env['searcher']; scorer = env['scorer']; unified_idx = env['unified_idx']
    eval_qs = env['eval_qs']
    Q_full_128_list = env['Q_full_128_list']; Q_full_fp16_list = env['Q_full_fp16_list']
    Q_sub3d_list = env['Q_sub3d_list']; topc_list = env['topc_list']; scores_list = env['scores_list']

    cfg_p  = selected_configs['plaid']['params']
    cfg_pt = selected_configs['ptms']['params']
    cfg_r  = selected_configs['ragrt']['params']

    BATCH_SIZES = [1, 2, 4, 8, 16, 32]
    results = {'plaid': {'qps': [], 'p99': []}, 'ptms': {'qps': [], 'p99': []}, 'ragrt': {'qps': [], 'p99': []}}

    print(f"{'Batch (B)':<10} | {'Engine':<18} | {'Throughput (QPS)':<18} | {'p99 Latency':<14} | {'Speedup vs PLAID'}")
    print("-" * 80)

    nc_r = cfg_r['nc']; ndocs_r = cfg_r['ndocs']; eids_r = cfg_r['eids']

    for B in BATCH_SIZES:
        N_BATCHES = 40 if B <= 8 else 25

        B_Q_full_128 = [Q_full_128_list[k % len(eval_qs)] for k in range(B)]
        B_Q_full_fp16 = [Q_full_fp16_list[k % len(eval_qs)] for k in range(B)]
        B_Q_sub3d     = [Q_sub3d_list[k % len(eval_qs)] for k in range(B)]
        B_topc        = [topc_list[k % len(eval_qs)][:, :nc_r].contiguous() for k in range(B)]
        B_scores      = [scores_list[k % len(eval_qs)] for k in range(B)]

        # 1. RAGRT Pipelined Batch
        def run_ragrt():
            return unified_idx.search_batch_pipelined(
                B_Q_full_128, B_Q_full_fp16, B_Q_sub3d, B_topc, B_scores,
                nc_r, ndocs_r, TOP_K, 0, 0, eids_r)

        for _ in range(3): run_ragrt()
        torch.cuda.synchronize()

        batch_times = []
        for _ in range(N_BATCHES):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            run_ragrt()
            torch.cuda.synchronize()
            batch_times.append((time.perf_counter() - t0) * 1000)

        tot_time_sec = sum(batch_times) / 1000.0
        qps_ragrt = (B * N_BATCHES) / tot_time_sec
        p99_ragrt = float(np.percentile(batch_times, 99)) / B
        results['ragrt']['qps'].append(qps_ragrt)
        results['ragrt']['p99'].append(p99_ragrt)

        # 2. Stock PLAID
        searcher.config.ncells = cfg_p['ncells']
        searcher.config.ndocs  = cfg_p['ndocs']
        def run_plaid_batch():
            for k in range(B):
                _ = searcher.ranker.rank(searcher.config, env['Q_batches'][k % len(eval_qs)])

        for _ in range(2): run_plaid_batch()
        torch.cuda.synchronize()
        p_times = []
        for _ in range(N_BATCHES):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            run_plaid_batch()
            torch.cuda.synchronize()
            p_times.append((time.perf_counter() - t0) * 1000)

        tot_p_sec = sum(p_times) / 1000.0
        qps_plaid = (B * N_BATCHES) / tot_p_sec
        p99_plaid = float(np.percentile(p_times, 99)) / B
        results['plaid']['qps'].append(qps_plaid)
        results['plaid']['p99'].append(p99_plaid)

        # 3. PLAID+TMS
        searcher.config.ncells = cfg_pt['ncells']
        searcher.config.ndocs  = cfg_pt['ndocs']
        def run_ptms_batch():
            for k in range(B):
                pids = get_stage3_pids(searcher.ranker, searcher.config, env['Q_batches'][k % len(eval_qs)])
                _ = scorer.score(env['Q_full_128_list'][k % len(eval_qs)], pids)

        for _ in range(2): run_ptms_batch()
        torch.cuda.synchronize()
        pt_times = []
        for _ in range(N_BATCHES):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            run_ptms_batch()
            torch.cuda.synchronize()
            pt_times.append((time.perf_counter() - t0) * 1000)

        tot_pt_sec = sum(pt_times) / 1000.0
        qps_ptms = (B * N_BATCHES) / tot_pt_sec
        p99_ptms = float(np.percentile(pt_times, 99)) / B
        results['ptms']['qps'].append(qps_ptms)
        results['ptms']['p99'].append(p99_ptms)

        sp_ragrt = qps_ragrt / qps_plaid
        sp_ptms  = qps_ptms / qps_plaid

        print(f"B={B:<7} | {'RAGRT (Pipelined)':<18} | {qps_ragrt:7.1f} QPS       | {p99_ragrt:5.2f} ms       | {sp_ragrt:5.2f}x")
        print(f"{'':<10} | {'PLAID+TMS':<18} | {qps_ptms:7.1f} QPS       | {p99_ptms:5.2f} ms       | {sp_ptms:5.2f}x")
        print(f"{'':<10} | {'Stock PLAID':<18} | {qps_plaid:7.1f} QPS       | {p99_plaid:5.2f} ms       | 1.00x")
        print("-" * 80)

    print("=" * 80 + "\n")

    with open(CONCURRENCY_CSV, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['batch_size', 'engine', 'qps', 'p99_ms'])
        for b_idx, B in enumerate(BATCH_SIZES):
            for eng in ['plaid', 'ptms', 'ragrt']:
                w.writerow([B, eng, f"{results[eng]['qps'][b_idx]:.2f}", f"{results[eng]['p99'][b_idx]:.2f}"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    col_map  = {'plaid': '#c0392b', 'ptms': '#e67e22', 'ragrt': '#2980b9'}
    lbl_map  = {'plaid': 'Stock PLAID', 'ptms': 'PLAID+TMS', 'ragrt': 'RAGRT (Pipelined)'}
    mark_map = {'plaid': 's', 'ptms': 'D', 'ragrt': 'o'}

    for eng in ['plaid', 'ptms', 'ragrt']:
        ax1.plot(BATCH_SIZES, results[eng]['qps'], color=col_map[eng], marker=mark_map[eng],
                 linewidth=2.2, markersize=7, label=lbl_map[eng])
    ax1.set_xscale('log', base=2); ax1.set_yscale('log')
    ax1.set_xlabel('Concurrent Batch Size (B)'); ax1.set_ylabel('Serving Throughput (Queries / Sec, log scale)')
    ax1.set_title('Serving Throughput vs. Batch Size', pad=15)
    ax1.set_xticks(BATCH_SIZES); ax1.set_xticklabels([str(b) for b in BATCH_SIZES])
    ax1.grid(True, which='both', linestyle=':', alpha=0.5); ax1.legend(loc='upper left', fontsize=9.5)

    for eng in ['plaid', 'ptms', 'ragrt']:
        ax2.plot(BATCH_SIZES, results[eng]['p99'], color=col_map[eng], marker=mark_map[eng],
                 linewidth=2.2, markersize=7, label=lbl_map[eng])
    ax2.set_xscale('log', base=2); ax2.set_xlabel('Concurrent Batch Size (B)'); ax2.set_ylabel('p99 Per-Query Latency (ms)')
    ax2.set_title('p99 Tail Latency vs. Batch Size', pad=15)
    ax2.set_xticks(BATCH_SIZES); ax2.set_xticklabels([str(b) for b in BATCH_SIZES])
    ax2.grid(True, which='both', linestyle=':', alpha=0.5); ax2.legend(loc='upper left', fontsize=9.5)

    plt.tight_layout()
    fig_name = f'fig6_concurrency_scaling_{ACTIVE_DATASET}.png'
    plt.savefig(os.path.join(OUT_DIR, fig_name), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {fig_name}")


def run_selectivity_sweep(env, selected_configs):
    print("=" * 80)
    print(f"FIGURE 7: SELECTIVITY SCALING SWEEP [{ACTIVE_DATASET.upper()}] (5%, 14%, 30%, 60%, 100%)")
    print("=" * 80)
    searcher = env['searcher']; scorer = env['scorer']; unified_idx = env['unified_idx']
    eval_qs = env['eval_qs']; qrels = env['qrels']
    Q_batches = env['Q_batches']; Q_full_128_list = env['Q_full_128_list']
    Q_full_fp16_list = env['Q_full_fp16_list']; Q_sub3d_list = env['Q_sub3d_list']
    topc_list = env['topc_list']; scores_list = env['scores_list']

    cfg_p  = selected_configs['plaid']['params']
    cfg_pt = selected_configs['ptms']['params']
    cfg_r  = selected_configs['ragrt']['params']

    synth_pred = env['synthetic_predicates']
    N = env['N']
    unified_idx.bind_index(
        env['csr_row_ptrs'], env['csr_col_eids'], env['csr_block_sums'], env['csr_lengths'], env['map_packed'],
        scorer.doc_offsets[:N+1], scorer.doclens[:N], scorer.codes, scorer.residuals,
        env['centroids_128'], scorer.bucket_weights, scorer.reversed_bit_map, scorer.decomp_table,
        synth_pred.to(torch.uint32), N, env['centroids_96'].shape[0])

    SELECTIVITY_TIERS = [(5, (1 << 0)), (14, (1 << 1)), (30, (1 << 2)), (60, (1 << 3)), (100, (1 << 4))]
    results = {'sel_pct': [], 'plaid': {'lat': [], 'rec': [], 'mrr': []}, 'ptms': {'lat': [], 'rec': [], 'mrr': [], 'cands': []}, 'ragrt': {'lat': [], 'rec': [], 'mrr': []}}

    print(f"{'Selectivity':<12} | {'Engine':<16} | {'Latency':<9} | {'Recall@10':<9} | {'MRR@10':<8} | {'Surviving Cands (TMS)'}")
    print("-" * 80)

    nc_r = cfg_r['nc']; ndocs_r = cfg_r['ndocs']; eids_r = cfg_r['eids']
    topc_r = [t[:, :nc_r].contiguous() for t in topc_list]

    try:
        for sel_pct, mask_bit in SELECTIVITY_TIERS:
            results['sel_pct'].append(sel_pct)

            # 1. Stock PLAID
            searcher.config.ncells = cfg_p['ncells']
            searcher.config.ndocs  = cfg_p['ndocs']
            p_lats, p_recs, p_mrrs = [], [], []
            for i in range(len(eval_qs)):
                q_lats = []
                for _ in range(NUM_TIMED_RUNS):
                    torch.cuda.synchronize(); t0 = time.perf_counter()
                    ranked_raw, _ = searcher.ranker.rank(searcher.config, Q_batches[i])
                    torch.cuda.synchronize()
                    q_lats.append((time.perf_counter() - t0) * 1000)
                p_lats.append(float(np.median(q_lats)))
                filt = [p for p in ranked_raw if (synth_pred[p].item() & mask_bit) != 0][:TOP_K]
                p_recs.append(recall_at_k(filt, qrels[eval_qs[i][0]], 10))
                p_mrrs.append(mrr_at_k(filt, qrels[eval_qs[i][0]], 10))

            med_lat_p = float(np.median(p_lats)); rec_p = float(np.mean(p_recs)); mrr_p = float(np.mean(p_mrrs))
            results['plaid']['lat'].append(med_lat_p); results['plaid']['rec'].append(rec_p); results['plaid']['mrr'].append(mrr_p)

            # 2. PLAID+TMS
            searcher.config.ncells = cfg_pt['ncells']
            searcher.config.ndocs  = cfg_pt['ndocs']
            pt_lats, pt_recs, pt_mrrs, pt_cands = [], [], [], []
            for i in range(len(eval_qs)):
                q_lats = []
                for _ in range(NUM_TIMED_RUNS):
                    torch.cuda.synchronize(); t0 = time.perf_counter()
                    pids_s3 = get_stage3_pids(searcher.ranker, searcher.config, Q_batches[i])
                    mask = (synth_pred[pids_s3.long()] & mask_bit) != 0
                    pids_f = pids_s3[mask][:cfg_pt['ndocs']]
                    torch.cuda.synchronize()
                    _ = scorer.score(Q_full_128_list[i], pids_f) if len(pids_f) > 0 else []
                    torch.cuda.synchronize()
                    q_lats.append((time.perf_counter() - t0) * 1000)
                pt_lats.append(float(np.median(q_lats)))
                pt_cands.append(len(pids_f))
                ranked = scorer.score(Q_full_128_list[i], pids_f) if len(pids_f) > 0 else []
                ranked = ranked.cpu().tolist() if isinstance(ranked, torch.Tensor) else ranked
                pt_recs.append(recall_at_k(ranked[:TOP_K], qrels[eval_qs[i][0]], 10))
                pt_mrrs.append(mrr_at_k(ranked[:TOP_K], qrels[eval_qs[i][0]], 10))

            med_lat_pt = float(np.median(pt_lats)); rec_pt = float(np.mean(pt_recs)); mrr_pt = float(np.mean(pt_mrrs)); mean_cands = float(np.mean(pt_cands))
            results['ptms']['lat'].append(med_lat_pt); results['ptms']['rec'].append(rec_pt); results['ptms']['mrr'].append(mrr_pt); results['ptms']['cands'].append(mean_cands)

            # 3. RAGRT
            r_lats, r_recs, r_mrrs = [], [], []
            for i in range(len(eval_qs)):
                q_lats = []
                for _ in range(NUM_TIMED_RUNS):
                    torch.cuda.synchronize(); t0 = time.perf_counter()
                    ranked_tensor = unified_idx.search_single_query_native(
                        Q_full_128_list[i], Q_full_fp16_list[i], Q_sub3d_list[i], topc_r[i], scores_list[i],
                        nc_r, ndocs_r, TOP_K, 0, mask_bit, eids_r)
                    torch.cuda.synchronize()
                    q_lats.append((time.perf_counter() - t0) * 1000)
                r_lats.append(float(np.median(q_lats)))
                ranked = ranked_tensor.cpu().tolist()
                r_recs.append(recall_at_k(ranked, qrels[eval_qs[i][0]], 10))
                r_mrrs.append(mrr_at_k(ranked, qrels[eval_qs[i][0]], 10))

            med_lat_r = float(np.median(r_lats)); rec_r = float(np.mean(r_recs)); mrr_r = float(np.mean(r_mrrs))
            results['ragrt']['lat'].append(med_lat_r); results['ragrt']['rec'].append(rec_r); results['ragrt']['mrr'].append(mrr_r)

            print(f"{sel_pct:>3}% subset  | {'RAGRT':<16} | {med_lat_r:5.2f} ms | {rec_r:.4f}    | {mrr_r:.4f}   | {ndocs_r} (Full depth)")
            print(f"{'':<12} | {'PLAID+TMS':<16} | {med_lat_pt:5.2f} ms | {rec_pt:.4f}    | {mrr_pt:.4f}   | {mean_cands:6.1f} (Starved!)")
            print(f"{'':<12} | {'Stock PLAID':<16} | {med_lat_p:5.2f} ms | {rec_p:.4f}    | {mrr_p:.4f}   | -")
            print("-" * 80)

    finally:
        unified_idx.bind_index(
            env['csr_row_ptrs'], env['csr_col_eids'], env['csr_block_sums'], env['csr_lengths'], env['map_packed'],
            scorer.doc_offsets[:N+1], scorer.doclens[:N], scorer.codes, scorer.residuals,
            env['centroids_128'], scorer.bucket_weights, scorer.reversed_bit_map, scorer.decomp_table,
            env['doc_predicates'].to(torch.uint32), N, env['centroids_96'].shape[0])

    with open(SELECTIVITY_CSV, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['selectivity_pct', 'engine', 'latency_ms', 'recall10', 'mrr10', 'surviving_cands'])
        for idx, s in enumerate(results['sel_pct']):
            w.writerow([s, 'plaid', f"{results['plaid']['lat'][idx]:.3f}", f"{results['plaid']['rec'][idx]:.4f}", f"{results['plaid']['mrr'][idx]:.4f}", "N/A"])
            w.writerow([s, 'ptms', f"{results['ptms']['lat'][idx]:.3f}", f"{results['ptms']['rec'][idx]:.4f}", f"{results['ptms']['mrr'][idx]:.4f}", f"{results['ptms']['cands'][idx]:.1f}"])
            w.writerow([s, 'ragrt', f"{results['ragrt']['lat'][idx]:.3f}", f"{results['ragrt']['rec'][idx]:.4f}", f"{results['ragrt']['mrr'][idx]:.4f}", f"{ndocs_r}"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    sels = results['sel_pct']
    col_map  = {'plaid': '#c0392b', 'ptms': '#e67e22', 'ragrt': '#2980b9'}
    lbl_map  = {'plaid': 'Stock PLAID', 'ptms': 'PLAID+TMS', 'ragrt': 'RAGRT (In-Engine)'}
    mark_map = {'plaid': 's', 'ptms': 'D', 'ragrt': 'o'}

    for eng in ['plaid', 'ptms', 'ragrt']:
        ax1.plot(sels, results[eng]['lat'], color=col_map[eng], marker=mark_map[eng], linewidth=2.2, markersize=7, label=lbl_map[eng])
    ax1.set_xlabel('Target Selectivity (%)'); ax1.set_ylabel('End-to-End Latency (ms)')
    ax1.set_title(f'Filtered Latency vs. Selectivity [{ACTIVE_DATASET.upper()}]', pad=15)
    ax1.grid(True, which='both', linestyle=':', alpha=0.5); ax1.set_xticks(sels); ax1.legend(loc='center right', fontsize=9.5)
    ax1.axvline(14, color='gray', linestyle='--', alpha=0.6, linewidth=1.2)

    for eng in ['plaid', 'ptms', 'ragrt']:
        ax2.plot(sels, results[eng]['rec'], color=col_map[eng], marker=mark_map[eng], linewidth=2.2, markersize=7, label=lbl_map[eng])
    ax2.set_xlabel('Target Selectivity (%)'); ax2.set_ylabel('Recall@10')
    ax2.set_title(f'Retrieval Quality vs. Selectivity [{ACTIVE_DATASET.upper()}]', pad=15)
    ax2.grid(True, which='both', linestyle=':', alpha=0.5); ax2.set_xticks(sels); ax2.legend(loc='lower right', fontsize=9.5)
    ax2.axvline(14, color='gray', linestyle='--', alpha=0.6, linewidth=1.2)

    plt.tight_layout()
    fig_name = f'fig7_selectivity_sweep_{ACTIVE_DATASET}.png'
    plt.savefig(os.path.join(OUT_DIR, fig_name), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {fig_name}")


def run_plot_only():
    print("=" * 80)
    print(f"REPLOTTING FIGURES FROM SAVED LIVE RUN DATA [{ACTIVE_DATASET.upper()}] (--plot-only)")
    print("=" * 80)
    if not os.path.exists(SELECTED_CFG_JSON):
        raise FileNotFoundError(f"Missing {SELECTED_CFG_JSON}. Run without --plot-only first.")

    with open(SELECTED_CFG_JSON) as f:
        selected_configs = json.load(f)

    # 1. Figure 1
    std_res = {}
    with open(STD_BENCH_CSV) as f:
        r = csv.reader(f); next(r)
        for row in r:
            std_res[row[0]] = (float(row[1]), float(row[2]), float(row[3]), float(row[4]))
    plot_standard_benchmark(std_res, selected_configs)

    # 2. Figure 5
    per_query_data = dict(np.load(LAT_DIST_NPZ))
    run_latency_distribution(per_query_data)

    # 3. Figure 4
    comb_pts = {'plaid': [], 'ptms': [], 'ragrt': []}
    with open(PARETO_COMB_CSV) as f:
        r = csv.reader(f); next(r)
        for row in r:
            comb_pts[row[0]].append({'cfg': row[1], 'lat': float(row[2]), 'mrr': float(row[3]), 'rec': float(row[4])})

    def get_frontier(points):
        sorted_pts = sorted(points, key=lambda x: x['lat'])
        front, max_m = [], -1.0
        for p in sorted_pts:
            if p['mrr'] > max_m: max_m = p['mrr']; front.append(p)
        return front

    frontiers = {k: get_frontier(comb_pts[k]) for k in ['plaid', 'ptms', 'ragrt']}
    plot_combined_pareto_frontier(frontiers, comb_pts, selected_configs)

    # 4. Figure 6
    conc_res = {'plaid': {'qps': [], 'p99': []}, 'ptms': {'qps': [], 'p99': []}, 'ragrt': {'qps': [], 'p99': []}}
    with open(CONCURRENCY_CSV) as f:
        r = csv.reader(f); next(r)
        for row in r:
            eng = row[1]
            conc_res[eng]['qps'].append(float(row[2]))
            conc_res[eng]['p99'].append(float(row[3]))

    BATCH_SIZES = [1, 2, 4, 8, 16, 32]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    col_map = {'plaid': '#c0392b', 'ptms': '#e67e22', 'ragrt': '#2980b9'}
    lbl_map = {'plaid': 'Stock PLAID', 'ptms': 'PLAID+TMS', 'ragrt': 'RAGRT (Pipelined)'}
    mark_map = {'plaid': 's', 'ptms': 'D', 'ragrt': 'o'}
    for eng in ['plaid', 'ptms', 'ragrt']:
        ax1.plot(BATCH_SIZES, conc_res[eng]['qps'], color=col_map[eng], marker=mark_map[eng], linewidth=2.2, markersize=7, label=lbl_map[eng])
    ax1.set_xscale('log', base=2); ax1.set_yscale('log')
    ax1.set_xlabel('Concurrent Batch Size (B)'); ax1.set_ylabel('Serving Throughput (Queries / Sec, log scale)')
    ax1.set_title('Serving Throughput vs. Batch Size', pad=15)
    ax1.set_xticks(BATCH_SIZES); ax1.set_xticklabels([str(b) for b in BATCH_SIZES])
    ax1.grid(True, which='both', linestyle=':', alpha=0.5); ax1.legend(loc='upper left', fontsize=9.5)

    for eng in ['plaid', 'ptms', 'ragrt']:
        ax2.plot(BATCH_SIZES, conc_res[eng]['p99'], color=col_map[eng], marker=mark_map[eng], linewidth=2.2, markersize=7, label=lbl_map[eng])
    ax2.set_xscale('log', base=2); ax2.set_xlabel('Concurrent Batch Size (B)'); ax2.set_ylabel('p99 Per-Query Latency (ms)')
    ax2.set_title('p99 Tail Latency vs. Batch Size', pad=15)
    ax2.set_xticks(BATCH_SIZES); ax2.set_xticklabels([str(b) for b in BATCH_SIZES])
    ax2.grid(True, which='both', linestyle=':', alpha=0.5); ax2.legend(loc='upper left', fontsize=9.5)

    plt.tight_layout()
    fig_name = f'fig6_concurrency_scaling_{ACTIVE_DATASET}.png'
    plt.savefig(os.path.join(OUT_DIR, fig_name), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {fig_name}")


def main():
    global INDEX_PATH, COLLECTION_PATH, SPARSE_CSR_DIR, QUESTIONS_PATH, QRELS_PATH
    global DRAM_CSV_PATH, PARETO_COMB_CSV, STD_BENCH_CSV, LAT_DIST_NPZ, CONCURRENCY_CSV, SELECTIVITY_CSV, SELECTED_CFG_JSON
    global ACTIVE_DATASET, NUM_EVAL_QUERIES, OUT_DIR

    parser = argparse.ArgumentParser(description="RAGRT Multi-Dataset Benchmark Suite")
    parser.add_argument("--dataset", choices=['lotte', 'msmarco'], default='lotte', help="Target dataset (lotte or msmarco)")
    parser.add_argument("--plot-only", action="store_true", help="Replot figures from saved live run data")
    args = parser.parse_args()

    ACTIVE_DATASET    = args.dataset
    TAG               = f"_{ACTIVE_DATASET}"
    cfg               = DATASETS[ACTIVE_DATASET]

    OUT_DIR           = os.path.join(BASE_DIR, ACTIVE_DATASET)
    os.makedirs(OUT_DIR, exist_ok=True)

    INDEX_PATH        = cfg['index']
    COLLECTION_PATH   = cfg['collection']
    SPARSE_CSR_DIR    = cfg['sparse_csr_dir']
    QUESTIONS_PATH    = cfg['questions']
    QRELS_PATH        = cfg['qrels']
    NUM_EVAL_QUERIES  = cfg['num_eval_queries']

    DRAM_CSV_PATH     = os.path.join(OUT_DIR, f"dram_results{TAG}.csv")
    PARETO_COMB_CSV   = os.path.join(OUT_DIR, f"pareto_combined_live{TAG}.csv")
    STD_BENCH_CSV     = os.path.join(OUT_DIR, f"standard_results_live{TAG}.csv")
    LAT_DIST_NPZ      = os.path.join(OUT_DIR, f"latency_dist_live{TAG}.npz")
    CONCURRENCY_CSV   = os.path.join(OUT_DIR, f"concurrency_results_live{TAG}.csv")
    SELECTIVITY_CSV   = os.path.join(OUT_DIR, f"selectivity_results_live{TAG}.csv")
    SELECTED_CFG_JSON = os.path.join(OUT_DIR, f"selected_configs_{ACTIVE_DATASET}.json")

    if args.plot_only:
        run_plot_only()
        return

    print("\n" + "=" * 80)
    print(f"STARTING PARETO-FIRST EVALUATION SUITE [DATASET: {ACTIVE_DATASET.upper()}]")
    print("=" * 80 + "\n")
    env = setup_engines()

    # 1. PARETO SWEEPS RUN FIRST
    frontiers, all_engine_points = run_combined_pareto_frontier(env)

    # 2. PROGRAMMATIC OPERATING POINT SELECTION (ISO-ACCURACY BY CONSTRUCTION)
    selected_configs = select_operating_points(frontiers)

    # 3. FIGURE 1: STANDARD BENCHMARK (Evaluates Selected Points + Instruments Stages Live)
    std_results, per_query_data, stage_breakdowns = run_standard_benchmark(env, selected_configs)

    # 4. FIGURE 3: LATENCY BREAKDOWN (Plots Recorded Figure 1 Stages Directly: Zero Subtraction)
    run_latency_breakdown(stage_breakdowns, std_results)

    # 5. FIGURE 2: FILTERED BENCHMARK (Evaluates Selected Points)
    filt_results = run_filtered_benchmark(env, selected_configs)

    # 6. FIGURE 5: LATENCY DISTRIBUTION & CDF (Uses Figure 1 Per-Query Data)
    run_latency_distribution(per_query_data)

    # 7. FIGURE 4: COMBINED PARETO PLOT (Stars Plot Programmatic Selection)
    plot_combined_pareto_frontier(frontiers, all_engine_points, selected_configs)

    # 8. FIGURE 6: CONCURRENCY SCALING (Evaluates Selected Points)
    run_concurrency_scaling(env, selected_configs)

    # 9. FIGURE 7: SELECTIVITY SCALING SWEEP (Evaluates Selected Points)
    run_selectivity_sweep(env, selected_configs)

    print("\n" + "=" * 80)
    print(f"ALL 7 BENCHMARKS COMPLETED FOR {ACTIVE_DATASET.upper()} (PARETO-FIRST DISCIPLINE)")
    print("=" * 80)


def run_latency_distribution(per_query_data):
    print("=" * 80)
    print(f"FIGURE 5: LATENCY PERCENTILES & CDF DISTRIBUTION [{ACTIVE_DATASET.upper()}]")
    print("=" * 80)
    plaid_lats = np.array(per_query_data['plaid_lats'])
    ptms_lats = np.array(per_query_data['ptms_lats'])
    ragrt_lats = np.array(per_query_data['ragrt_lats'])
    q_ntoks = np.array(per_query_data['q_ntoks'])

    def stats(lats):
        return (float(np.median(lats)), float(np.percentile(lats, 95)),
                float(np.percentile(lats, 99)), float(np.max(lats)))

    def tok_corr(lats):
        if len(lats) != len(q_ntoks) or np.std(q_ntoks) == 0:
            return 0.0
        return float(np.corrcoef(q_ntoks, lats)[0, 1])

    rows = [
        ('Stock PLAID', stats(plaid_lats), tok_corr(plaid_lats)),
        ('PLAID+TMS', stats(ptms_lats), tok_corr(ptms_lats)),
        ('RAGRT', stats(ragrt_lats), tok_corr(ragrt_lats)),
    ]
    print(f"{'Engine':<12} | {'p50':<9} | {'p95':<9} | {'p99':<9} | {'Max':<9} | Token Length Corr (r)")
    print("-" * 75)
    for name, (p50, p95, p99, mx), corr in rows:
        print(f"{name:<12} | {p50:5.2f} ms | {p95:5.2f} ms | {p99:5.2f} ms | {mx:5.2f} ms | r = {corr:+.3f}")

    fig, ax = plt.subplots(figsize=(10, 6))
    for lats, col, lbl, mark in [(plaid_lats, '#c0392b', 'Stock PLAID', 's'),
                                  (ptms_lats, '#e67e22', 'PLAID+TMS', 'D'),
                                  (ragrt_lats, '#2980b9', 'RAGRT', 'o')]:
        xs = np.sort(lats)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        ax.plot(xs, ys, color=col, marker=mark, markevery=max(1, len(xs)//20),
                linewidth=2.2, markersize=5, label=lbl)
    ax.set_xlabel('Per-Query Latency (ms)')
    ax.set_ylabel('CDF')
    ax.set_title(f'Latency CDF Distribution [{ACTIVE_DATASET.upper()}]', pad=15)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='lower right', fontsize=10)
    plt.tight_layout()
    fig_name = f'fig5_latency_cdf_{ACTIVE_DATASET}.png'
    plt.savefig(os.path.join(OUT_DIR, fig_name), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {fig_name}")


if __name__ == '__main__':
    main()
