"""
Peak VRAM Measurement Benchmark - LIVE ONLY.
Measures peak GPU memory allocated per query using torch.cuda.max_memory_allocated().
Generates vram_results.csv and fig7_vram.png.
"""
import os, sys, time, csv
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = "/home/min/a/cashman3/RTRAG/src"
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "../reference/colbert-plaid"))

from generate_figures import setup_engines, get_stage3_pids, K_CANDIDATES, TOP_K, N_COARSE, K_EIDS, PRUNE_TAU

NUM_QUERIES = 20

def main():
    print("=" * 70)
    print("PEAK VRAM BENCHMARK (LIVE)")
    print("=" * 70)
    env = setup_engines()
    searcher = env['searcher']; scorer = env['scorer']
    unified_idx = env['unified_idx']
    Q_batches = env['Q_batches']; Q_full_128_list = env['Q_full_128_list']
    Q_full_fp16_list = env['Q_full_fp16_list']; Q_sub3d_list = env['Q_sub3d_list']
    topc_list = env['topc_list']; scores_list = env['scores_list']

    results = {}

    for engine in ['Stock PLAID', 'PLAID+TMS', 'RAGRT']:
        vram_peaks = []
        print(f"  Profiling {engine}...", flush=True)

        for i in range(NUM_QUERIES):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

            if engine == 'Stock PLAID':
                _ = searcher.ranker.rank(searcher.config, Q_batches[i])
            elif engine == 'PLAID+TMS':
                pids_s3 = get_stage3_pids(searcher.ranker, searcher.config, Q_batches[i])
                _ = scorer.score(Q_full_128_list[i], pids_s3)
            else:  # RAGRT
                _ = unified_idx.search_single_query_native(
                    Q_full_128_list[i], Q_full_fp16_list[i], Q_sub3d_list[i],
                    topc_list[i], scores_list[i],
                    N_COARSE, K_CANDIDATES, TOP_K, 0, 0, K_EIDS, 0, PRUNE_TAU)

            torch.cuda.synchronize()
            peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            vram_peaks.append(peak_mb)

        med_vram = float(np.median(vram_peaks))
        results[engine] = med_vram
        print(f"    Peak VRAM: {med_vram:.2f} MB")

    with open('vram_results.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['engine', 'peak_vram_mb'])
        for k, v in results.items():
            w.writerow([k, f"{v:.2f}"])

    # Plot fig7_vram.png
    engines = list(results.keys())
    peaks = list(results.values())
    colors = ['#c0392b', '#e67e22', '#2980b9']

    fig, ax = plt.subplots(figsize=(8, 5.5))
    bars = ax.bar(engines, peaks, color=colors, edgecolor='black', width=0.55)
    ax.set_ylabel('Peak Working Memory per Query (MB)')
    ax.set_title(f'Peak VRAM Allocation ({NUM_QUERIES} queries, median)')
    ax.set_ylim(0, max(peaks) * 1.3)

    for bar, val in zip(bars, peaks):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(peaks)*0.02,
                f'{val:.1f} MB', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('fig7_vram.png', dpi=150, bbox_inches='tight')
    print("  Saved fig7_vram.png\n")

if __name__ == '__main__':
    main()
