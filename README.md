# RAGRT: Late‑Interaction Retrieval with RT‑Accelerated Candidate Generation

## Evaluation pipeline (`src/`)

1. Index: `build_lotte_sparse_csr.sh` / `build_msmarco_sparse_csr.sh` (codebooks, CSR, eval set, `predicates.npy`); check with `check_index.py`.
2. Exact ground truth: `build_ground_truth.py --dataset <lotte|msmarco>` (full-corpus TileMaxSim per query, unfiltered and per filter).
3. Benchmarks: `run_benchmarks.py --dataset <name>` (tune/test split, sweeps, operating points, filtered runs, throughput, drop counters).
4. Figures and `summary.md`: `plot_results.py --dataset <name>`; DRAM: `profile_dram.py --dataset <name>` (Nsight Compute, calibrated).

Paths live in `eval_config.py`. CPU tests: `tests/test_index_lib.py`, `tests/test_eval_lib.py`, `tests/test_harness_smoke.py`. GPU test (after building): `tests/test_stage1_gpu.py`.

No-RT ablation: `run_benchmarks.py --engines ragrt,ragrt_bf` (engine `ragrt_bf` runs Stage 1 as an exact brute-force top-k on CUDA cores; Stages 2-4 unchanged). PCA-filter diagnostic: `diag_pca_filter.py`.
