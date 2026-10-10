# RAGRT: Late‑Interaction Retrieval with RT‑Accelerated Candidate Generation

## Evaluation pipeline (`src/`)

1. Index: `build_lotte_sparse_csr.sh` / `build_msmarco_sparse_csr.sh` (codebooks, CSR, eval set, `predicates.npy`); check with `check_index.py`.
2. Exact ground truth: `build_ground_truth.py --dataset <lotte|msmarco>` (full-corpus TileMaxSim per query, unfiltered and per filter).
3. Benchmarks: `run_benchmarks.py --dataset <name>` (tune/test split, sweeps, operating points, filtered runs, throughput, stage counters).
4. Figures and `summary.md`: `plot_results.py --dataset <name>`; DRAM: `profile_dram.py --dataset <name>` (Nsight Compute, calibrated).

Paths live in `eval_config.py`. CPU tests: `tests/test_index_lib.py`, `tests/test_eval_lib.py`, `tests/test_harness_smoke.py`, `tests/test_builder_cpu.py`. GPU tests (after building): `tests/test_gpu_kernels.py`, `tests/test_stage1_gpu.py`.

## RAGRT stages and options

- Stage 1 (`--geometry`): `polar` (default) puts codeword c on the plane x·c = 1 (one triangle per codeword); a ray from the origin along the unit query slice d hits it at t = 1/(d·c), so tmax = 1/τ returns exactly the codewords with d·c ≥ τ. τ is calibrated per subspace and per k level on TUNE queries (`--rt_quantile`). `fan` is the original scene (hits about every codeword with d·c > 0; RT acts as a half-space filter).
- Codebook size E is set at training time (`train_codebooks.py --num_entries E`, GPU k-means); eids are 8-bit up to E=256 and 16-bit beyond.
- Stage 3 (`--stage3`): `sparse` (default) radix-sorts only the touched postings; `dense` uses an ntok × N table.
- Stage 4 (`--rerank`): `wmma` (default) tensor-core TileMaxSim; `simt` is the original kernel. PLAID+TMS uses the same kernel.
- No-RT ablation: engine `ragrt_bf` runs Stage 1 on CUDA cores with identical output (polar: the same threshold as a linear scan; fan: exact top-k).

Diagnostics: `diag_stage1.py` (recall of the exact top-k codewords per ray, hits per ray, Stage-1 time per variant), `diag_pca_filter.py`.

E sweep: `build_e_sweep.sh <dataset> E...` builds one index per E (shared rotation and predicates), `run_e_sweep.sh <dataset> <base results subdir> E...` benchmarks RAGRT on each and imports PLAID/PLAID+TMS from the base run (`--baseline_from`), and `summarize_e_sweep.py` writes `e_sweep.md` and `fig_e_sweep.png`.
