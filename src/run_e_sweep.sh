#!/bin/bash
# Benchmark RAGRT (RT and no-RT) on each E-sweep index, importing PLAID / PLAID+TMS from
# the main run (BASE, a results subdir produced with the same split and rerank kernel).
#   ./run_e_sweep.sh lotte v2_E256 1024 4096 16384
set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$(dirname "$0")"
DATASET="$1"; BASE="$2"; shift 2
ROOT="${RAGRT_RESULTS:-/local/scratch/a/cashman3/ragrt_results}/$DATASET"
for E in "$@"; do
  DIR="/local/scratch/a/cashman3/ragrt_${DATASET}_E${E}"
  echo "== E=$E ($DIR)"
  python3 run_benchmarks.py --dataset "$DATASET" --sparse_csr_dir "$DIR" --results_subdir "E${E}" \
      --baseline_from "$ROOT/$BASE" --engines ragrt,ragrt_bf
  python3 plot_results.py --dataset "$DATASET" --results_subdir "E${E}"
done
python3 summarize_e_sweep.py --dataset "$DATASET" --base "$BASE" --Es "256,$(IFS=,; echo "$*")"
