#!/bin/bash
# Build the RAGRT index for unified LoTTE (2.43M passages): codebooks, CSR, predicates.
set -euo pipefail
export PYTHONUNBUFFERED=1   # progress shows up in nohup logs right away
cd "$(dirname "$0")"

INDEX="/home/min/a/cashman3/RTRAG/src/experiments/unified_lotte/indexes/unified.dev.2bit"
COLLECTION="/local/scratch/a/cashman3/lotte/unified/collection.tsv"
OUT_DIR="/local/scratch/a/cashman3/juno_pq_lotte_full_sparse8"
TOP_M=10
LOTTE_ROOT="/local/scratch/a/cashman3/lotte"            # official lotte.tar.gz extracted here (<domain>/dev/...)
EVAL_DIR="/local/scratch/a/cashman3/lotte/eval_pooled_dev"   # must match eval_config.py

echo "== Step 1: rotation + per-subspace codebooks (this index only)"
python3 train_codebooks.py --index "$INDEX" --collection "$COLLECTION" --outdir "$OUT_DIR" --top_m "$TOP_M"

echo "== Step 2: sparse CSR (24-bit pids, block sums)"
python3 build_full_lotte_sparse_csr.py --index "$INDEX" --collection "$COLLECTION" --outdir "$OUT_DIR" --top_m "$TOP_M"

echo "== Step 3: evaluation set (all domains) and predicates (synthetic + verified domain bits)"
python3 prepare_lotte_eval.py --lotte_root "$LOTTE_ROOT" --unified_collection "$COLLECTION" --outdir "$EVAL_DIR"
python3 build_predicates.py --outdir "$OUT_DIR" --num_passages 2428854 --domains_json "$EVAL_DIR/domains.json"

echo "LoTTE RAGRT index ready in $OUT_DIR"
