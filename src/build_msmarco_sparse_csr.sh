#!/bin/bash
# Build the RAGRT index for MS MARCO: rotation + codebooks, CSR, synthetic predicates.
set -euo pipefail
export PYTHONUNBUFFERED=1   # progress shows up in nohup logs right away
cd "$(dirname "$0")"

INDEX_DIR="/local/scratch/a/cashman3/msmarco/indexes/msmarco/indexes/msmarco.dev.2bit"
SYMLINK_DIR="/local/scratch/a/cashman3/msmarco/indexes/msmarco.dev.2bit"
COLLECTION="/local/scratch/a/cashman3/msmarco/collection.tsv"
OUT_DIR="/local/scratch/a/cashman3/msmarco/msmarco_sparse8"
TOP_M=10

ln -sfn "$INDEX_DIR" "$SYMLINK_DIR"

echo "== Step 1: rotation + per-subspace codebooks (this index only)"
python3 train_codebooks.py --index "$SYMLINK_DIR" --collection "$COLLECTION" \
    --outdir "$OUT_DIR" --top_m "$TOP_M"

echo "== Step 2: sparse CSR (24-bit pids, block sums)"
python3 build_full_lotte_sparse_csr.py --index "$SYMLINK_DIR" --collection "$COLLECTION" \
    --outdir "$OUT_DIR" --top_m "$TOP_M"

echo "== Step 3: predicates (synthetic selectivity bits)"
python3 build_predicates.py --outdir "$OUT_DIR" --num_passages 8841823

echo "MS MARCO RAGRT index ready in $OUT_DIR"
