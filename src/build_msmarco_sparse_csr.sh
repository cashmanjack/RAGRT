#!/bin/bash
set -e

INDEX_DIR="/local/scratch/a/cashman3/msmarco/indexes/msmarco/indexes/msmarco.dev.2bit"
SYMLINK_DIR="/local/scratch/a/cashman3/msmarco/indexes/msmarco.dev.2bit"
COLLECTION="/local/scratch/a/cashman3/msmarco/collection.tsv"
OUT_DIR="/local/scratch/a/cashman3/msmarco/msmarco_sparse8"

echo "=========================================================="
echo "Step 1: Creating clean symlink for MS MARCO index"
echo "=========================================================="
ln -sfn "$INDEX_DIR" "$SYMLINK_DIR"

echo "=========================================================="
echo "Step 2: Compiling MS MARCO Sparse CSR Structure (8.8M docs)"
echo "=========================================================="
python3 build_full_lotte_sparse_csr.py \
    --index "$SYMLINK_DIR" \
    --collection "$COLLECTION" \
    --outdir "$OUT_DIR" \
    --top_m 10

echo "=========================================================="
echo "Step 3: Generating MS MARCO Synthetic Predicate Bitmasks"
echo "=========================================================="
python3 -c "
import os, numpy as np
SPARSE_DIR = '$OUT_DIR'
os.makedirs(SPARSE_DIR, exist_ok=True)
TOTAL_PASSAGES = 8841823
np.random.seed(42)
predicates = np.zeros(TOTAL_PASSAGES, dtype=np.uint32)
for bit, sel in enumerate([0.05, 0.14, 0.30, 0.60]):
    predicates[np.random.rand(TOTAL_PASSAGES) < sel] |= (1 << bit)
predicates |= (1 << 4) # 100%
np.save(os.path.join(SPARSE_DIR, 'synthetic_predicates.npy'), predicates)
print('MS MARCO synthetic predicates created successfully.')
"

echo "=========================================================="
echo "MS MARCO INDEX & CSR READY FOR BENCHMARKING!"
echo "=========================================================="
