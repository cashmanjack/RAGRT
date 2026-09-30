#!/bin/bash
set -e

LOCAL_ROOT="/home/min/a/cashman3/RTRAG/src/experiments"
DOMAINS=("technology" "lifestyle" "recreation" "writing")

for DOMAIN in "${DOMAINS[@]}"; do
    INDEX_PATH="${LOCAL_ROOT}/lotte_${DOMAIN}/indexes/${DOMAIN}.dev.2bit"
    OUT_DIR="${LOCAL_ROOT}/juno_pq_${DOMAIN}"
    
    if [ -d "$INDEX_PATH" ]; then
        echo "================================================================="
        echo "Building Map & Codebooks for: $DOMAIN"
        echo "Index:   $INDEX_PATH"
        echo "Outdir:  $OUT_DIR"
        echo "================================================================="
        python3 build_lotte_map.py --index "$INDEX_PATH" --outdir "$OUT_DIR"
    else
        echo "Skipping $DOMAIN: Index not found at $INDEX_PATH"
    fi
done
