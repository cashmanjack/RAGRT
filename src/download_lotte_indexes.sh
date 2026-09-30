#!/bin/bash
set -e

BASE_DIR="/home/min/a/cashman3/RTRAG/src/experiments"
mkdir -p "$BASE_DIR"
cd "$BASE_DIR"

DOMAINS=("technology" "lifestyle" "recreation" "writing")

for DOMAIN in "${DOMAINS[@]}"; do
    INDEX_DIR="${BASE_DIR}/lotte_${DOMAIN}/indexes/${DOMAIN}.dev.2bit"
    if [ ! -d "$INDEX_DIR" ]; then
        echo "=========================================================="
        echo "Downloading pre-built index for: $DOMAIN"
        echo "=========================================================="
        TAR_FILE="${DOMAIN}.dev.2bit.tar.gz"
        wget -c "https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/lotte_indexes/${TAR_FILE}" -O "${TAR_FILE}" || \
        curl -C - -O "https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/lotte_indexes/${TAR_FILE}"
        
        mkdir -p "${BASE_DIR}/lotte_${DOMAIN}/indexes"
        tar -xvf "${TAR_FILE}" -C "${BASE_DIR}/lotte_${DOMAIN}/indexes/"
        rm -f "${TAR_FILE}"
    else
        echo "Index for $DOMAIN already exists at $INDEX_DIR."
    fi
done
echo "All LoTTE indexes are ready!"
