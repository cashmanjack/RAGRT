#!/bin/bash
# Build RAGRT indexes with larger codebooks (E codewords per subspace) for the E sweep.
# Each E gets its own directory; the 128->96 rotation and the predicates are shared with
# the main index (same frame, so results are comparable). E=256 is the main index itself.
#   ./build_e_sweep.sh lotte 1024 4096 16384
# Disk: each index needs ~3 bytes per posting plus ~4 bytes per list; check df first.
set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$(dirname "$0")"

DATASET="$1"; shift
case "$DATASET" in
  lotte)
    INDEX="/home/min/a/cashman3/RTRAG/src/experiments/unified_lotte/indexes/unified.dev.2bit"
    COLLECTION="/local/scratch/a/cashman3/lotte/unified/collection.tsv"
    MAIN="/local/scratch/a/cashman3/juno_pq_lotte_full_sparse8" ;;
  msmarco)
    INDEX="/local/scratch/a/cashman3/msmarco/indexes/msmarco.dev.2bit"
    COLLECTION="/local/scratch/a/cashman3/msmarco/collection.tsv"
    MAIN="/local/scratch/a/cashman3/msmarco/msmarco_sparse8" ;;
  *) echo "usage: $0 lotte|msmarco E [E ...]"; exit 1 ;;
esac
TOP_M=10

for E in "$@"; do
  OUT="/local/scratch/a/cashman3/ragrt_${DATASET}_E${E}"
  SAMPLES=$(( E >= 4096 ? 8000000 : 4000000 ))
  echo "== E=$E -> $OUT  (free on scratch: $(df -h --output=avail /local/scratch/a/cashman3 | tail -1))"
  mkdir -p "$OUT"
  python3 train_codebooks.py --index "$INDEX" --collection "$COLLECTION" --outdir "$OUT" \
      --top_m "$TOP_M" --num_entries "$E" --sample_tokens "$SAMPLES" --rotation_from "$MAIN"
  python3 build_full_lotte_sparse_csr.py --index "$INDEX" --collection "$COLLECTION" --outdir "$OUT" --top_m "$TOP_M"
  ln -sfn "$MAIN/predicates.npy" "$OUT/predicates.npy"
  python3 check_index.py "$OUT"
  echo "== E=$E done: $(du -sh "$OUT" | cut -f1)"
done
