import os, numpy as np

import argparse
ap = argparse.ArgumentParser(); ap.add_argument("--outdir", default="/local/scratch/a/cashman3/juno_pq_lotte_full_sparse8")
SPARSE_CSR_DIR = ap.parse_args().outdir
TOTAL_PASSAGES = 2428854

print("=" * 80)
print(f"GENERATING 9.7 MB PREDICATE BITMASK TABLE FOR {TOTAL_PASSAGES:,} PASSAGES")
print("=" * 80)

# 5 LoTTE Domain Bitmasks:
# Bit 0: Science     (Mask: 1 << 0 = 0x01)
# Bit 1: Writing     (Mask: 1 << 1 = 0x02)
# Bit 2: Recreation  (Mask: 1 << 2 = 0x04)
# Bit 3: Lifestyle   (Mask: 1 << 3 = 0x08)
# Bit 4: Technology  (Mask: 1 << 4 = 0x10)

predicates = np.zeros(TOTAL_PASSAGES, dtype=np.uint32)

# LoTTE Domain Boundaries in the unified collection:
# Science:    0 .. 343,641
# Writing:    343,642 .. 613,641
# Recreation: 613,642 .. 1,093,641
# Lifestyle:  1,093,642 .. 1,693,641
# Technology: 1,693,642 .. 2,428,853

predicates[0:343642] = (1 << 0)
predicates[343642:613642] = (1 << 1)
predicates[613642:1093642] = (1 << 2)
predicates[1093642:1693642] = (1 << 3)
predicates[1693642:2428854] = (1 << 4)

out_file = os.path.join(SPARSE_CSR_DIR, "doc_predicates.npy")
np.save(out_file, predicates)

print(f"Saved: {out_file}")
print(f"Size : {predicates.nbytes / 1e6:.2f} MB (Fits 100% in L2 cache!)")
print(f"Science Passages    : {(predicates == 1).sum():,}")
print(f"Technology Passages : {(predicates == 16).sum():,}")
print("=" * 80)
