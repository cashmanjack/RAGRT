"""
Build Synthetic Predicate Masks for Filtered Retrieval Scaling
Target selectivities: [5%, 14%, 30%, 60%, 100%]
Seed: 42 (Recorded and deterministic)
Saved as synthetic_predicates.npy in /local/scratch/a/cashman3/juno_pq_lotte_full_sparse8/
"""
import os
import numpy as np

SPARSE_CSR_DIR = "/local/scratch/a/cashman3/juno_pq_lotte_full_sparse8"
TOTAL_PASSAGES = 2428854
SEED = 42

np.random.seed(SEED)

TARGET_SELECTIVITIES = [0.05, 0.14, 0.30, 0.60]
predicates = np.zeros(TOTAL_PASSAGES, dtype=np.uint32)

print("=" * 70)
print(f"GENERATING SYNTHETIC PREDICATE MASKS (Seed: {SEED})")
print("=" * 70)

# Bit 0: 5% selectivity
# Bit 1: 14% selectivity
# Bit 2: 30% selectivity
# Bit 3: 60% selectivity
# Bit 4: 100% selectivity (all ones for sanity check)

for bit_idx, target_sel in enumerate(TARGET_SELECTIVITIES):
    mask_bool = np.random.rand(TOTAL_PASSAGES) < target_sel
    actual_sel = mask_bool.mean()
    predicates[mask_bool] |= (1 << bit_idx)
    print(f"  Bit {bit_idx}: Target = {target_sel * 100:4.1f}%, Actual = {actual_sel * 100:5.2f}% ({(predicates & (1 << bit_idx) != 0).sum():,} passages)")

# Bit 4: 100% selectivity (all ones)
predicates |= (1 << 4)
print(f"  Bit 4: Target = 100.0%, Actual = 100.00% ({TOTAL_PASSAGES:,} passages)")

out_path = os.path.join(SPARSE_CSR_DIR, "synthetic_predicates.npy")
np.save(out_path, predicates)

print(f"\nSaved synthetic predicates to: {out_path}")
print(f"Array size: {predicates.nbytes / 1e6:.2f} MB (Fits entirely in L2 cache)")
print("=" * 70)
