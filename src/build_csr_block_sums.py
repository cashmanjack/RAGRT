"""
Build 52 MB Hierarchical Block Prefix Sums for MS MARCO CSR Index.
Replaces 6.72 GB csr_offsets.npy with a 52 MB block_sums table.
"""
import os, time
import numpy as np

SPARSE_DIR = "/local/scratch/a/cashman3/msmarco/msmarco_sparse8"
LEN_PATH   = os.path.join(SPARSE_DIR, "csr_lengths.npy")
OFF_PATH   = os.path.join(SPARSE_DIR, "csr_offsets.npy")
OUT_PATH   = os.path.join(SPARSE_DIR, "csr_block_sums_128.npy")

BLOCK_SIZE = 128

print("=" * 80)
print("BUILDING HIERARCHICAL BLOCK PREFIX SUMS (BLOCK SIZE = 128)")
print("=" * 80)

# Memory-map the lengths and original offsets
lengths = np.load(LEN_PATH, mmap_mode='r')
offsets = np.load(OFF_PATH, mmap_mode='r')
total_lists = len(lengths)
num_blocks  = (total_lists + BLOCK_SIZE - 1) // BLOCK_SIZE

print(f"Total CSR Lists : {total_lists:,}")
print(f"Total Blocks    : {num_blocks:,}")
print(f"Target Size     : {num_blocks * 8 / 1e6:.2f} MB (vs 6,720 MB offsets)")

t0 = time.time()

# Chunked computation over CPU to avoid memory spikes
block_sums = np.zeros(num_blocks, dtype=np.int64)

# Stream lengths in chunks of 10,000,000 lists
CHUNK_LISTS = 10_240_000 # Must be multiple of 128
curr_sum = np.int64(0)

for start in range(0, total_lists, CHUNK_LISTS):
    end = min(start + CHUNK_LISTS, total_lists)
    chunk_l = lengths[start:end].astype(np.int64)
    
    # Pad chunk to multiple of 128 if last chunk
    pad_len = (128 - (len(chunk_l) % 128)) % 128
    if pad_len > 0:
        chunk_l = np.pad(chunk_l, (0, pad_len))
        
    chunk_blocks = chunk_l.reshape(-1, 128).sum(axis=1)
    
    b_start = start // 128
    b_end   = b_start + len(chunk_blocks)
    
    block_sums[b_start] = curr_sum
    if len(chunk_blocks) > 1:
        block_sums[b_start + 1 : b_end] = curr_sum + np.cumsum(chunk_blocks[:-1])
        
    curr_sum += np.sum(chunk_blocks)
    print(f"  Processed {end:,} / {total_lists:,} lists ({(end/total_lists)*100:.1f}%)")

np.save(OUT_PATH, block_sums)
print(f"\nSaved {OUT_PATH} in {time.time() - t0:.2f} s")

# AIRTIGHT VALIDATION: Verify exact bit-for-bit match on 10,000,000 sample lists
print("\nValidating reconstruction against original 6.72 GB offsets array...")
test_n = min(10_000_000, total_lists)
sample_indices = np.random.choice(total_lists, test_n, replace=False)
sample_indices.sort()

b_indices = sample_indices // 128
b_starts  = b_indices * 128

reconstructed = np.empty(test_n, dtype=np.int64)
for idx, (list_id, b_id, b_st) in enumerate(zip(sample_indices, b_indices, b_starts)):
    reconstructed[idx] = block_sums[b_id] + np.sum(lengths[b_st:list_id], dtype=np.int64)

is_exact = np.array_equal(reconstructed, offsets[sample_indices])
print(f"Exact match on {test_n:,} random lists: {is_exact}")
assert is_exact, "FATAL: Reconstructed offsets did not match original offsets!"
print("=" * 80)
