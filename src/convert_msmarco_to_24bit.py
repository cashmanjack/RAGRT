"""
Streaming converter: unpacks (pid << 9 | pos) in csr_packed.npy 
and emits 24-bit packed passage IDs to csr_packed_24.npy.
"""
import os, sys, time
import numpy as np

SPARSE_DIR = "/local/scratch/a/cashman3/msmarco/msmarco_sparse8"
IN_PATH    = os.path.join(SPARSE_DIR, "csr_packed.npy")
OUT_PATH   = os.path.join(SPARSE_DIR, "csr_packed_24.npy")

CHUNK_SIZE = 50_000_000 # Process 50M entries per chunk (~200 MB RAM)

print("=" * 70)
print("CONVERTING MS MARCO POSTINGS TO 24-BIT (3 BYTES PER PASSAGE ID)")
print("=" * 70)

# Memory-map the 23.8 GB input array
in_mmap = np.load(IN_PATH, mmap_mode='r')
total_entries = len(in_mmap)
total_bytes = total_entries * 3
print(f"Total entries: {total_entries:,}")
print(f"Target size  : {total_bytes / 1e9:.2f} GB")

# Pre-allocate output file via disk-backed memmap
out_mmap = np.memmap(OUT_PATH, dtype=np.uint8, mode='w+', shape=(total_bytes,))

t0 = time.time()
for start in range(0, total_entries, CHUNK_SIZE):
    end = min(start + CHUNK_SIZE, total_entries)
    n = end - start

    # Extract 24-bit pid from (pid << 9) | pos
    pids = (in_mmap[start:end] >> 9).astype(np.uint32)

    # Pack into 3 contiguous bytes (little-endian)
    b0 = (pids & 0xFF).astype(np.uint8)
    b1 = ((pids >> 8) & 0xFF).astype(np.uint8)
    b2 = ((pids >> 16) & 0xFF).astype(np.uint8)

    # Interleave into output array
    out_byte_start = start * 3
    out_byte_end   = end * 3

    chunk_24 = np.empty((n, 3), dtype=np.uint8)
    chunk_24[:, 0] = b0
    chunk_24[:, 1] = b1
    chunk_24[:, 2] = b2

    out_mmap[out_byte_start:out_byte_end] = chunk_24.reshape(-1)

    elapsed = time.time() - t0
    rate = end / elapsed / 1e6
    pct = (end / total_entries) * 100
    print(f"  Processed {end:,} / {total_entries:,} entries ({pct:5.1f}%) | {rate:.1f} M entries/sec")

out_mmap.flush()
del in_mmap, out_mmap

print("\nConversion complete! Verifying file on disk...")
print(f"Created: {OUT_PATH} ({os.path.getsize(OUT_PATH) / 1e9:.2f} GB)")
print("=" * 70)
