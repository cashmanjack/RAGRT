"""
Sanity-check a built RAGRT index directory (no GPU, no ColBERT needed).

  python3 check_index.py <sparse_csr_dir> [<sparse_csr_dir> ...]

Checks: CSR shape invariants, block sums, codebook/rotation fingerprint,
pid range (max pid vs passage count, share of pids >= 2^22 that the old
int32 packing used to corrupt). Reads csr_packed_24.npy in chunks via mmap.
"""
import os, sys, json
import numpy as np

import ragrt_index_lib as L

CHUNK_POSTINGS = 100_000_000


def check(d):
    print(f"\n== {d}")
    ok = True

    def expect(cond, msg):
        nonlocal ok
        print(("  ok   " if cond else "  FAIL ") + msg)
        ok &= bool(cond)

    row_ptrs = np.load(os.path.join(d, "csr_row_ptrs.npy"), mmap_mode="r")
    lengths = np.load(os.path.join(d, "csr_lengths.npy"), mmap_mode="r")
    eids = np.load(os.path.join(d, "csr_col_eids.npy"), mmap_mode="r")
    bsums = np.load(os.path.join(d, "csr_block_sums_128.npy"))
    packed = np.load(os.path.join(d, "csr_packed_24.npy"), mmap_mode="r")
    cb = np.load(os.path.join(d, "codebooks.npy"))
    meta = json.load(open(os.path.join(d, "codebook_meta.json")))
    R = np.load(os.path.join(d, "svd_rotation_128_to_96.npy"))
    preds = np.load(os.path.join(d, "synthetic_predicates.npy"), mmap_mode="r")
    num_passages = len(preds)

    n_lists = len(lengths)
    n_post = len(packed) // 3
    lsum = int(lengths.sum(dtype=np.int64))
    print(f"  {n_lists:,} lists, {n_post:,} postings, {num_passages:,} passages")

    expect(len(packed) % 3 == 0, "packed length is a multiple of 3")
    expect(lsum == n_post, f"sum(lengths) == postings ({lsum:,})")
    expect(len(eids) == n_lists, "one eid per list")
    expect(row_ptrs.shape[0] == L.NUM_SUBSPACES, f"row_ptrs has {L.NUM_SUBSPACES} subspace rows")
    expect(int(row_ptrs[0, 0]) == 0 and int(row_ptrs[-1, -1]) == n_lists, "row_ptrs span [0, num_lists]")
    expect(bool(np.all(np.diff(np.asarray(row_ptrs).ravel().astype(np.int64)) >= 0)), "row_ptrs non-decreasing")
    expect(int(lengths.min()) >= 1, "no empty lists")
    expect(np.array_equal(bsums, L.block_sums(np.asarray(lengths))), "block sums match lengths")

    expect(cb.shape[0] == L.NUM_SUBSPACES and cb.shape[2] == L.SUBSPACE_DIM and cb.shape[1] <= 256,
           f"codebooks shape {tuple(cb.shape)}")
    expect(meta.get("rotation_sha256") == L.rotation_fingerprint(R), "codebooks match this rotation")
    print(f"  codebook mean rel_err {meta.get('mean_rel_err', float('nan')):.3f}")

    max_pid, hi_count = -1, 0
    for a in range(0, n_post, CHUNK_POSTINGS):
        pids = L.unpack_pids_24(packed[3 * a: 3 * min(a + CHUNK_POSTINGS, n_post)])
        max_pid = max(max_pid, int(pids.max()))
        hi_count += int(np.count_nonzero(pids >= (1 << 22)))
    expect(max_pid == num_passages - 1, f"max pid {max_pid:,} == num_passages - 1")
    print(f"  postings with pid >= 2^22 (old bug range): {hi_count:,} ({hi_count / n_post:.1%})")
    return ok


if __name__ == "__main__":
    dirs = sys.argv[1:]
    if not dirs:
        sys.exit(__doc__)
    results = [check(d) for d in dirs]
    print("\nALL OK" if all(results) else "\nSOME CHECKS FAILED")
    sys.exit(0 if all(results) else 1)
