"""
Shared, numpy-only building blocks for the RAGRT offline index.

Everything that decides the on-disk format lives here so the CSR builder, the
codebook trainer and the tests all use the same code. Nothing in this file
imports torch, CUDA or ColBERT, so it can be unit tested on any machine
(see tests/test_index_lib.py).

On-disk format produced by build_full_lotte_sparse_csr.py:
  csr_row_ptrs.npy        int64  [S, num_centroids + 1]  global index into col arrays
  csr_col_eids.npy        uint8 (E <= 256) or uint16 (E <= 65536) [num_lists]  fine codeword id
  csr_lengths.npy         uint16 [num_lists]             postings per list
  csr_block_sums_128.npy  int64  [ceil(num_lists/128)]   offset of list 128*b
  csr_packed_24.npy       uint8  [3 * num_postings]      little-endian 24-bit pids
"""
import hashlib
import numpy as np

NUM_SUBSPACES = 32
SUBSPACE_DIM = 3
PROJ_DIM = NUM_SUBSPACES * SUBSPACE_DIM  # 96

PID_BITS = 24
MAX_PID = (1 << PID_BITS) - 1            # 16,777,215
MAX_LIST_LENGTH = np.iinfo(np.uint16).max
MAX_EIDS_UINT8 = 256                     # eids stored as uint8 up to here
MAX_EIDS = 1 << 16                       # ... and as uint16 beyond (kernels template on the width)
BLOCK_SIZE = 128                         # must match ragrt_fused_kernel.cu (>> 7)


# ---------------------------------------------------------------------------
# Rotation (128D -> 96D) and its fingerprint
# ---------------------------------------------------------------------------
def compute_rotation(centroids_128):
    """Top-96 right singular vectors of the centered centroids, as [128, 96]."""
    c = np.asarray(centroids_128, dtype=np.float64)
    c = c - c.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(c, full_matrices=False)
    return np.ascontiguousarray(vh[:PROJ_DIM].T).astype(np.float32)


def rotation_fingerprint(R):
    """Stable hash of the rotation, used to prove codebooks match the index frame."""
    R = np.ascontiguousarray(np.asarray(R, dtype=np.float32))
    assert R.shape == (128, PROJ_DIM), f"rotation must be [128, {PROJ_DIM}], got {R.shape}"
    return hashlib.sha256(R.tobytes()).hexdigest()


def l2_normalize(x, eps=1e-12):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)


# ---------------------------------------------------------------------------
# Subspace selection and residual slices (one rule, shared by trainer + builder)
# ---------------------------------------------------------------------------
def top_m_active_mask(emb_96, top_m):
    """
    Bool [n, S]: the top_m subspaces per token by squared norm of the token's
    96D embedding slice. This is the rule the CSR builder indexes with; the
    trainer must use the same rule so codewords fit the slices actually indexed.
    """
    emb_96 = np.asarray(emb_96, dtype=np.float32)
    assert emb_96.shape[1] == PROJ_DIM
    assert 1 <= top_m <= NUM_SUBSPACES
    energy = np.square(emb_96.reshape(-1, NUM_SUBSPACES, SUBSPACE_DIM)).sum(axis=-1)
    idx = np.argpartition(-energy, top_m - 1, axis=-1)[:, :top_m]
    mask = np.zeros(energy.shape, dtype=bool)
    np.put_along_axis(mask, idx, True, axis=-1)
    return mask


def residual_96(emb_96, centroids_96, codes):
    """Token residual in the rotated, normalized 96D frame."""
    return np.asarray(emb_96, np.float32) - np.asarray(centroids_96, np.float32)[codes]


def subspace_slice(x_96, s):
    return x_96[:, SUBSPACE_DIM * s: SUBSPACE_DIM * s + SUBSPACE_DIM]


# ---------------------------------------------------------------------------
# ColBERT stores 512 padding rows after the last real token
# ---------------------------------------------------------------------------
COLBERT_PAD_ROWS = 512   # ResidualEmbeddings.load_chunks: num_embeddings += 512


def num_real_tokens(stored_rows, doclens_sum):
    """
    ColBERT's ResidualEmbeddings.load_chunks allocates num_embeddings + 512 rows
    ("pad for access with strides"), so codes/residuals have 512 trailing rows
    that belong to no passage (uninitialized memory). Accept exactly that padding
    and return the real token count; anything else is a real mismatch.
    """
    stored_rows, doclens_sum = int(stored_rows), int(doclens_sum)
    if stored_rows in (doclens_sum, doclens_sum + COLBERT_PAD_ROWS):
        return doclens_sum
    raise ValueError(f"doclens sum {doclens_sum:,} != stored rows {stored_rows:,} "
                     f"(difference {stored_rows - doclens_sum:,}, expected 0 or {COLBERT_PAD_ROWS})")


# ---------------------------------------------------------------------------
# k-means and quantization (numpy, chunked; 3D data so this is cheap)
# ---------------------------------------------------------------------------
def eid_dtype(num_entries):
    """On-disk dtype of csr_col_eids for E codewords per subspace."""
    if num_entries <= MAX_EIDS_UINT8:
        return np.uint8
    if num_entries <= MAX_EIDS:
        return np.uint16
    raise ValueError(f"E={num_entries} > {MAX_EIDS} does not fit 16-bit eids")


def assign_nearest(x, centers, chunk=None):
    """Returns (ids int32 [n], squared error float32 [n]). Memory per chunk ~ chunk * E floats."""
    if chunk is None:
        chunk = max(1024, (1 << 26) // max(1, len(centers)))
    x = np.asarray(x, dtype=np.float32)
    centers = np.asarray(centers, dtype=np.float32)
    c_sq = np.square(centers).sum(axis=1)
    ids = np.empty(len(x), dtype=np.int32)
    err = np.empty(len(x), dtype=np.float32)
    for a in range(0, len(x), chunk):
        xb = x[a:a + chunk]
        d = np.square(xb).sum(axis=1, keepdims=True) - 2.0 * (xb @ centers.T) + c_sq[None, :]
        j = np.argmin(d, axis=1)
        ids[a:a + chunk] = j
        err[a:a + chunk] = np.maximum(d[np.arange(len(xb)), j], 0.0)
    return ids, err


def kmeans(x, k, iters=25, seed=0, init_sample=200_000):
    """
    Lloyd's algorithm with k-means++ init on a subsample. Empty clusters are
    reseeded from the points with the largest current error. Returns centers [k, d].
    """
    x = np.asarray(x, dtype=np.float32)
    assert len(x) >= k, f"need at least k={k} points, got {len(x)}"
    rng = np.random.default_rng(seed)

    pool = x[rng.choice(len(x), size=min(init_sample, len(x)), replace=False)]
    centers = np.empty((k, x.shape[1]), dtype=np.float32)
    centers[0] = pool[rng.integers(len(pool))]
    d2 = np.square(pool - centers[0]).sum(axis=1)
    for j in range(1, k):
        p = d2 / d2.sum() if d2.sum() > 0 else None
        centers[j] = pool[rng.choice(len(pool), p=p)]
        d2 = np.minimum(d2, np.square(pool - centers[j]).sum(axis=1))

    for _ in range(iters):
        ids, err = assign_nearest(x, centers)
        counts = np.bincount(ids, minlength=k)
        sums = np.zeros_like(centers, dtype=np.float64)
        np.add.at(sums, ids, x)
        nonempty = counts > 0
        centers[nonempty] = (sums[nonempty] / counts[nonempty, None]).astype(np.float32)
        empty = np.flatnonzero(~nonempty)
        if len(empty):
            worst = np.argsort(-err)[:len(empty)]
            centers[empty] = x[worst]
    return centers


# ---------------------------------------------------------------------------
# 24-bit pid packing (the MS MARCO bug lived here)
# ---------------------------------------------------------------------------
def pack_pids_24(pids):
    """int pids -> uint8 [3n], little endian. Fails loudly instead of wrapping."""
    pids = np.asarray(pids)
    assert np.issubdtype(pids.dtype, np.integer), f"pids must be integers, got {pids.dtype}"
    if len(pids):
        lo, hi = int(pids.min()), int(pids.max())
        if lo < 0 or hi > MAX_PID:
            raise OverflowError(f"pid range [{lo}, {hi}] does not fit in {PID_BITS} bits")
    p = pids.astype(np.uint32)
    out = np.empty((len(p), 3), dtype=np.uint8)
    out[:, 0] = p & 0xFF
    out[:, 1] = (p >> 8) & 0xFF
    out[:, 2] = (p >> 16) & 0xFF
    return out.reshape(-1)


def unpack_pids_24(packed):
    b = np.asarray(packed, dtype=np.uint8).reshape(-1, 3).astype(np.uint32)
    return (b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)).astype(np.int64)


# ---------------------------------------------------------------------------
# CSR for one subspace (pure function, unit tested against brute force)
# ---------------------------------------------------------------------------
def build_subspace_csr(cids, eids, pids, num_centroids, col_base, packed_base):
    """
    Group the active tokens of one subspace by (cid, eid), postings sorted by pid.

    cids, eids, pids: per active token. col_base: number of lists already emitted
    by earlier subspaces (row pointers are global). packed_base: number of
    postings already emitted (offsets are global, in postings, not bytes).

    Returns dict with row_ptrs int64 [num_centroids+1], col_eids, lengths,
    offsets (int64, used only for validation), pids_sorted.
    """
    cids = np.asarray(cids, dtype=np.int64)
    eids = np.asarray(eids, dtype=np.int64)
    pids = np.asarray(pids, dtype=np.int64)
    n = len(cids)
    assert len(eids) == n and len(pids) == n
    if n:
        assert cids.min() >= 0 and cids.max() < num_centroids, "cid out of range"
        assert eids.min() >= 0 and eids.max() < (1 << 16), "eid out of range"
        assert pids.min() >= 0 and pids.max() <= MAX_PID, "pid does not fit 24 bits"

    order = np.lexsort((pids, eids, cids))   # by cid, then eid, then pid
    c, e, p = cids[order], eids[order], pids[order]

    if n:
        change = np.concatenate(([True], (c[1:] != c[:-1]) | (e[1:] != e[:-1])))
        starts = np.flatnonzero(change)
    else:
        starts = np.zeros(0, dtype=np.int64)
    lengths = np.diff(np.append(starts, n))
    if len(lengths) and lengths.max() > MAX_LIST_LENGTH:
        raise OverflowError(f"posting list of length {lengths.max()} does not fit uint16")

    list_cids = c[starts]
    row_ptrs = np.empty(num_centroids + 1, dtype=np.int64)
    row_ptrs[0] = col_base
    row_ptrs[1:] = col_base + np.cumsum(np.bincount(list_cids, minlength=num_centroids))

    return {
        "row_ptrs": row_ptrs,
        "col_eids": e[starts],
        "lengths": lengths.astype(np.int64),
        "offsets": starts.astype(np.int64) + packed_base,
        "pids_sorted": p,
    }


def block_sums(lengths, block=BLOCK_SIZE):
    """block_sums[b] = sum(lengths[:block*b]), matching the Stage 2 kernel."""
    lengths = np.asarray(lengths, dtype=np.int64)
    nb = (len(lengths) + block - 1) // block
    padded = np.zeros(nb * block, dtype=np.int64)
    padded[:len(lengths)] = lengths
    totals = padded.reshape(nb, block).sum(axis=1)
    out = np.zeros(nb, dtype=np.int64)
    out[1:] = np.cumsum(totals[:-1])
    return out


def offsets_from_block_sums(bsums, lengths, list_ids, block=BLOCK_SIZE):
    """Same arithmetic as the kernel: block sum plus the lengths before list_id in its block."""
    lengths = np.asarray(lengths, dtype=np.int64)
    out = np.empty(len(list_ids), dtype=np.int64)
    for i, lid in enumerate(list_ids):
        b = lid // block
        out[i] = bsums[b] + lengths[b * block: lid].sum()
    return out


# ---------------------------------------------------------------------------
# Query rows used for candidate generation (single definition for all callers)
# ---------------------------------------------------------------------------
QUERY_ROWS_DEFAULT = "mask"


def query_ntok(searcher, query_text, mode=QUERY_ROWS_DEFAULT):
    """
    Number of leading ColBERT query rows used for candidate generation
    (Stage 4 always reranks with all 32 rows).

      "mask"   : real wordpiece tokens incl. [CLS], [Q], [SEP] (tokenizer attention mask).
      "all"    : all 32 rows, including the [MASK] augmentation rows (what PLAID probes).
      "legacy" : min(words + 4, 32). Undercounts wordpieces, e.g. 10 vs 13 real tokens
                 for "is sudan iv hydrophobic or hydrophilic?". Kept only to reproduce
                 numbers from before Oct 2026.
    `searcher` only needs .checkpoint.query_tokenizer.tensorize([text]) -> (ids, mask).
    """
    if mode == "all":
        return 32
    if mode == "legacy":
        return min(len(query_text.split()) + 4, 32)
    if mode != "mask":
        raise ValueError(f"unknown query_ntok mode {mode!r}")
    _, mask = searcher.checkpoint.query_tokenizer.tensorize([query_text])
    return int(min(int(mask[0].sum()), 32))
