"""
Build the RAGRT sparse CSR index for one ColBERT index (LoTTE, MS MARCO, ...).

Requires codebooks trained for THIS index by train_codebooks.py (same outdir).
The rotation fingerprint and top_m recorded by the trainer are checked here, so
codebooks from another index or another rotation are rejected.

Writes csr_row_ptrs, csr_col_eids, csr_lengths, csr_block_sums_128 and
csr_packed_24 directly (see ragrt_index_lib.py for the format). Token positions
are not stored: the runtime never reads them.
"""
import os, sys, json, argparse, gc
import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "../reference/colbert-plaid"))

from colbert import Searcher
from colbert.indexing.loaders import load_doclens
from colbert.indexing.codecs.residual_embeddings import ResidualEmbeddings
from colbert.utils.utils import flatten

import ragrt_index_lib as L
import gpu_kmeans as G

CHUNK = 2_000_000


def load_matching_codebooks(outdir, R, top_m):
    cb_path = os.path.join(outdir, "codebooks.npy")
    meta_path = os.path.join(outdir, "codebook_meta.json")
    if not (os.path.exists(cb_path) and os.path.exists(meta_path)):
        sys.exit(f"FATAL: {cb_path} / codebook_meta.json missing. Run train_codebooks.py "
                 f"for this index with --outdir {outdir} first.")
    meta = json.load(open(meta_path))
    fp = L.rotation_fingerprint(R)
    if meta.get("rotation_sha256") != fp:
        sys.exit("FATAL: codebooks were trained in a different 128->96 rotation than this "
                 "index uses. Retrain with train_codebooks.py.")
    if meta.get("top_m") != top_m:
        sys.exit(f"FATAL: codebooks trained with top_m={meta.get('top_m')}, building with "
                 f"top_m={top_m}. They must match.")
    codebooks = np.load(cb_path).astype(np.float32)
    S, E, D = codebooks.shape
    assert S == L.NUM_SUBSPACES and D == L.SUBSPACE_DIM, f"bad codebook shape {codebooks.shape}"
    L.eid_dtype(E)   # raises if E does not fit 16 bits
    return codebooks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--collection", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--top_m", type=int, default=10)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print(f"Loading searcher from {args.index}...")
    searcher = Searcher(index=args.index, collection=args.collection)
    centroids_128 = searcher.ranker.codec.centroids.detach().cpu().float().numpy()
    num_centroids = centroids_128.shape[0]

    rot_path = os.path.join(args.outdir, "svd_rotation_128_to_96.npy")
    if not os.path.exists(rot_path):
        sys.exit(f"FATAL: {rot_path} missing. train_codebooks.py creates it; run that first.")
    R = np.load(rot_path).astype(np.float32)
    codebooks = load_matching_codebooks(args.outdir, R, args.top_m)

    centroids_96 = L.l2_normalize(centroids_128 @ R)
    np.save(os.path.join(args.outdir, "centroids_96d_svd.npy"), centroids_96)

    doclens = np.asarray(flatten(load_doclens(args.index, flatten=False)), dtype=np.int64)
    num_passages = len(doclens)
    if num_passages - 1 > L.MAX_PID:
        sys.exit(f"FATAL: {num_passages:,} passages exceed the 24-bit pid format.")

    codes = searcher.ranker.embeddings.codes.numpy()
    residuals_packed = searcher.ranker.embeddings.residuals.numpy()
    try:
        total_tokens = L.num_real_tokens(codes.shape[0], doclens.sum())
    except ValueError as e:
        sys.exit(f"FATAL: {e}. Refusing to pad or truncate pids.")
    if total_tokens != codes.shape[0]:
        print(f"Dropping {codes.shape[0] - total_tokens} ColBERT padding rows")
    codes = codes[:total_tokens]
    residuals_packed = residuals_packed[:total_tokens]
    pids_all = np.repeat(np.arange(num_passages, dtype=np.int32), doclens)
    print(f"Corpus: {num_passages:,} passages, {total_tokens:,} tokens, {num_centroids:,} centroids")

    E = codebooks.shape[1]
    edt = L.eid_dtype(E)
    print(f"E={E} codewords per subspace -> {np.dtype(edt).name} eids")
    # Temp files: the active mask as bits [32, T/8] and, per subspace, the eids of its active
    # tokens in token order (top_m of 32 subspaces are active, so ~top_m * 2 bytes per token).
    tmp_mask = os.path.join(args.outdir, "active_bits_tmp.mmap")
    tmp_eid_paths = [os.path.join(args.outdir, f"eids_s{s:02d}_tmp.bin") for s in range(L.NUM_SUBSPACES)]
    active_bits = np.memmap(tmp_mask, dtype=np.uint8, mode="w+", shape=(L.NUM_SUBSPACES, (total_tokens + 7) // 8))
    eid_files = [open(p, "wb") for p in tmp_eid_paths]

    print(f"Quantizing residual slices (top_m={args.top_m})...")
    assert CHUNK % 8 == 0
    R_cuda = torch.from_numpy(R).cuda()
    for start in range(0, total_tokens, CHUNK):
        end = min(start + CHUNK, total_tokens)
        obj = ResidualEmbeddings(torch.from_numpy(codes[start:end]), torch.from_numpy(residuals_packed[start:end]))
        emb_128 = searcher.ranker.codec.decompress(obj).cuda().float()
        emb_96 = torch.nn.functional.normalize(emb_128 @ R_cuda, p=2, dim=-1).cpu().numpy()
        mask = L.top_m_active_mask(emb_96, args.top_m)
        resid = L.residual_96(emb_96, centroids_96, codes[start:end])
        for s in range(L.NUM_SUBSPACES):
            m = mask[:, s]
            bits = np.packbits(m)
            active_bits[s, start // 8: start // 8 + len(bits)] = bits
            if m.any():
                ids, _ = G.assign_np(L.subspace_slice(resid[m], s), codebooks[s])
                eid_files[s].write(ids.astype(edt).tobytes())
        if (start // CHUNK) % 10 == 0 or end == total_tokens:
            print(f"  {end:,} / {total_tokens:,} tokens")
    active_bits.flush()
    for f in eid_files:
        f.close()

    del residuals_packed, searcher
    gc.collect(); torch.cuda.empty_cache()

    def active_of(s):
        return np.unpackbits(np.asarray(active_bits[s]), count=total_tokens).astype(bool)

    total_postings = int(sum(os.path.getsize(p) // np.dtype(edt).itemsize for p in tmp_eid_paths))
    print(f"Compiling CSR: {total_postings:,} postings")
    packed_path = os.path.join(args.outdir, "csr_packed_24.npy")
    packed_out = np.lib.format.open_memmap(packed_path, mode="w+", dtype=np.uint8,
                                           shape=(3 * total_postings,))

    row_ptrs_all, col_eids_all, lengths_all = [], [], []
    probe_ids, probe_offs = [], []           # sample of (global list id, true offset) for validation
    rng = np.random.default_rng(0)
    col_base, packed_base = 0, 0
    for s in range(L.NUM_SUBSPACES):
        m = active_of(s)
        eids_s = np.fromfile(tmp_eid_paths[s], dtype=edt)
        assert len(eids_s) == int(m.sum()), f"subspace {s}: {len(eids_s)} eids for {int(m.sum())} active tokens"
        csr = L.build_subspace_csr(codes[m], eids_s, pids_all[m], num_centroids, col_base, packed_base)
        n_s = len(csr["pids_sorted"])
        packed = L.pack_pids_24(csr["pids_sorted"])
        # Round trip check on every subspace: decoding must give back the exact pids.
        assert np.array_equal(L.unpack_pids_24(packed), csr["pids_sorted"]), f"pid round trip failed in subspace {s}"
        packed_out[3 * packed_base: 3 * (packed_base + n_s)] = packed

        n_lists = len(csr["lengths"])
        row_ptrs_all.append(csr["row_ptrs"])
        col_eids_all.append(csr["col_eids"].astype(edt))
        lengths_all.append(csr["lengths"].astype(np.uint16))   # build_subspace_csr checked the range
        if n_lists:
            pr = np.unique(np.concatenate([[0, n_lists - 1], rng.integers(0, n_lists, 8000)]))
            probe_ids.append(col_base + pr); probe_offs.append(csr["offsets"][pr])
        print(f"  subspace {s:2d}: {n_lists:,} lists, {n_s:,} postings", flush=True)
        col_base += n_lists; packed_base += n_s
        del csr, packed, m, eids_s; gc.collect()
    packed_out.flush(); del packed_out
    assert packed_base == total_postings

    row_ptrs = np.stack(row_ptrs_all).astype(np.int64)
    lengths = np.concatenate(lengths_all); del lengths_all
    bsums = L.block_sums(lengths)

    # Kernel offset reconstruction must reproduce the true offsets.
    probe, want = np.concatenate(probe_ids), np.concatenate(probe_offs)
    assert np.array_equal(L.offsets_from_block_sums(bsums, lengths, probe), want), \
        "block-sum offset reconstruction mismatch"

    np.save(os.path.join(args.outdir, "csr_row_ptrs.npy"), row_ptrs)
    np.save(os.path.join(args.outdir, "csr_col_eids.npy"), np.concatenate(col_eids_all))
    np.save(os.path.join(args.outdir, "csr_lengths.npy"), lengths)
    np.save(os.path.join(args.outdir, "csr_block_sums_128.npy"), bsums)

    del active_bits
    os.remove(tmp_mask)
    for p in tmp_eid_paths:
        os.remove(p)
    print(f"[DONE] {len(lengths):,} lists, {total_postings:,} postings "
          f"({3 * total_postings / 1e9:.2f} GB packed) in {args.outdir}")


if __name__ == "__main__":
    main()
