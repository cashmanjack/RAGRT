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
    assert E <= L.MAX_EIDS_UINT8, (f"E={E} > 256: csr_col_eids is uint8 and the kernels "
                                   f"assume 256 entries; widen them before building.")
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

    codes = searcher.ranker.embeddings.codes.numpy()
    residuals_packed = searcher.ranker.embeddings.residuals.numpy()
    total_tokens = codes.shape[0]

    doclens = np.asarray(flatten(load_doclens(args.index, flatten=False)), dtype=np.int64)
    num_passages = len(doclens)
    if num_passages - 1 > L.MAX_PID:
        sys.exit(f"FATAL: {num_passages:,} passages exceed the 24-bit pid format.")
    if doclens.sum() != total_tokens:
        sys.exit(f"FATAL: doclens sum {doclens.sum():,} != token count {total_tokens:,}. "
                 f"Refusing to pad or truncate pids.")
    pids_all = np.repeat(np.arange(num_passages, dtype=np.int32), doclens)
    print(f"Corpus: {num_passages:,} passages, {total_tokens:,} tokens, {num_centroids:,} centroids")

    tmp_eids = os.path.join(args.outdir, "all_eids_tmp.mmap")
    tmp_mask = os.path.join(args.outdir, "active_mask_tmp.mmap")
    all_eids = np.memmap(tmp_eids, dtype=np.uint8, mode="w+", shape=(L.NUM_SUBSPACES, total_tokens))
    active = np.memmap(tmp_mask, dtype=bool, mode="w+", shape=(L.NUM_SUBSPACES, total_tokens))

    print(f"Quantizing residual slices (top_m={args.top_m})...")
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
            active[s, start:end] = m
            if m.any():
                ids, _ = L.assign_nearest(L.subspace_slice(resid[m], s), codebooks[s])
                row = np.zeros(end - start, dtype=np.uint8)
                row[m] = ids
                all_eids[s, start:end] = row
        if (start // CHUNK) % 10 == 0 or end == total_tokens:
            print(f"  {end:,} / {total_tokens:,} tokens")
    all_eids.flush(); active.flush()

    del residuals_packed, searcher
    gc.collect(); torch.cuda.empty_cache()

    total_postings = int(sum(int(np.count_nonzero(active[s])) for s in range(L.NUM_SUBSPACES)))
    print(f"Compiling CSR: {total_postings:,} postings")
    packed_path = os.path.join(args.outdir, "csr_packed_24.npy")
    packed_out = np.lib.format.open_memmap(packed_path, mode="w+", dtype=np.uint8,
                                           shape=(3 * total_postings,))

    row_ptrs_all, col_eids_all, lengths_all, offsets_all = [], [], [], []
    col_base, packed_base = 0, 0
    for s in range(L.NUM_SUBSPACES):
        m = np.asarray(active[s])
        csr = L.build_subspace_csr(codes[m], np.asarray(all_eids[s])[m], pids_all[m],
                                   num_centroids, col_base, packed_base)
        n_s = len(csr["pids_sorted"])
        packed = L.pack_pids_24(csr["pids_sorted"])
        # Round trip check on every subspace: decoding must give back the exact pids.
        assert np.array_equal(L.unpack_pids_24(packed), csr["pids_sorted"]), f"pid round trip failed in subspace {s}"
        packed_out[3 * packed_base: 3 * (packed_base + n_s)] = packed

        row_ptrs_all.append(csr["row_ptrs"]); col_eids_all.append(csr["col_eids"])
        lengths_all.append(csr["lengths"]);   offsets_all.append(csr["offsets"])
        col_base += len(csr["col_eids"]); packed_base += n_s
        del csr, packed, m; gc.collect()
    packed_out.flush(); del packed_out
    assert packed_base == total_postings

    row_ptrs = np.stack(row_ptrs_all)
    if row_ptrs.max() > np.iinfo(np.int32).max:
        sys.exit(f"FATAL: {row_ptrs.max():,} lists overflow int32 row pointers.")
    lengths = np.concatenate(lengths_all)
    offsets = np.concatenate(offsets_all)
    bsums = L.block_sums(lengths)

    # Kernel offset reconstruction must reproduce the true offsets.
    rng = np.random.default_rng(0)
    probe = np.unique(np.concatenate([[0, len(lengths) - 1], rng.integers(0, len(lengths), 200_000)]))
    assert np.array_equal(L.offsets_from_block_sums(bsums, lengths, probe), offsets[probe]), \
        "block-sum offset reconstruction mismatch"

    np.save(os.path.join(args.outdir, "csr_row_ptrs.npy"), row_ptrs.astype(np.int32))
    np.save(os.path.join(args.outdir, "csr_col_eids.npy"), np.concatenate(col_eids_all).astype(np.uint8))
    np.save(os.path.join(args.outdir, "csr_lengths.npy"), lengths.astype(np.uint16))
    np.save(os.path.join(args.outdir, "csr_block_sums_128.npy"), bsums)

    del all_eids, active
    os.remove(tmp_eids); os.remove(tmp_mask)
    print(f"[DONE] {len(lengths):,} lists, {total_postings:,} postings "
          f"({3 * total_postings / 1e9:.2f} GB packed) in {args.outdir}")


if __name__ == "__main__":
    main()
