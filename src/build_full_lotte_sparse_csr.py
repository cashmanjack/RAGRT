import os, sys, shutil, torch, numpy as np, argparse
from scipy.spatial import cKDTree

BASE_DIR = "/home/min/a/cashman3/RTRAG/src"
sys.path.insert(0, os.path.join(BASE_DIR, "../reference/colbert-plaid"))

from colbert import Searcher
from colbert.indexing.loaders import load_doclens
from colbert.indexing.codecs.residual_embeddings import ResidualEmbeddings
from colbert.utils.utils import flatten

NUM_SUBSPACES = 32
NUM_ENTRIES   = 256
CHUNK         = 2_000_000

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="/home/min/a/cashman3/RTRAG/src/experiments/unified_lotte/indexes/unified.dev.2bit")
    ap.add_argument("--collection", default="/local/scratch/a/cashman3/lotte/unified/collection.tsv")
    ap.add_argument("--outdir", default="/local/scratch/a/cashman3/juno_pq_lotte_full_sparse8")
    ap.add_argument("--top_m", type=int, default=10, help="Top-M subspaces (M=10 recovers full recall)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    print("=" * 90)
    print(f"BUILDING FULL LOTTE 2.4M PASSAGE SPARSE CSR INDEX (Top-{args.top_m})")
    print("=" * 90)

    print(f"Loading searcher from {args.index}...")
    searcher = Searcher(index=args.index, collection=args.collection)
    
    centroids_128 = searcher.ranker.codec.centroids.detach().cpu().float()
    MAX_CID = centroids_128.shape[0]

    rot_path = os.path.join(args.outdir, "svd_rotation_128_to_96.npy")
    if os.path.exists(rot_path):
        print("Using existing SVD rotation matrix R...")
        R = torch.from_numpy(np.load(rot_path)).float()
    else:
        print("Computing SVD projection matrix R...")
        centered = centroids_128 - centroids_128.mean(dim=0, keepdim=True)
        _, S, Vh = torch.linalg.svd(centered, full_matrices=False)
        R = Vh[:96, :].T.contiguous().float()
        np.save(rot_path, R.numpy())

    centroids_96 = torch.nn.functional.normalize(centroids_128.cuda() @ R.cuda(), p=2, dim=-1).cpu().numpy()
    np.save(os.path.join(args.outdir, "centroids_96d_svd.npy"), centroids_96)

    codes = searcher.ranker.embeddings.codes.numpy()
    residuals_packed = searcher.ranker.embeddings.residuals.numpy()
    total_tokens = codes.shape[0]
    print(f"Full LoTTE Corpus: {total_tokens:,} tokens | Coarse Centroids: {MAX_CID}")

    doclens = flatten(load_doclens(args.index, flatten=False))
    dl = torch.tensor(doclens, dtype=torch.long)
    pids_all = torch.repeat_interleave(torch.arange(len(dl), dtype=torch.int32), dl).numpy()
    offs = torch.cumsum(dl, 0)
    starts = torch.cat([torch.zeros(1, dtype=torch.long), offs[:-1]])
    pos_all = (torch.arange(len(pids_all), dtype=torch.long) - torch.repeat_interleave(starts, dl)).numpy().astype(np.int16)

    if len(pids_all) != total_tokens:
        pids_all = pids_all[:total_tokens] if len(pids_all) > total_tokens else np.pad(pids_all, (0, total_tokens - len(pids_all)))
        pos_all  = pos_all[:total_tokens]  if len(pos_all)  > total_tokens else np.pad(pos_all,  (0, total_tokens - len(pos_all)))

    cb_path = os.path.join(args.outdir, "codebooks.npy")
    if not os.path.exists(cb_path):
        src_cb = "/local/scratch/a/cashman3/juno_pq_science_sparse8/codebooks.npy"
        if not os.path.exists(src_cb):
            src_cb = "/local/scratch/a/cashman3/juno_pq_science_3d/codebooks.npy"
        shutil.copyfile(src_cb, cb_path)
        print(f"Copied codebooks from {src_cb}")

    codebooks = np.load(cb_path)
    trees = [cKDTree(codebooks[s]) for s in range(NUM_SUBSPACES)]

    print(f"Quantizing tokens with Semantic-Energy Top-{args.top_m} Subspace Allocation...")
    all_eids = np.zeros((NUM_SUBSPACES, total_tokens), dtype=np.int16)
    active_mask = np.zeros((NUM_SUBSPACES, total_tokens), dtype=bool)

    R_cuda = R.cuda()
    for start in range(0, total_tokens, CHUNK):
        end = min(start + CHUNK, total_tokens)
        obj = ResidualEmbeddings(torch.tensor(codes[start:end]), torch.tensor(residuals_packed[start:end]))
        
        # 1. Full 128D -> Optimal 96D SVD Projection
        emb_128 = searcher.ranker.codec.decompress(obj).cuda().float()
        emb_96 = torch.nn.functional.normalize(emb_128 @ R_cuda, p=2, dim=-1).cpu().numpy()
        
        # 2. Compute TRUE semantic energy directly from emb_96
        emb_reshaped = emb_96.reshape(-1, NUM_SUBSPACES, 3)
        emb_sq_norms = np.sum(emb_reshaped**2, axis=-1)

        # 3. Select Top-M highest energy subspaces
        top_m_idx = np.argpartition(-emb_sq_norms, args.top_m, axis=-1)[:, :args.top_m]

        # 4. Quantize residuals in active subspaces
        resid = emb_96 - centroids_96[codes[start:end]]

        for s in range(NUM_SUBSPACES):
            mask_s = np.any(top_m_idx == s, axis=-1)
            active_mask[s, start:end] = mask_s
            if np.any(mask_s):
                sub_vecs = resid[mask_s, 3*s : 3*s + 3]
                _, eids = trees[s].query(sub_vecs)
                all_eids[s, start:end][mask_s] = eids.astype(np.int16)

        print(f"  Processed {end:,} / {total_tokens:,} tokens ({end / total_tokens * 100:.1f}%)")

    print("\nCompiling Unified Full-Corpus CSR Tables...")
    all_row_ptrs, all_col_eids, all_offsets, all_lengths, all_packed = [], [], [], [], []
    curr_global_col, curr_global_packed = 0, 0

    for s in range(NUM_SUBSPACES):
        mask_s = active_mask[s]
        cids_s = codes[mask_s]
        eids_s = all_eids[s, mask_s]
        pids_s = pids_all[mask_s]
        pos_s  = pos_all[mask_s]
        num_s_tokens = len(cids_s)

        # Sort by (cid, eid, pid): rows group (cid, eid); postings inside each row
        # are additionally ordered by pid so the GPU can skip repeated pids
        # for the same passage with a cheap cross-lane compare.
        sort_keys = (cids_s.astype(np.int64) << 40) | (eids_s.astype(np.int64) << 24) | pids_s.astype(np.int64)
        order = np.argsort(sort_keys, kind="stable")

        cids_sorted = cids_s[order]
        eids_sorted = eids_s[order]
        pids_sorted = pids_s[order]
        pos_sorted  = pos_s[order]

        change = np.concatenate(([True], (cids_sorted[1:] != cids_sorted[:-1]) | (eids_sorted[1:] != eids_sorted[:-1])))
        boundaries = np.nonzero(change)[0]

        unique_cids = cids_sorted[boundaries]
        unique_eids = eids_sorted[boundaries]
        starts_b = boundaries
        ends_b = np.append(boundaries[1:], num_s_tokens)
        lengths_b = ends_b - starts_b

        row_ptrs = np.zeros(MAX_CID + 1, dtype=np.int32)
        counts = np.bincount(unique_cids, minlength=MAX_CID)
        row_ptrs[0] = curr_global_col
        row_ptrs[1:] = curr_global_col + np.cumsum(counts)

        col_eids = unique_eids.astype(np.uint8)
        active_lengths = lengths_b.astype(np.uint16)
        active_offsets = (starts_b + curr_global_packed).astype(np.int64)
        # Packing contract: (pid << 9) | position. Token positions MUST fit in 9
        # bits; fail loudly at build time instead of silently corrupting pids.
        max_pos = int(pos_sorted.max()) if len(pos_sorted) else 0
        assert max_pos < 512, (
            f"token position {max_pos} does not fit in the 9-bit packing; "
            "widen the packing before building")
        packed_s = (pids_sorted.astype(np.int32) << 9) | pos_sorted.astype(np.int32)

        all_row_ptrs.append(row_ptrs)
        all_col_eids.append(col_eids)
        all_offsets.append(active_offsets)
        all_lengths.append(active_lengths)
        all_packed.append(packed_s)

        curr_global_col += len(col_eids)
        curr_global_packed += len(packed_s)

    flat_row_ptrs = np.stack(all_row_ptrs)
    flat_col_eids = np.concatenate(all_col_eids)
    flat_offsets  = np.concatenate(all_offsets)
    flat_lengths  = np.concatenate(all_lengths)
    flat_packed   = np.concatenate(all_packed)

    np.save(os.path.join(args.outdir, "csr_row_ptrs.npy"), flat_row_ptrs)
    np.save(os.path.join(args.outdir, "csr_col_eids.npy"), flat_col_eids)
    np.save(os.path.join(args.outdir, "csr_offsets.npy"), flat_offsets)
    np.save(os.path.join(args.outdir, "csr_lengths.npy"), flat_lengths)
    np.save(os.path.join(args.outdir, "csr_packed.npy"), flat_packed)

    meta_mb = (flat_row_ptrs.nbytes + flat_col_eids.nbytes + flat_offsets.nbytes + flat_lengths.nbytes) / 1e6
    packed_mb = flat_packed.nbytes / 1e6
    print(f"\n[DONE] Full LoTTE CSR Index Compiled:")
    print(f"  Total CSR Metadata  : {meta_mb:.2f} MB")
    print(f"  Packed Postings     : {packed_mb:.2f} MB")
    print(f"  Total VRAM Footprint: {(meta_mb + packed_mb)/1024:.2f} GB")

if __name__ == "__main__":
    main()
