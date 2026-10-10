"""
CPU end-to-end test of train_codebooks.py + build_full_lotte_sparse_csr.py on a fake
ColBERT index (no GPU, no ColBERT). Uses E=300 so the 16-bit eid path is exercised,
then checks every posting list against a brute-force recomputation, and runs check_index.
Run: python3 tests/test_builder_cpu.py
"""
import os, sys, types, tempfile, json
import numpy as np
import torch

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
sys.path.insert(0, SRC)
torch.Tensor.cuda = lambda self, *a, **k: self          # CPU stand-in

N_DOCS, NC, E = 2000, 160, 300   # NC >= 96 for the SVD rotation
RNG = np.random.default_rng(0)
DOCLENS = RNG.integers(5, 40, size=N_DOCS)
T = int(DOCLENS.sum())
CENT = torch.nn.functional.normalize(torch.tensor(RNG.normal(size=(NC, 128)), dtype=torch.float32), dim=-1)
CODES = torch.tensor(np.concatenate([RNG.integers(0, NC, T), np.zeros(512)]), dtype=torch.int32)   # + ColBERT pad
RESID = torch.tensor(RNG.integers(0, 256, size=(T + 512, 32)), dtype=torch.uint8)


def decompress(obj):
    noise = (obj.residuals.float() - 127.5) / 600.0                 # [n, 32] -> spread over 128 dims
    return torch.nn.functional.normalize(CENT[obj.codes.long()] + noise.repeat(1, 4), dim=-1)


class ResidualEmbeddings:
    def __init__(self, codes, residuals):
        self.codes, self.residuals = codes, residuals


class Searcher:
    def __init__(self, index, collection):
        codec = types.SimpleNamespace(centroids=CENT, decompress=decompress)
        self.ranker = types.SimpleNamespace(codec=codec, embeddings=types.SimpleNamespace(codes=CODES, residuals=RESID))


colbert = types.ModuleType("colbert"); colbert.Searcher = Searcher
mods = {"colbert": colbert, "colbert.indexing": types.ModuleType("colbert.indexing"),
        "colbert.indexing.loaders": types.ModuleType("colbert.indexing.loaders"),
        "colbert.indexing.codecs": types.ModuleType("colbert.indexing.codecs"),
        "colbert.indexing.codecs.residual_embeddings": types.ModuleType("colbert.indexing.codecs.residual_embeddings"),
        "colbert.utils": types.ModuleType("colbert.utils"), "colbert.utils.utils": types.ModuleType("colbert.utils.utils")}
mods["colbert.indexing.loaders"].load_doclens = lambda index, flatten=False: [DOCLENS[:150].tolist(), DOCLENS[150:].tolist()]
mods["colbert.indexing.codecs.residual_embeddings"].ResidualEmbeddings = ResidualEmbeddings
mods["colbert.utils.utils"].flatten = lambda L: [x for l in L for x in l]
sys.modules.update(mods)

import ragrt_index_lib as L
import gpu_kmeans as G


def main():
    out = tempfile.mkdtemp(prefix="ragrt_builder_")
    import train_codebooks, build_full_lotte_sparse_csr as B, build_predicates, check_index
    sys.argv = ["train_codebooks.py", "--index", "x", "--collection", "y", "--outdir", out,
                "--num_entries", str(E), "--top_m", "8", "--iters", "5", "--sample_tokens", "100000"]
    train_codebooks.main()
    B.CHUNK = 16384                                         # several chunks, multiple of 8
    sys.argv = ["build.py", "--index", "x", "--collection", "y", "--outdir", out, "--top_m", "8"]
    B.main()
    assert not [f for f in os.listdir(out) if "tmp" in f], "temp files left behind"

    eids = np.load(os.path.join(out, "csr_col_eids.npy"))
    rp = np.load(os.path.join(out, "csr_row_ptrs.npy"))
    ln = np.load(os.path.join(out, "csr_lengths.npy"))
    bs = np.load(os.path.join(out, "csr_block_sums_128.npy"))
    pk = L.unpack_pids_24(np.load(os.path.join(out, "csr_packed_24.npy")))
    assert eids.dtype == np.uint16 and rp.dtype == np.int64, (eids.dtype, rp.dtype)

    # brute force: same rotation, residuals, mask and nearest codeword
    R = np.load(os.path.join(out, "svd_rotation_128_to_96.npy"))
    cb = np.load(os.path.join(out, "codebooks.npy"))
    c96 = L.l2_normalize(CENT.numpy() @ R)
    emb = decompress(ResidualEmbeddings(CODES[:T], RESID[:T])).numpy() @ R
    emb = L.l2_normalize(emb)
    mask = L.top_m_active_mask(emb, 8)
    resid = L.residual_96(emb, c96, CODES[:T].numpy())
    pids = np.repeat(np.arange(N_DOCS), DOCLENS)
    lists = {}
    for s in range(32):
        m = mask[:, s]
        ids, _ = G.assign_np(L.subspace_slice(resid[m], s), cb[s])
        for c, e, p in zip(CODES[:T].numpy()[m], ids, pids[m]):
            lists.setdefault((s, int(c), int(e)), []).append(int(p))
    n_checked = 0
    for s in range(32):
        for cid in range(NC):
            for li in range(rp[s, cid], rp[s, cid + 1]):
                off = int(L.offsets_from_block_sums(bs, ln, [li])[0])
                got = pk[off: off + int(ln[li])].tolist()
                assert got == sorted(lists.pop((s, cid, int(eids[li])))), (s, cid, li)
                n_checked += 1
    assert not lists, f"{len(lists)} lists missing from the CSR"
    print(f"checked {n_checked} lists against brute force")

    sys.argv = ["build_predicates.py", "--outdir", out, "--num_passages", str(N_DOCS)]
    build_predicates.main()
    assert check_index.check(out)
    print("BUILDER TEST PASSED")


if __name__ == "__main__":
    main()
