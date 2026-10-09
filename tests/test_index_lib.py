"""
CPU-only tests for src/ragrt_index_lib.py. Run with:  python3 tests/test_index_lib.py
(also works under pytest). No GPU, torch or ColBERT needed.
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import ragrt_index_lib as L


def test_pid_round_trip_across_old_overflow_boundaries():
    pids = np.array([0, 1, 255, 256, 65_535, 65_536, 4_194_303, 4_194_304,
                     6_000_000, 8_388_607, 8_388_608, 8_841_822, L.MAX_PID], dtype=np.int64)
    assert np.array_equal(L.unpack_pids_24(L.pack_pids_24(pids)), pids)


def test_old_packing_was_broken():
    # Documents the bug this replaces: int32 (pid << 9) then >> 9 wraps above 2^22.
    pids = np.array([4_194_303, 4_194_304, 8_388_608], dtype=np.int64)
    old = ((pids.astype(np.int32) << 9) >> 9).astype(np.uint32) & 0xFFFFFF
    assert old[0] == 4_194_303 and old[1] != 4_194_304 and old[2] != 8_388_608


def test_pid_overflow_fails_loudly():
    for bad in ([L.MAX_PID + 1], [-1]):
        try:
            L.pack_pids_24(np.array(bad))
        except OverflowError:
            continue
        raise AssertionError(f"pack_pids_24 accepted {bad}")


def _brute_force_lists(cids, eids, pids):
    lists = {}
    for c, e, p in zip(cids, eids, pids):
        lists.setdefault((int(c), int(e)), []).append(int(p))
    return {k: sorted(v) for k, v in lists.items()}


def test_synthetic_csr_matches_brute_force():
    rng = np.random.default_rng(0)
    num_centroids, S = 7, 3
    col_base = packed_base = 0
    all_lengths, all_offsets, all_pids, rows = [], [], [], []
    for s in range(S):
        n = 500
        cids = rng.integers(0, num_centroids, n)
        cids[cids == 3] = 4                      # centroid 3 has no lists
        eids = rng.integers(0, 6, n)
        pids = rng.integers(0, L.MAX_PID + 1, n)  # include pids far above 2^22
        csr = L.build_subspace_csr(cids, eids, pids, num_centroids, col_base, packed_base)
        expect = _brute_force_lists(cids, eids, pids)

        # every (cid, eid) list is found through row_ptrs + col_eids exactly as Stage 2 does
        for (c, e), plist in expect.items():
            r0, r1 = csr["row_ptrs"][c] - col_base, csr["row_ptrs"][c + 1] - col_base
            hits = np.flatnonzero(csr["col_eids"][r0:r1] == e)
            assert len(hits) == 1, (s, c, e)
            j = r0 + hits[0]
            off = csr["offsets"][j] - packed_base
            got = csr["pids_sorted"][off: off + csr["lengths"][j]]
            assert got.tolist() == plist, (s, c, e)
        assert csr["row_ptrs"][3] == csr["row_ptrs"][4]           # empty row
        assert len(csr["lengths"]) == len(expect)
        # eids are sorted within each row (binary search requirement)
        for c in range(num_centroids):
            r0, r1 = csr["row_ptrs"][c] - col_base, csr["row_ptrs"][c + 1] - col_base
            assert np.all(np.diff(csr["col_eids"][r0:r1]) > 0)

        all_lengths.append(csr["lengths"]); all_offsets.append(csr["offsets"])
        all_pids.append(csr["pids_sorted"]); rows.append(csr["row_ptrs"])
        col_base += len(csr["lengths"]); packed_base += n

    lengths = np.concatenate(all_lengths); offsets = np.concatenate(all_offsets)
    # global offsets are contiguous across subspaces
    assert offsets[0] == 0 and np.array_equal(offsets[1:], np.cumsum(lengths)[:-1])
    # kernel-style block-sum reconstruction equals true offsets for every list
    bs = L.block_sums(lengths, block=8)
    assert np.array_equal(L.offsets_from_block_sums(bs, lengths, np.arange(len(lengths)), block=8), offsets)
    # 24-bit file round trip on the concatenated postings
    flat = np.concatenate(all_pids)
    assert np.array_equal(L.unpack_pids_24(L.pack_pids_24(flat)), flat)


def test_list_length_overflow_fails_loudly():
    n = L.MAX_LIST_LENGTH + 1
    try:
        L.build_subspace_csr(np.zeros(n, int), np.zeros(n, int), np.arange(n), 1, 0, 0)
    except OverflowError:
        return
    raise AssertionError("uint16 list length overflow not caught")


def test_top_m_mask_matches_original_rule():
    rng = np.random.default_rng(1)
    emb = L.l2_normalize(rng.normal(size=(1000, L.PROJ_DIM)))
    m = 10
    mask = L.top_m_active_mask(emb, m)
    energy = np.square(emb.reshape(-1, 32, 3)).sum(-1)
    orig = np.argpartition(-energy, m, axis=-1)[:, :m]        # rule used by the old builder
    ref = np.zeros_like(mask); np.put_along_axis(ref, orig, True, axis=-1)
    assert np.array_equal(mask, ref) and np.all(mask.sum(1) == m)


def test_kmeans_recovers_clusters_and_reports_error():
    rng = np.random.default_rng(2)
    true = rng.normal(size=(16, 3)).astype(np.float32)
    x = true[rng.integers(0, 16, 20_000)] + 0.01 * rng.normal(size=(20_000, 3)).astype(np.float32)
    centers = L.kmeans(x, 16, iters=15, seed=0)
    _, err = L.assign_nearest(x, centers)
    assert err.mean() < 1e-3, err.mean()          # noise variance is 3e-4
    _, d = L.assign_nearest(true, centers)
    assert d.max() < 1e-3


def test_rotation_is_orthonormal_and_fingerprinted():
    rng = np.random.default_rng(3)
    R = L.compute_rotation(rng.normal(size=(500, 128)))
    assert R.shape == (128, 96) and np.allclose(R.T @ R, np.eye(96), atol=1e-4)
    assert L.rotation_fingerprint(R) == L.rotation_fingerprint(R.copy())
    assert L.rotation_fingerprint(R) != L.rotation_fingerprint(-R)


def test_colbert_padding_rows_are_dropped_but_other_mismatches_fail():
    assert L.num_real_tokens(266_206_025, 266_205_513) == 266_205_513   # real LoTTE numbers
    assert L.num_real_tokens(1000, 1000) == 1000
    for stored in (1001, 1511, 1513, 999):
        try:
            L.num_real_tokens(stored, 1000)
        except ValueError:
            continue
        raise AssertionError(f"stored={stored} should have failed")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t(); print(f"PASS {t.__name__}")
    print(f"{len(tests)} tests passed")
