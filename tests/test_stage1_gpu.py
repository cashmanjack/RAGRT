"""
GPU test (run on the server after building the extension): the brute-force Stage 1
must return exactly torch's top-k positive codewords per ray, ordered by value.
  python3 tests/test_stage1_gpu.py
"""
import os, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import rtrag_corr_3d as X

S = 32


def check(E, k, R=32 * 20, max_hits=128, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    cw = torch.randn(S * E, 3, device="cuda", generator=g) * 0.1
    q = torch.randn(R, 3, device="cuda", generator=g)
    ids, vals, n = X.bruteforce_stage1(q, cw, k, max_hits)
    torch.cuda.synchronize()
    sub = torch.arange(R, device="cuda") % S                       # ray = token * 32 + subspace
    scores = torch.einsum("rd,red->re", q, cw.view(S, E, 3)[sub])  # [R, E]
    for r in range(R):
        pos = (scores[r] > 0).nonzero().squeeze(1)
        want = min(k, max_hits, pos.numel())
        assert int(n[r]) == want, (r, int(n[r]), want)
        if want == 0:
            continue
        order = torch.argsort(scores[r][pos], descending=True, stable=True)
        exp_ids = pos[order][:want]
        got = ids[r, :want].long()
        # equal values may tie-break differently than torch's sort; compare as sets of values
        assert torch.allclose(scores[r][got], scores[r][exp_ids], atol=1e-6), r
        assert torch.allclose(vals[r, :want], scores[r][got], atol=1e-6), r
        assert len(set(got.tolist())) == want, f"duplicate ids in ray {r}"
    print(f"PASS E={E} k={k}")


if __name__ == "__main__":
    for E, k in [(256, 1), (256, 16), (256, 64), (256, 128), (1024, 64), (4096, 32)]:
        check(E, k)
    print("brute-force Stage 1 matches torch")
