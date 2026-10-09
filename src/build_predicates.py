"""
Per-passage predicate bitmask (uint32) used by the in-kernel filter.

  bits 0-4  synthetic selectivity, independent of relevance (same RNG as before, seed 42):
            bit 0: 5%   bit 1: 14%   bit 2: 30%   bit 3: 60%   bit 4: 100%
  bits 8-12 LoTTE domains, from the offsets found by prepare_lotte_eval.py (domains.json):
            8 writing, 9 recreation, 10 science, 11 technology, 12 lifestyle

Writes <outdir>/predicates.npy.

Usage:
  python3 build_predicates.py --outdir <sparse dir> --num_passages N [--domains_json domains.json]
"""
import os, json, argparse
import numpy as np

SELECTIVITIES = [0.05, 0.14, 0.30, 0.60]          # bit 4 = everything
DOMAIN_BIT_BASE = 8
DOMAINS = ["writing", "recreation", "science", "technology", "lifestyle"]


def synthetic_bits(num_passages, seed=42):
    rng = np.random.default_rng(seed)
    pred = np.zeros(num_passages, dtype=np.uint32)
    for bit, sel in enumerate(SELECTIVITIES):
        pred[rng.random(num_passages) < sel] |= np.uint32(1 << bit)
    pred |= np.uint32(1 << len(SELECTIVITIES))
    return pred


def domain_bits(num_passages, domains):
    pred = np.zeros(num_passages, dtype=np.uint32)
    covered = np.zeros(num_passages, dtype=bool)
    for i, d in enumerate(DOMAINS):
        o, n = domains[d]["offset"], domains[d]["count"]
        assert not covered[o:o + n].any(), f"{d} overlaps another domain"
        pred[o:o + n] |= np.uint32(1 << (DOMAIN_BIT_BASE + i))
        covered[o:o + n] = True
    assert covered.all(), "domains do not cover every passage"
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--num_passages", type=int, required=True)
    ap.add_argument("--domains_json", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    pred = synthetic_bits(args.num_passages, args.seed)
    for bit, sel in enumerate(SELECTIVITIES):
        print(f"  synthetic bit {bit}: target {sel:.0%}, actual {((pred >> bit) & 1).mean():.2%}")
    if args.domains_json:
        dj = json.load(open(args.domains_json))
        assert dj["num_passages"] == args.num_passages, "domains.json is for a different collection"
        pred |= domain_bits(args.num_passages, dj["domains"])
        for i, d in enumerate(DOMAINS):
            print(f"  domain bit {DOMAIN_BIT_BASE + i} ({d}): {((pred >> (DOMAIN_BIT_BASE + i)) & 1).mean():.2%}")
    out = os.path.join(args.outdir, "predicates.npy")
    np.save(out, pred)
    print(f"Saved {out} ({args.num_passages:,} passages)")


if __name__ == "__main__":
    main()
