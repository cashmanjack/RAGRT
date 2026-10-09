"""
Synthetic predicate bitmasks for selectivity sweeps (independent of relevance).
  bit 0: 5%   bit 1: 14%   bit 2: 30%   bit 3: 60%   bit 4: 100%
Usage: python3 build_synthetic_predicates.py --outdir <sparse dir> --num_passages N [--seed 42]
"""
import os, argparse
import numpy as np

SELECTIVITIES = [0.05, 0.14, 0.30, 0.60]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--num_passages", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    pred = np.zeros(args.num_passages, dtype=np.uint32)
    for bit, sel in enumerate(SELECTIVITIES):
        m = rng.random(args.num_passages) < sel
        pred[m] |= np.uint32(1 << bit)
        print(f"  bit {bit}: target {sel:.0%}, actual {m.mean():.2%}")
    pred |= np.uint32(1 << len(SELECTIVITIES))
    out = os.path.join(args.outdir, "synthetic_predicates.npy")
    np.save(out, pred)
    print(f"Saved {out} ({args.num_passages:,} passages, seed {args.seed})")


if __name__ == "__main__":
    main()
