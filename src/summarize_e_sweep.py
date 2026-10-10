"""
Codebook-size (E) sweep summary: one table and one Pareto figure across E.
E=256 is read from --base (the main run's results subdir), other E from E<E>/.

  python3 summarize_e_sweep.py --dataset lotte --base v2_E256 --Es 256,1024,4096,16384
Writes <results>/<dataset>/e_sweep.md and fig_e_sweep.png
"""
import os, json, argparse
import numpy as np

import eval_config as C
import eval_lib as E


def load(root, sub):
    p = os.path.join(root, sub, "results.json")
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(C.DATASETS))
    ap.add_argument("--base", required=True)
    ap.add_argument("--Es", default="256,1024,4096,16384")
    args = ap.parse_args()
    root = os.path.join(C.RESULTS_ROOT, args.dataset)
    runs = {}
    for e in [int(x) for x in args.Es.split(",")]:
        R = load(root, args.base if e == 256 else f"E{e}")
        if R is None:
            print(f"  E={e}: no results yet")
            continue
        runs[e] = R

    lines = [f"# E sweep: {args.dataset}", "",
             "Fastest TUNE config per target, measured on TEST (median ms, exact R@10).", "",
             "| target | E | RAGRT (RT) | RAGRT (no RT) | RT speedup | PLAID+TMS | vs PLAID+TMS | RAGRT config | hits/ray |",
             "|---|---|---|---|---|---|---|---|---|"]
    targets = []
    for R in runs.values():
        for t in R.get("selection", {}):
            if t not in targets:
                targets.append(t)
    for t in targets:
        for e, R in runs.items():
            s = R.get("selection", {}).get(t, {})
            cell = {}
            for eng in ("ragrt", "ragrt_bf", "ptms"):
                x = s.get(eng)
                r = R.get("test", {}).get(f"{eng}|{x['key']}") if x else None
                cell[eng] = r
            f = lambda r: f"{r['lat']['median']:.2f} ({r['gt_r10']['mean']:.3f})" if r else "unreachable"
            sp = (f"{cell['ragrt_bf']['lat']['median'] / cell['ragrt']['lat']['median']:.2f}x"
                  if cell["ragrt"] and cell["ragrt_bf"] else "n/a")
            vs = (f"{cell['ptms']['lat']['median'] / cell['ragrt']['lat']['median']:.2f}x"
                  if cell["ragrt"] and cell["ptms"] else "n/a")
            key = s["ragrt"]["key"] if s.get("ragrt") else ""
            d = R.get("drops", {}).get(f"ragrt|{key}", {})
            hpr = f"{d['hits_per_ray']:.0f}" if "hits_per_ray" in d else "n/a"
            lines.append(f"| {t} | {e} | {f(cell['ragrt'])} | {f(cell['ragrt_bf'])} | {sp} | {f(cell['ptms'])} | {vs} | {key} | {hpr} |")
    md = "\n".join(lines) + "\n"
    open(os.path.join(root, "e_sweep.md"), "w").write(md)
    print(md)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    cmap = plt.get_cmap("viridis")
    for i, (e, R) in enumerate(sorted(runs.items())):
        col = cmap(i / max(1, len(runs) - 1))
        for eng, ls in (("ragrt", "-"), ("ragrt_bf", ":")):
            pts = R.get("sweep", {}).get(eng)
            if not pts:
                continue
            fr = E.pareto_frontier(pts)
            ax.plot([p["lat_median"] for p in fr], [p["gt_r10"] for p in fr], ls, color=col, marker=".",
                    label=f"E={e} {'RT' if eng == 'ragrt' else 'no RT'}")
    base = runs.get(256) or next(iter(runs.values()))
    if base.get("sweep", {}).get("ptms"):
        fr = E.pareto_frontier(base["sweep"]["ptms"])
        ax.plot([p["lat_median"] for p in fr], [p["gt_r10"] for p in fr], "-", color="#d62728", marker="x", label="PLAID+TMS")
    ax.set_xscale("log"); ax.set_xlabel("median latency, TUNE (ms)"); ax.set_ylabel("exact recall@10")
    ax.grid(alpha=0.3); ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(root, "fig_e_sweep.png"), dpi=160)
    print(f"Saved {os.path.join(root, 'fig_e_sweep.png')}")


if __name__ == "__main__":
    main()
