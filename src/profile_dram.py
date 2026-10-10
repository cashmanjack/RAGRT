"""
DRAM traffic per query with Nsight Compute, calibrated (replaces measure_dram.py /
dram_profile_worker.py, which read nsys GPU-metric samples over a whole capture).

How it works:
  * The worker runs N TEST queries per engine at the primary operating point, each
    engine inside its own NVTX range (dram_plaid, dram_ptms, dram_ragrt), plus a
    calibration kernel inside dram_calib that reads 2 GiB and writes 2 GiB (both far
    larger than the 96 MB L2, so every byte must cross DRAM).
  * The driver launches ncu once per range with --nvtx-include, collects
    dram__bytes_read.sum and dram__bytes_write.sum for every kernel in that range,
    and sums them (OptiX launches are kernels too, so the RT stage is included).
  * Calibration: measured calib bytes / 8 GiB. Engine numbers are reported raw and
    divided by that ratio.
  * Two cache modes: --cache-control all flushes L2 before every kernel (cold, an
    upper bound); none keeps L2 across kernels (warm, closer to steady state, and the
    one that shows LoTTE's dense Stage-3 table fitting in the 96 MB L2).

Requires run_benchmarks.py stages select + test (for the operating point and TEST qids).

  python3 profile_dram.py --dataset msmarco [--n 50]
"""
import os, sys, json, time, shutil, argparse, subprocess

import eval_config as C
import eval_lib as E

METRICS = ["dram__bytes_read.sum", "dram__bytes_write.sum"]
CALIB_FLOATS = 1 << 29              # 2 GiB of float32
CALIB_BYTES = 2 * 4 * CALIB_FLOATS   # 2 GiB read + 2 GiB write
RANGES = ["calib", "plaid", "ptms", "ragrt"]


def worker(args):
    import numpy as np
    import torch

    if args.range == "calib":                        # no index needed, keeps GPU memory free
        src = torch.ones(CALIB_FLOATS, dtype=torch.float32, device="cuda")
        dst = torch.empty_like(src)
        torch.mul(src, 1.0, out=dst)                 # warmup
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push("dram_calib")
        torch.mul(src, 1.0, out=dst)                 # one elementwise kernel: 2 GiB in, 2 GiB out
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
        return

    from engines import Engines
    ds = C.dataset(args.dataset)
    R = json.load(open(os.path.join(ds["results_dir"], "results.json")))
    sel = R["selection"][R["primary_target"]]
    perq = np.load(os.path.join(ds["results_dir"], "test_perquery.npz"))
    qids = [str(q) for q in perq["test_qids"][:args.n]]
    text = dict(E.load_questions(ds["questions_path"]))
    eng = Engines(ds, load_ragrt=(args.range == "ragrt"))
    bank = [eng.encode(text[q]) for q in qids]
    p = sel[args.range]["params"]

    def run():
        for Qf, n in bank:
            eng.search(args.range, Qf, n, p)

    run()                                            # warmup, outside the range
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push(f"dram_{args.range}")
    run()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()


def driver(args):
    ncu = args.ncu or shutil.which("ncu") or "/usr/local/cuda/bin/ncu"
    ds = C.dataset(args.dataset)
    R = json.load(open(os.path.join(ds["results_dir"], "results.json")))
    sel = R["selection"][R["primary_target"]]
    out = {"n_queries": args.n, "operating_point": {e: sel[e]["key"] for e in ("plaid", "ptms", "ragrt") if sel.get(e)}}
    for mode in ("none", "all"):
        res = {}
        for rng in RANGES:
            if rng != "calib" and not sel.get(rng):
                continue
            cmd = [ncu, "--target-processes", "all", "--nvtx", "--nvtx-include", f"dram_{rng}/",
                   "--metrics", ",".join(METRICS), "--csv", "--page", "raw", "--print-units", "base",
                   "--cache-control", mode,
                   sys.executable, os.path.abspath(__file__), "--worker", "--range", rng,
                   "--dataset", args.dataset, "--n", str(args.n)]
            log = os.path.join(ds["results_dir"], f"ncu_{mode}_{rng}.log")
            print(f"  ncu [{mode}] {rng}  (log: {log})", flush=True)
            t0 = time.time()
            with open(log, "w") as fh:
                proc = subprocess.Popen(cmd, cwd=C.BASE_DIR, stdout=fh, stderr=subprocess.STDOUT, text=True)
                while proc.poll() is None:
                    time.sleep(30)
                    n = sum(1 for l in open(log, errors="ignore") if l.startswith("==PROF== Profiling"))
                    print(f"    {time.time() - t0:5.0f} s, {n} kernels profiled", flush=True)
            text = open(log, errors="ignore").read()
            if proc.returncode != 0:
                sys.exit(f"FATAL: ncu failed for {rng} ({mode}).\n{text[-2000:]}\n"
                         "If this mentions ERR_NVGPUCTRPERM, GPU counters are restricted to admins on this machine.")
            totals, nk = E.parse_ncu_csv(text, METRICS)
            if nk == 0:
                sys.exit(f"FATAL: ncu profiled no kernels in dram_{rng}; check the NVTX range name.")
            res[rng] = {"read_bytes": totals[METRICS[0]], "write_bytes": totals[METRICS[1]], "kernels": nk}
        ratio = (res["calib"]["read_bytes"] + res["calib"]["write_bytes"]) / CALIB_BYTES
        mode_out = {"calibration_ratio": ratio, "raw": res}
        for rng in RANGES[1:]:
            if rng in res:
                tot = res[rng]["read_bytes"] + res[rng]["write_bytes"]
                mode_out[rng] = {"mb_per_query_raw": tot / args.n / 1e6,
                                 "mb_per_query_calibrated": tot / args.n / 1e6 / ratio,
                                 "kernels_per_query": res[rng]["kernels"] / args.n}
        out[f"cache_{mode}"] = mode_out
        print(f"  [{mode}] calibration ratio {ratio:.3f}  " + "  ".join(
            f"{r}: {mode_out[r]['mb_per_query_calibrated']:.1f} MB/q" for r in RANGES[1:] if r in mode_out), flush=True)
    path = os.path.join(ds["results_dir"], "dram.json")
    E.dump_json(out, path)
    print(f"Saved {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(C.DATASETS))
    ap.add_argument("--n", type=int, default=10, help="TEST queries per engine (ncu serializes every kernel: keep small)")
    ap.add_argument("--ncu", default=None)
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--range", choices=RANGES)
    a = ap.parse_args()
    worker(a) if a.worker else driver(a)
