"""
Pure-python evaluation helpers (no torch, no GPU): data loading, query splits,
metrics, bootstrap confidence intervals, Pareto frontiers, operating-point
selection, latency summaries and Nsight Compute CSV parsing.
Unit tested in tests/test_eval_lib.py.
"""
import csv, hashlib, io, json
import numpy as np


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_questions(path):
    """[(qid, text)] in file order."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2:
                out.append((p[0].strip(), p[1]))
    return out


def load_qrels(path):
    """qid -> set(pid). Accepts 4-column TREC (qid 0 pid rel) or 3-column (qid pid rel)."""
    qrels = {}
    with open(path) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 4 and float(p[3]) > 0:
                qrels.setdefault(p[0].strip(), set()).add(int(p[2]))
            elif len(p) == 3 and float(p[2]) > 0:
                qrels.setdefault(p[0].strip(), set()).add(int(p[1]))
    return qrels


def make_split(qids, tune_size, seed=0):
    """
    Deterministic tune/test split by a seeded hash of each qid, so it does not
    depend on file order or on which other queries are present. Both lists keep
    the input order.
    """
    def key(q):
        return hashlib.blake2b(f"{seed}:{q}".encode(), digest_size=8).digest()
    ranked = sorted(qids, key=key)
    tune = set(ranked[:tune_size])
    return [q for q in qids if q in tune], [q for q in qids if q not in tune]


# ---------------------------------------------------------------------------
# Per-query metrics
# ---------------------------------------------------------------------------
def mrr_at_k(ranked, relevant, k=10):
    for i, pid in enumerate(ranked[:k]):
        if pid in relevant:
            return 1.0 / (i + 1)
    return 0.0


def success_at_k(ranked, relevant, k=5):
    return 1.0 if any(p in relevant for p in ranked[:k]) else 0.0


def recall_at_k(ranked, relevant, k):
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & set(relevant)) / len(relevant)


def gt_recall(ranked, gt, k):
    """Overlap of the engine's top-k with the exact top-k (gt ordered best first, -1 padded)."""
    truth = [p for p in gt[:k] if p >= 0]
    if not truth:
        return float("nan")
    return len(set(ranked[:k]) & set(truth)) / len(truth)


def official_metric(name, ranked, relevant):
    if name == "mrr@10":
        return mrr_at_k(ranked, relevant, 10)
    if name == "success@5":
        return success_at_k(ranked, relevant, 5)
    raise ValueError(name)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def bootstrap_ci(values, n_boot=2000, alpha=0.05, seed=0):
    """(mean, lo, hi) percentile bootstrap over queries; NaNs are dropped."""
    v = np.asarray(values, dtype=np.float64)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    means = v[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(v.mean()), float(lo), float(hi)


def paired_diff_ci(a, b, n_boot=2000, alpha=0.05, seed=0):
    """Mean of (a - b) over the same queries with a bootstrap CI; pairs with a NaN are dropped."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape, "paired metrics need the same queries"
    ok = ~(np.isnan(a) | np.isnan(b))
    return bootstrap_ci(a[ok] - b[ok], n_boot, alpha, seed)


def latency_summary(lats_ms):
    v = np.asarray(lats_ms, dtype=np.float64)
    return {"median": float(np.median(v)), "mean": float(v.mean()),
            "p95": float(np.percentile(v, 95)), "p99": float(np.percentile(v, 99)),
            "max": float(v.max()), "n": int(len(v))}


# ---------------------------------------------------------------------------
# Frontiers and operating points
# ---------------------------------------------------------------------------
def pareto_frontier(points, quality="gt_r10", latency="lat_median"):
    """Points sorted by latency, keeping each one that strictly improves quality."""
    front, best = [], -np.inf
    for p in sorted(points, key=lambda p: (p[latency], -p[quality])):
        if p[quality] > best:
            front.append(p)
            best = p[quality]
    return front


def select_fastest_at(points, target, quality="gt_r10", latency="lat_median"):
    """Fastest point whose quality reaches target, or None."""
    ok = [p for p in points if p[quality] >= target]
    return min(ok, key=lambda p: p[latency]) if ok else None


# ---------------------------------------------------------------------------
# Nsight Compute
# ---------------------------------------------------------------------------
def parse_ncu_csv(text, metrics):
    """
    Sum each metric over all profiled kernels in `ncu --csv --page raw --print-units base`
    output. Returns ({metric: total}, num_kernels). Rows before the header (ncu log
    lines) are skipped; the units row right after the header is skipped.
    """
    lines = text.splitlines()
    start = next((i for i, l in enumerate(lines) if l.startswith('"ID"') or l.startswith("ID,")), None)
    if start is None:
        raise ValueError("no ncu CSV header found")
    rows = list(csv.reader(io.StringIO("\n".join(lines[start:]))))
    header = rows[0]
    col = {m: header.index(m) for m in metrics}
    totals = {m: 0.0 for m in metrics}
    n = 0
    for r in rows[1:]:
        if len(r) != len(header):
            continue
        try:
            vals = {m: float(r[col[m]].replace(",", "")) for m in metrics}
        except ValueError:
            continue                        # units row or n/a
        for m in metrics:
            totals[m] += vals[m]
        n += 1
    return totals, n


def dump_json(obj, path):
    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=default)
