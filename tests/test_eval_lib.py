"""CPU tests for eval_lib (no GPU). Run: python3 tests/test_eval_lib.py"""
import os, sys, math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import eval_lib as E


def test_split_is_deterministic_disjoint_and_order_independent():
    qids = [f"q{i}" for i in range(1000)]
    tune, test = E.make_split(qids, 100, seed=0)
    assert len(tune) == 100 and len(test) == 900 and not set(tune) & set(test)
    tune2, _ = E.make_split(list(reversed(qids)), 100, seed=0)
    assert set(tune2) == set(tune)                       # file order does not matter
    assert tune == [q for q in qids if q in set(tune)]   # input order kept
    tune3, _ = E.make_split(qids + ["extra"], 100, seed=0)
    assert len(set(tune3) & set(tune)) >= 99             # adding a query barely moves the split
    assert set(E.make_split(qids, 100, seed=1)[0]) != set(tune)


def test_metrics():
    assert E.mrr_at_k([5, 7, 9], {9}, 10) == 1 / 3
    assert E.mrr_at_k([5, 7, 9], {9}, 2) == 0.0
    assert E.success_at_k([1, 2, 3, 4, 5, 6], {6}, 5) == 0.0
    assert E.success_at_k([1, 2, 3, 4, 5, 6], {5}, 5) == 1.0
    assert E.recall_at_k([1, 2, 3], {2, 8}, 3) == 0.5
    assert E.official_metric("success@5", [1, 2], {2}) == 1.0


def test_gt_recall_ignores_padding_and_is_nan_without_truth():
    gt = [10, 11, 12, -1, -1]
    assert E.gt_recall([11, 99, 10], gt, 10) == 2 / 3
    assert E.gt_recall([1, 2, 3], gt, 2) == 0.0
    assert math.isnan(E.gt_recall([1], [-1, -1], 10))


def test_bootstrap_ci():
    m, lo, hi = E.bootstrap_ci([0.5] * 50)
    assert m == lo == hi == 0.5
    v = np.random.default_rng(0).random(2000)
    m, lo, hi = E.bootstrap_ci(v)
    assert lo < m < hi and hi - lo < 0.04
    m, lo, hi = E.bootstrap_ci([1.0, float("nan"), 0.0])
    assert m == 0.5


def test_paired_diff_ci_detects_consistent_gain():
    rng = np.random.default_rng(1)
    a = rng.random(500)
    m, lo, hi = E.paired_diff_ci(a + 0.01, a)
    assert abs(m - 0.01) < 1e-12 and lo > 0
    try:
        E.paired_diff_ci([1, 2], [1, 2, 3])
    except AssertionError:
        pass
    else:
        raise AssertionError("mismatched lengths must fail")


def test_frontier_and_selection():
    pts = [{"lat_median": 1, "gt_r10": 0.5}, {"lat_median": 2, "gt_r10": 0.4},
           {"lat_median": 3, "gt_r10": 0.9}, {"lat_median": 3, "gt_r10": 0.95},
           {"lat_median": 5, "gt_r10": 0.94}]
    front = E.pareto_frontier(pts)
    assert [p["gt_r10"] for p in front] == [0.5, 0.95]
    assert E.select_fastest_at(pts, 0.9)["gt_r10"] in (0.9, 0.95)
    assert E.select_fastest_at(pts, 0.9)["lat_median"] == 3
    assert E.select_fastest_at(pts, 0.99) is None


def test_latency_summary():
    s = E.latency_summary(list(range(1, 101)))
    assert s["median"] == 50.5 and s["n"] == 100 and s["max"] == 100


def test_parse_ncu_csv():
    text = "\n".join([
        "==PROF== Connected to process 4242",
        '"ID","Process ID","Kernel Name","dram__bytes_read.sum","dram__bytes_write.sum"',
        '"","","","byte","byte"',
        '"0","4242","k1","1,048,576","2048"',
        '"1","4242","k2","100","0"',
        "==PROF== Disconnected",
    ])
    tot, n = E.parse_ncu_csv(text, ["dram__bytes_read.sum", "dram__bytes_write.sum"])
    assert n == 2 and tot["dram__bytes_read.sum"] == 1048676 and tot["dram__bytes_write.sum"] == 2048


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"{len(tests)} tests passed")
