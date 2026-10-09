"""
Single source of truth for dataset paths and evaluation settings.
Everything that runs an experiment imports this; nothing else hard-codes paths.
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COLBERT_DIR = os.path.join(BASE_DIR, "../reference/colbert-plaid")
PTX_PATH = os.path.join(BASE_DIR, "optix_corr_3d.ptx")
RESULTS_ROOT = os.environ.get("RAGRT_RESULTS", "/local/scratch/a/cashman3/ragrt_results")

TOP_K = 100                 # every engine returns its top-100
GT_DEPTH = 100              # exact MaxSim ground truth depth
TUNE_SIZE = 1000            # queries used to sweep configs and pick operating points
SPLIT_SEED = 0

SYNTHETIC_MASKS = [("5%", 1 << 0), ("14%", 1 << 1), ("30%", 1 << 2), ("60%", 1 << 3)]
LOTTE_DOMAINS = ["writing", "recreation", "science", "technology", "lifestyle"]
DOMAIN_BIT_BASE = 8         # must match build_predicates.py

DATASETS = {
    "lotte": {
        "index": "/home/min/a/cashman3/RTRAG/src/experiments/unified_lotte/indexes/unified.dev.2bit",
        "collection": "/local/scratch/a/cashman3/lotte/unified/collection.tsv",
        "sparse_csr_dir": "/local/scratch/a/cashman3/juno_pq_lotte_full_sparse8",
        "eval_dir": "/local/scratch/a/cashman3/lotte/eval_pooled_dev",   # prepare_lotte_eval.py
        "questions": "questions.tsv",
        "qrels": "qrels.tsv",
        "official_metric": "success@5",
        "has_domains": True,
    },
    "msmarco": {
        "index": "/local/scratch/a/cashman3/msmarco/indexes/msmarco.dev.2bit",
        "collection": "/local/scratch/a/cashman3/msmarco/collection.tsv",
        "sparse_csr_dir": "/local/scratch/a/cashman3/msmarco/msmarco_sparse8",
        "eval_dir": "/local/scratch/a/cashman3/msmarco",
        "questions": "queries.dev.small.tsv",
        "qrels": "qrels.dev.small.tsv",
        "official_metric": "mrr@10",
        "has_domains": False,
    },
}


def dataset(name):
    cfg = dict(DATASETS[name])
    cfg["name"] = name
    cfg["questions_path"] = os.path.join(cfg["eval_dir"], cfg["questions"])
    cfg["qrels_path"] = os.path.join(cfg["eval_dir"], cfg["qrels"])
    cfg["queries_meta_path"] = os.path.join(cfg["eval_dir"], "queries.json") if cfg["has_domains"] else None
    cfg["predicates_path"] = os.path.join(cfg["sparse_csr_dir"], "predicates.npy")
    cfg["results_dir"] = os.path.join(RESULTS_ROOT, name)
    cfg["gt_path"] = os.path.join(RESULTS_ROOT, name, "ground_truth.npz")
    return cfg
