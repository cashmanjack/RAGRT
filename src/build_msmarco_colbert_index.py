"""
Build 2-bit ColBERTv2 Index for MS MARCO (8.8M Passages).
Output Directory: /local/scratch/a/cashman3/msmarco/indexes/msmarco.dev.2bit
"""
import os, sys, torch

BASE_DIR = "/home/min/a/cashman3/RTRAG/src"
sys.path.insert(0, os.path.join(BASE_DIR, "../reference/colbert-plaid"))

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer

COLLECTION = "/local/scratch/a/cashman3/msmarco/collection.tsv"
INDEX_ROOT = "/local/scratch/a/cashman3/msmarco/indexes"
INDEX_NAME = "msmarco.dev.2bit"

if __name__ == '__main__':
    os.makedirs(INDEX_ROOT, exist_ok=True)
    with Run().context(RunConfig(nranks=1, experiment="msmarco", root=INDEX_ROOT)):
        config = ColBERTConfig(
            nbits=2,
            kmeans_niters=20,
            ncells=2,
            centroid_score_threshold=0.45,
            ndocs=4096
        )
        indexer = Indexer(checkpoint="colbert-ir/colbertv2.0", config=config)
        indexer.index(name=INDEX_NAME, collection=COLLECTION, overwrite=True)
