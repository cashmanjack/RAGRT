"""
RAGRT Plugin: high-level Python retrieval interface.

    from colbert import Searcher
    from ragrt_plugin import RAGRTPlugin
    searcher = Searcher(index="...", collection="...")
    plugin = RAGRTPlugin(searcher, sparse_csr_dir="...")
    top_pids = plugin.search("is sudan iv hydrophobic or hydrophilic?", top_k=100)
"""
import os
import numpy as np
import torch

import rtrag_corr_3d
import ragrt_index_lib as L
from fast_tilemaxsim_scorer import FastTileMaxSimScorer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PTX_PATH = os.path.join(BASE_DIR, "optix_corr_3d.ptx")

DEFAULT_N_COARSE     = 32
DEFAULT_K_CANDIDATES = 4096
DEFAULT_K_EIDS       = 16
DEFAULT_TOP_K        = 100


def _load(sparse_csr_dir, name, dtype=None):
    a = np.load(os.path.join(sparse_csr_dir, name))
    return torch.from_numpy(a.astype(dtype) if dtype is not None else a).cuda()


class RAGRTPlugin:
    def __init__(self, searcher, sparse_csr_dir, ptx_path=PTX_PATH,
                 predicates_file="predicates.npy"):
        self.searcher = searcher
        self.scorer = FastTileMaxSimScorer(searcher)
        self.num_passages = len(searcher.ranker.doclens)

        self.R_proj = _load(sparse_csr_dir, "svd_rotation_128_to_96.npy").float()
        self.centroids_96 = _load(sparse_csr_dir, "centroids_96d_svd.npy").float()
        doc_predicates = _load(sparse_csr_dir, predicates_file, np.uint32)

        self.index = rtrag_corr_3d.CorrIndex3D(ptx_path)
        cb = np.load(os.path.join(sparse_csr_dir, "codebooks.npy"))
        self.index.build(torch.tensor(cb, dtype=torch.float32).contiguous(), 0.95, 4)

        n = self.num_passages
        self.index.bind_index(
            _load(sparse_csr_dir, "csr_row_ptrs.npy", np.int32),
            _load(sparse_csr_dir, "csr_col_eids.npy"),
            _load(sparse_csr_dir, "csr_block_sums_128.npy", np.int64),
            _load(sparse_csr_dir, "csr_lengths.npy"),
            _load(sparse_csr_dir, "csr_packed_24.npy"),
            self.scorer.doc_offsets[:n + 1], self.scorer.doclens[:n],
            self.scorer.codes, self.scorer.residuals,
            searcher.ranker.codec.centroids.detach().cuda().to(torch.float16).contiguous(),
            self.scorer.bucket_weights, self.scorer.reversed_bit_map, self.scorer.decomp_table,
            doc_predicates, n, self.centroids_96.shape[0],
        )

    def encode_query(self, query_text, n_coarse=DEFAULT_N_COARSE):
        """128D query rows plus the 32x3D rays and coarse centroid probes."""
        Q_full = self.searcher.encode(query_text).squeeze(0).cuda()
        ntok = L.query_ntok(self.searcher, query_text)
        Q_act = Q_full[:ntok, :].contiguous()
        Q_96 = torch.nn.functional.normalize(Q_act @ self.R_proj, p=2, dim=-1)
        Q_sub = Q_96.view(ntok, L.NUM_SUBSPACES, L.SUBSPACE_DIM).contiguous()
        scores = Q_96 @ self.centroids_96.T
        topc = scores.topk(k=n_coarse, dim=-1).indices.contiguous().to(torch.int32)
        return Q_act.float().contiguous(), Q_full.to(torch.float16).contiguous(), Q_sub, topc, scores

    @torch.no_grad()
    def search(self, query_text, top_k=DEFAULT_TOP_K, category_mask=0,
               n_coarse=DEFAULT_N_COARSE, k_candidates=DEFAULT_K_CANDIDATES, k_eids=DEFAULT_K_EIDS):
        Q_128, Q_fp16, Q_sub, topc, scores = self.encode_query(query_text, n_coarse)
        ranked = self.index.search_single_query_native(
            Q_128, Q_fp16, Q_sub, topc, scores, n_coarse, k_candidates, top_k,
            query_mask=category_mask, k_eids=k_eids)
        return ranked.cpu().tolist()
