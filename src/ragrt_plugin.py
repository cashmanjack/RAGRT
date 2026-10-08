"""
RAGRT Plugin: High-Level Python Retrieval Interface for RAGRT 3D
Wraps CorrIndex3D, SVD projection, and Fused TileMaxSim scoring into a clean API.

Usage:
    from colbert import Searcher
    from ragrt_plugin import RAGRTPlugin

    searcher = Searcher(index="...", collection="...")
    plugin = RAGRTPlugin(searcher)
    top_pids = plugin.search("is sudan iv hydrophobic or hydrophilic?", top_k=100)
"""
import os, sys, torch
import numpy as np

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
SPARSE_CSR_DIR = "/local/scratch/a/cashman3/juno_pq_lotte_full_sparse8"
PTX_PATH       = os.path.join(BASE_DIR, "optix_corr_3d.ptx")

import rtrag_corr_3d
from fast_tilemaxsim_scorer import FastTileMaxSimScorer

# Champion Parameters
DEFAULT_N_COARSE     = 32
DEFAULT_K_CANDIDATES = 4096
DEFAULT_K_EIDS       = 16
DEFAULT_PRUNE_TAU    = 0.05
DEFAULT_TOP_K        = 100


class RAGRTPlugin:
    def __init__(self, searcher, ptx_path=PTX_PATH, sparse_csr_dir=SPARSE_CSR_DIR):
        self.searcher = searcher
        self.scorer = FastTileMaxSimScorer(searcher)
        self.num_passages = len(searcher.ranker.doclens)

        # 1. Load SVD projection and 96D centroids
        self.R_proj = torch.from_numpy(
            np.load(os.path.join(sparse_csr_dir, "svd_rotation_128_to_96.npy"))
        ).cuda().float()
        self.centroids_96 = torch.from_numpy(
            np.load(os.path.join(sparse_csr_dir, "centroids_96d_svd.npy"))
        ).cuda().float()
        self.doc_predicates = torch.from_numpy(
            np.load(os.path.join(sparse_csr_dir, "doc_predicates.npy"))
        ).cuda().to(torch.uint32)

        # 2. Build 3D OptiX BVH & Bind Hierarchy
        cb = np.load(os.path.join(sparse_csr_dir, "codebooks.npy"))
        sc = np.load(os.path.join(sparse_csr_dir, "super_codewords.npy"))
        gr = np.load(os.path.join(sparse_csr_dir, "group_radius.npy"))
        ga = np.load(os.path.join(sparse_csr_dir, "group_assign.npy"))

        self.index = rtrag_corr_3d.CorrIndex3D(ptx_path)
        self.index.build(torch.tensor(cb, dtype=torch.float32).contiguous(), 0.95, 4)
        self.index.bind_hierarchy(
            torch.tensor(sc).float(), torch.tensor(gr).float(), torch.tensor(ga).int()
        )

        # 3. Bind Compressed Sparse Row (CSR) Index & TileMaxSim Buffers
        csr_row_ptrs = torch.from_numpy(np.load(os.path.join(sparse_csr_dir, "csr_row_ptrs.npy")).astype(np.int32)).cuda()
        csr_col_eids = torch.from_numpy(np.load(os.path.join(sparse_csr_dir, "csr_col_eids.npy"))).cuda()
        csr_offsets  = torch.from_numpy(np.load(os.path.join(sparse_csr_dir, "csr_offsets.npy")).astype(np.int64)).cuda()
        csr_lengths  = torch.from_numpy(np.load(os.path.join(sparse_csr_dir, "csr_lengths.npy"))).cuda()
        map_packed   = torch.from_numpy(np.load(os.path.join(sparse_csr_dir, "csr_packed_24.npy"))).cuda()
        centroids_128 = searcher.ranker.codec.centroids.detach().cuda().to(torch.float16).contiguous()

        self.index.bind_index(
            csr_row_ptrs, csr_col_eids, csr_offsets, csr_lengths, map_packed,
            self.scorer.doc_offsets[:self.num_passages + 1], self.scorer.doclens[:self.num_passages],
            self.scorer.codes, self.scorer.residuals, centroids_128,
            self.scorer.bucket_weights, self.scorer.reversed_bit_map, self.scorer.decomp_table,
            self.doc_predicates, self.num_passages, self.centroids_96.shape[0]
        )

    def encode_query(self, query_text):
        """Encodes query text into 128D FP16 and projected 32x3D subspace rays."""
        Q_full = self.searcher.encode(query_text).squeeze(0).cuda()
        ntok = min(len(query_text.split()) + 4, 32)
        Q_act = Q_full[:ntok, :].contiguous()

        # Project 128D -> 96D via SVD and reshape to 32 x 3D
        Q_96 = torch.nn.functional.normalize(Q_act @ self.R_proj, p=2, dim=-1)
        Q_sub = Q_96.view(ntok, 32, 3).contiguous()

        scores = Q_96 @ self.centroids_96.T
        topc = scores.topk(k=DEFAULT_N_COARSE, dim=-1).indices.contiguous().to(torch.int32)

        return Q_act.float().contiguous(), Q_full.to(torch.float16).contiguous(), Q_sub, topc, scores

    @torch.no_grad()
    def search(self, query_text, top_k=DEFAULT_TOP_K, category_mask=0):
        """
        Executes end-to-end RAGRT search:
        Stage 1 (OptiX Raygen/AnyHit) -> Stage 2 (CSR Gather) -> 
        Stage 3 (Cooperative Scoring) -> Stage 4 (Fused TileMaxSim).
        """
        Q_128_fp32, Q_fp16, Q_sub, topc, scores = self.encode_query(query_text)

        ranked_tensor = self.index.search_single_query_native(
            Q_128_fp32, Q_fp16, Q_sub, topc, scores,
            DEFAULT_N_COARSE, DEFAULT_K_CANDIDATES, top_k,
            0, category_mask, DEFAULT_K_EIDS, 0, DEFAULT_PRUNE_TAU
        )
        return ranked_tensor.cpu().tolist()
