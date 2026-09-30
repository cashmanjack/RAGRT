import torch
import rtrag_corr_3d

class FastTileMaxSimScorer:
    def __init__(self, searcher):
        self.codec = searcher.ranker.codec
        self.doclens = torch.tensor(searcher.ranker.doclens, dtype=torch.int32, device='cuda')
        self.doc_offsets = torch.cat([
            torch.zeros(1, dtype=torch.int64, device='cuda'),
            torch.cumsum(self.doclens.to(torch.int64), dim=0)
        ])
        
        # Pointers to ColBERT's index in VRAM
        self.codes = searcher.ranker.embeddings.codes.cuda().to(torch.int32)
        self.residuals = searcher.ranker.embeddings.residuals.cuda().contiguous()
        self.centroids = self.codec.centroids.cuda().to(torch.float16).contiguous()
        self.bucket_weights = self.codec.bucket_weights.cuda().to(torch.float16).contiguous()
        self.reversed_bit_map = self.codec.reversed_bit_map.cuda().to(torch.uint8).contiguous()
        self.decomp_table = self.codec.decompression_lookup_table.cuda().to(torch.uint8).contiguous()

    @torch.no_grad()
    def score(self, Q_full, candidate_pids):
        if candidate_pids is None or len(candidate_pids) == 0:
            return []

        if isinstance(candidate_pids, torch.Tensor):
            pids = candidate_pids.to(dtype=torch.int32, device='cuda').contiguous()
            cand_list = candidate_pids.cpu().tolist()
        else:
            pids = torch.tensor(candidate_pids, dtype=torch.int32, device='cuda')
            cand_list = candidate_pids

        # Contract for the fused kernel: it reads Q as [32, 128] on the device.
        # Fail here instead of silently misreading a shorter query matrix.
        Q = Q_full.to(torch.float16).contiguous()
        assert Q.dim() == 2 and Q.shape[0] == 32 and Q.shape[1] == 128, \
            f"fused MaxSim kernel needs Q of shape [32, 128], got {tuple(Q.shape)}"

        # Launch Fused Residual Decompression + TileMaxSim (Zero intermediate DRAM writes!)
        scores = rtrag_corr_3d.native_tile_maxsim_fused_decomp(
            Q,
            pids,
            self.doc_offsets,
            self.doclens,
            self.codes,
            self.residuals,
            self.centroids,
            self.bucket_weights,
            self.reversed_bit_map,
            self.decomp_table
        )

        top_k = min(100, len(cand_list))
        topk_indices = torch.topk(scores, top_k).indices.cpu().tolist()
        return [cand_list[i] for i in topk_indices]
