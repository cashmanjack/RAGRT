"""
RAGRT Plugin: Fixed-Function Triangle RT Engine with Autotuned Parameters
"""
import os, sys, time, torch, numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, os.path.abspath("../reference/colbert-plaid"))
import rtrag_corr

LOCAL_ROOT = "/home/min/a/cashman3/RTRAG/src/experiments"
NUM_SUBSPACES = 64
NUM_ENTRIES = 256

# AUTOTUNED OPTIMAL BALANCED PARAMETERS (< 5.5ms Total Latency, High Recall & MRR)
RAGRT_RADIUS = 0.95       
RAGRT_MAX_HITS = 128     
RAGRT_TOP_K_CENTS = 48   
RAGRT_TOP_K_EIDS = 48    

class RAGRTPlugin:
    def __init__(self, searcher, domain="science", num_categories=5):
        self.searcher = searcher
        self.num_categories = num_categories
        self.centroids = searcher.ranker.codec.centroids.detach().cpu().float().cuda()
        self._tokenizer = AutoTokenizer.from_pretrained("colbert-ir/colbertv2.0")

        if domain.startswith('msmarco'):
            pq_dir = os.path.join(LOCAL_ROOT, f"msmarco/juno_pq_{domain}")
        else:
            pq_dir = os.path.join(LOCAL_ROOT, f"juno_pq_{domain}")

        codebooks_np = np.load(os.path.join(pq_dir, "codebooks.npy"))
        self.codebooks_cpu = torch.tensor(codebooks_np, dtype=torch.float32)

        map_offsets_list, map_lengths_list, packed_list = [], [], []
        current_offset = 0

        for s in range(NUM_SUBSPACES):
            off = np.load(os.path.join(pq_dir, f"offsets_s{s:02d}.npy")).astype(np.int64)
            length = np.load(os.path.join(pq_dir, f"lengths_s{s:02d}.npy")).astype(np.int32)
            packed = np.load(os.path.join(pq_dir, f"packed_s{s:02d}.npy")).astype(np.int32)

            valid = length > 0
            off[valid] += current_offset

            map_offsets_list.append(torch.from_numpy(off))
            map_lengths_list.append(torch.from_numpy(length))
            packed_list.append(torch.from_numpy(packed))
            current_offset += len(packed)

        self.map_offsets = torch.stack(map_offsets_list).cuda()
        self.map_lengths = torch.stack(map_lengths_list).cuda()
        self.flat_packed = torch.cat(packed_list).cuda()

        cb_tensor = self.codebooks_cpu.transpose(0, 1).reshape(256, 128).contiguous()
        cats = (torch.arange(256, dtype=torch.int32) % self.num_categories)

        self.idx = rtrag_corr.CorrIndex("optix_corr.ptx")
        self.idx.build(cb_tensor, cats, NUM_SUBSPACES, RAGRT_RADIUS, 1, self.num_categories)

    def encode(self, query_text):
        Q_full = self.searcher.encode(query_text).squeeze(0).float()
        enc = self._tokenizer(query_text, padding='max_length', truncation=True, max_length=32, return_tensors='pt')
        mask = enc['attention_mask'].bool().squeeze(0)
        return Q_full[mask].cuda()

    def generate(self, Q, num_candidates=500, category_mask=0x1F):
        ntok = Q.shape[0]
        num_passages = len(self.searcher.ranker.doclens)

        scores = Q @ self.centroids.T
        topc = scores.topk(k=RAGRT_TOP_K_CENTS, dim=-1).indices.contiguous().to(torch.int32)

        Q_sub = Q.view(ntok, NUM_SUBSPACES, 2).contiguous()
        vm = category_mask * torch.ones(ntok, dtype=torch.int32, device='cuda')

        passage_scores = self.idx.fused_search(
            Q_sub.cpu(),
            vm.cpu(),
            topc,
            scores,
            self.map_offsets,
            self.map_lengths,
            self.flat_packed,
            RAGRT_MAX_HITS,
            RAGRT_TOP_K_EIDS,
            RAGRT_TOP_K_CENTS,
            num_passages
        )

        valid_mask = passage_scores > 0
        if valid_mask.sum().item() == 0:
            return []

        top_k = min(num_candidates, num_passages)
        _, top_idx = torch.topk(passage_scores, top_k)
        return top_idx.cpu().tolist()
