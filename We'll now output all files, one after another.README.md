# RTRAG: Late‑Interaction Retrieval with RT‑Accelerated Candidate Generation

RTRAG accelerates **ColBERT/PLAID‑style late‑interaction retrieval** by
offloading the *candidate generation* and *centroid‑based pruning* stages to
the **ray‑tracing (RT) cores** of modern NVIDIA GPUs. The final exact MaxSim
scoring is still performed by **TileMaxSim**, keeping the retrieval quality
identical to the reference implementation.

## What it does

1. **Offline**:
   - Train a product‑quantization codebook from document token embeddings.
   - Build an inverted list from each centroid to the documents that use it.
   - Construct a static OptiX scene where each centroid is a **sphere**
     carrying the centroid's coordinates and a pointer to its inverted list.

2. **Online** (per query):
   - Encode the query into token‑wise embeddings.
   - Cast **one ray per query token** into the centroid scene.
   - Use **any‑hit traversal** to obtain the closest centroids per token.
   - Immediately aggregate the hit centroids into approximate per‑document
     scores via the centroid‑to‑documents inverted lists.
   - Prune the candidate set using only these approximate scores.
   - Score the surviving candidates with **exact MaxSim** (TileMaxSim).

## Files

- `r_trag/rt_scene.py` — host‑side RT scene model (centroids, inverted lists).
- `r_trag/candidate_generation.py` — RT‑style candidate generation and
  approximate scoring simulation.
- `r_trag/pipeline.py` — main `RTRAG` class that coordinates all stages.
- `r_trag/__init__.py` — package exports.
- `r_trag/__main__.py` — minimal runnable example.

## Usage

