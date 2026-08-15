I am trying to implement retrieval augmented generation (RAG) which takes advantage of ray tracing cores for multiple stages. RAG first performs centroid interaction which finds the nearest centroid among a centroid set, then approximate MaxSim is performed to prune candidate documents, finally the remaining candidate set is scored with higher precision. The first two stages, centroid interaction and centroid pruning, are prime candidates for RT core acceleration. I also want to explore the use of multiple channels on the accelerator including hit angle, color, shape, opacity, time, and more as a part of the compressed token vector.


# RTRAG — Design Doc

## Goal
Accelerate ColBERT/PLAID-style late-interaction RAG retrieval using RT cores,
with TileMaxSim as the final exact-scoring kernel. Target contribution:
RT-native candidate generation + pruning for late interaction, which does
not exist in prior work (JUNO/JUNO++ are single-vector only).

## What operation we're actually accelerating
NOT plain top-k ANN. Late interaction scoring is a two-level reduction:
for each query token, MAX similarity over all of a document's tokens
(MaxSim), then SUM those per-token maxes across all query tokens ->
one relevance score per candidate document.

## Pipeline stages and RT-core placement
1. Encode query -> ~32 token embeddings.                    [not RT]
2. Centroid interaction: per query token, find nearby         [RT — primary
   centroids in the corpus-wide centroid set.                 target]
3. Inverted-index lookup: centroid -> candidate documents.   [attached to
                                                                RT hit data]
4. Centroid-based pruning: approximate per-document score     [RT-fusable
   using only centroid-level similarity (no residual           with step 2]
   decompression yet). Multiple rounds; each round strictly
   depends on the previous round's surviving candidate set
   (no fusing ACROSS rounds, only within one round).
5. Final residual decompression + exact MaxSim scoring on     [TileMaxSim —
   the small surviving candidate set.                          not RT]
6. Rank, return top-k.

## RT scene / query design
- Scene (built offline, once): one sphere per centroid, positioned at
  centroid coordinates. Each sphere carries (via SBT record / instance
  data) a pointer into the centroid->documents inverted list.
- Rays (per query): one ray per query token (~32 rays/query), launched
  against the shared centroid scene. This is the amortization case that
  makes RT cores worth it here — one static BVH, many rays per query,
  reused across all queries (vs. JUNO's one-ray-per-query regime).
- Distance/similarity via t_hit trick (see papers/juno.md for exact
  derivation) — avoids fetching centroid coordinates from global memory.
  MIPS variant (JUNO++ radius adjustment) needed since ColBERT similarity
  is inner-product based, not L2.
- MaxSim's "max" is NOT free from closest-hit alone (closest-hit gives
  corpus-wide nearest, not per-document nearest) -> need any-hit traversal
  + software grouping-by-document-ID + atomic max accumulation per
  (document, query-token) pair, streamed as hits arrive.

## Compression: reuse, don't design
Use ColBERTv2's own residual compression as-is (centroid ID + quantized
residual, 1-2 bits/dim). Holding this fixed makes RT-vs-software the only
variable that moves — same approach JUNO took keeping PQ itself unchanged
and only replacing the search mechanism.

## Stretch goal: additional hardware channels (Fouad's suggestion)
Beyond xyz position, OptiX exposes several channels readable ~free at hit
time: hit angle/barycentrics, opacity/facing bits, primitive type (shape),
per-instance custom data ("color"), temporal BVH (time). Idea: spread
residual vector components across these instead of storing them as an
opaque blob decompressed entirely in software. Does NOT inherently improve
accuracy — it shifts the throughput/accuracy Pareto frontier by giving
free storage capacity, which can be reinvested as either speed or finer
quantization. Treat as a stretch goal on top of the core pipeline, not
step one.

## Baseline / comparison plan
- Correctness: RT backend output must match software (PLAID) backend
  output (or be within a defined approximation bound) before any
  performance claim — same validation discipline TileMaxSim used
  (identical rankings/nDCG vs reference).
- Fair baseline = PLAID's own pipeline WITH TileMaxSim as its scoring
  kernel (current best software pipeline), not original 2022 PLAID with
  its slower original scoring kernel.
- Profile PLAID's own stage latency breakdown FIRST (JUNO-Sec-3-style
  profiling) before assuming candidate generation is the bottleneck worth
  RT-accelerating — Amdahl's law risk if TileMaxSim already dominates the
  cost picture.
- Datasets: MS MARCO v1 (in-domain), Wikipedia Open QA (in-domain), LoTTE
  and MS MARCO v2 (out-of-domain / scale), matching PLAID's own eval suite.
- Metrics: latency (mean/tail) at fixed recall, throughput-vs-recall
  Pareto curve (not single numbers), per-stage latency breakdown, index
  memory footprint.
- Hardware: RTX A6000 (48GB) for dev + full-scale eval, no memory
  constraint expected even at MS MARCO v2 scale (~27GB compressed index).

## Known risks / open questions
- Approximation error compounding: hit-count-style shortcuts (JUNO Sec
  5.4) are validated only for single-vector accumulation; unclear how
  error compounds across dozens of per-token maxes+sums in MaxSim.
- Uncertain whether end-to-end speedup will be large — PLAID+TileMaxSim
  is an already-optimized target, harder to beat than JUNO's FAISS
  baseline. Paper is still valuable via the systems/artifact contribution
  and the channel-encoding investigation even without a blowout number.
- Voronoi project (separate, on the side) may be relevant later for
  IVF-stage (not PQ-stage) cluster assignment specifically — not
  integrated into RTRAG's current scope.

## Reference material
- papers/juno.md, papers/junopp.md — RT-core ANN mechanism, t_hit trick
- papers/plaid.md, papers/colbertv2.md — late interaction + PLAID pipeline
- papers/tilemaxsim.md — final scoring kernel, drop-in interface pattern
- reference/colbert-plaid/ — fork target for candidate-gen/pruning stage
- reference/tilemaxsim/ — final scoring kernel to integrate
