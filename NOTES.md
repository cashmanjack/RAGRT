# RTRAG — Project Notes

## Goal
RT-core-accelerated candidate generation + pruning for ColBERT/PLAID-style
late interaction retrieval, with TileMaxSim as the final exact-scoring kernel.

## Reference material
- reference/colbert-plaid/  — PLAID + ColBERTv2 (Stanford)
- reference/tilemaxsim/     — IO-aware MaxSim scoring kernel
- papers/                   — distilled notes on JUNO, JUNO++, PLAID, ColBERTv2, TileMaxSim

## Current stage
- [ ] Read PLAID's centroid-interaction function signature (find it in reference/colbert-plaid)
- [ ] Prototype OptiX scene (centroid spheres) offline build
- [ ] Prototype per-query-token ray launch, t_hit-based similarity
- [ ] Wire RT kernel behind same interface as PLAID's centroid interaction
- [ ] Validate output parity vs PLAID software baseline
- [ ] Integrate TileMaxSim as final scorer

## Open questions
-
