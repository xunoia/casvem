# CaSVeM v1 — Proof of Concept (Archived)

This is the first prototype of CaSVeM. It proved the concept but was redesigned in v3.

**Key architecture**: Graph + vector hybrid memory with basic in-memory caching.

**Why archived**: The graph layer added complexity without improving retrieval quality.
The cache was simple TTL-based, not learned from usage patterns.

**Active version**: See [casvem-v3](../casvem-v3/) for the production architecture.

## What v1 taught us

- Vector similarity retrieval works well for personal memory
- Graph relationships between memories added latency without clear accuracy gains
- Cache needs to be semantic (not just exact-key) to get real hit rates
- Token cost tracking is essential to prove the value proposition

## Benchmark datasets

The `benchmark/` folder contains three open datasets used across all versions:
- `beam_data/` — BEAM benchmark (kv_retrieval + longdialogue)
- `locomo_data/` — LoCoMo conversational memory dataset (10 records)
- `longmemeval_data/` — LongMemEval oracle (500 records)

These are referenced by v3 benchmarks via absolute path (not copied).
