# CaSVeM — Cache-Augmented Semantic Vector Memory

> **AI memory that gets cheaper as it scales — not more expensive.**

Every other AI memory system charges the same cost per query regardless of repetition.  
CaSVeM builds a learned cache from usage. Repeated queries cost **zero tokens** served in ~15ms.

---

## The Core Insight

```
Without CaSVeM:  10,000 queries × $0.02/query = $200/day (flat, forever)
With CaSVeM:     10,000 queries × $0.02 × (1 - hit_rate) → approaches $0 as cache warms
```

The flywheel: more queries → warmer cache → lower cost → economic moat at scale.

---

## Repository Structure

This repo contains all three versions of CaSVeM, showing the architectural evolution:

```
casvem/
├── casvem-v1/     Proof of concept — graph + vector hybrid, basic caching
├── casvem-v2/     Improved retrieval, multi-provider LLM abstraction  
└── casvem-v3/     Production architecture — 5-layer cache hierarchy (ACTIVE)
```

**v3 is the main version.** v1 and v2 are kept to show the research progression.

---

## v3 Architecture (Active Development)

```
Query
  → LRU Cache Gate (L1 hot / L2 warm)    ← zero-cost hit path, ~15ms
      → Roaring Bitmap pre-filter         ← O(1) metadata filter
          → HNSW vector search            ← sub-10ms ANN
              → Cross-encoder Reranker    ← precision re-ranking
                  → Context Builder       ← token-budget aware
                      → LLM (Gemini/Ollama) ← cold path only
                          → Cache Writeback flywheel
```

### Three Optimizations (all in Phase 1)

| # | Optimization | Impact |
|---|-------------|--------|
| 1 | Cosine similarity collision check on cache hits | Eliminates false positives |
| 2 | `asyncio.gather()` + Semaphore for batch LLM judging | ~12× faster benchmarking |
| 3 | Exact token counts from API metadata | Real USD cost tracking |

---

## Quickstart (v3)

```bash
cd casvem-v3
cp .env.example .env
# Add your GEMINI_API_KEY to .env
./run.sh          # start server on :8000
./test.sh         # run unit tests + live API tests + dataset benchmarks
```

### API

```bash
# Add a memory
curl -X POST http://localhost:8000/memory \
  -H "Content-Type: application/json" \
  -d '{"text": "User prefers Python over JavaScript", "memory_type": "preference"}'

# Query memories
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"text": "What language does the user prefer?"}'

# Stats (cache hit rate, token cost)
curl http://localhost:8000/stats
```

---

## Benchmark Results

See [casvem-v3/benchmark/result.md](casvem-v3/benchmark/result.md) for full results.

**Key number — cache performance:**

| Query type | Latency | LLM tokens | Cost |
|-----------|---------|------------|------|
| Cold (first time) | ~5,000ms | ~90 tokens | ~$0.000009 |
| Cached (L2 hit) | **16ms** | **0 tokens** | **$0.00** |
| Speedup | **680×** | **100% savings** | — |

---

## Tech Stack

| Layer | Tech |
|-------|------|
| Language | Python 3.12 |
| API | FastAPI + uvicorn |
| Vector search | usearch (HNSW, cosine) |
| Encoder | all-MiniLM-L6-v2 — 384-dim, CPU-only |
| Reranker | ms-marco-MiniLM-L-6-v2 — local |
| Metadata DB | SQLite WAL mode |
| Bitmap filter | pyroaring Roaring Bitmap |
| Cache | cachetools LRUCache (L1/L2) |
| LLM | Gemini 2.5 Flash (switchable to Ollama) |
| Phase 2 | scikit-learn MLP cache predictor |

---

## Versioning

| Version | Status | Key feature |
|---------|--------|-------------|
| v1 | Archived | Graph + vector hybrid, proof of concept |
| v2 | Archived | Multi-provider LLM, improved retrieval |
| v3 | **Active** | 5-layer cache hierarchy, learned cache flywheel |

---

## Why This Matters for Production AI

As AI assistants proliferate, every app needs to remember things about users. Today:

- Mem0 charges $0.02/query flat → $60K/month at 100K users × 10 queries/day
- No system learns from usage patterns
- Memory costs scale linearly with usage

CaSVeM breaks this: cost scales sublinearly because the cache absorbs repeated patterns.
The more users, the more queries share patterns, the lower the per-query cost.

---

---

*Built by Mujahed & Aimann — [Xunoia Technologies Private Limited](https://xunoia.com) — [casvem-v3](casvem-v3/) is the active codebase.*

© 2026 Xunoia Technologies Private Limited
