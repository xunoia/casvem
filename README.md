# CaSVeM — Cached Smart Vector Memory

> **AI memory that gets cheaper as it scales — not more expensive.**

Every AI memory system today charges the same cost per query, forever.  
CaSVeM learns from usage. Repeated queries cost **zero tokens** and return in **~15ms**.  
The more you use it, the cheaper it gets. That's the flywheel.

---

## The Problem with AI Memory Today

```
┌─────────────────────────────────────────────────────────────────┐
│  User asks: "What's my name?"  ← asked 10,000 times today      │
│                                                                  │
│  Every AI memory system today:                                   │
│    Query 1:      → LLM call → $0.0002                           │
│    Query 2:      → LLM call → $0.0002   (same answer!)          │
│    Query 10,000: → LLM call → $0.0002   (STILL same answer!)    │
│                                                                  │
│  Total: $2.00 to answer the same question 10,000 times          │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  With CaSVeM:                                                    │
│    Query 1:      → LLM call → $0.0002   (cache miss, pays)      │
│    Query 2:      → Cache hit → $0.00  ✓ (47ms, zero tokens)     │
│    Query 10,000: → Cache hit → $0.00  ✓ (15ms, zero tokens)     │
│                                                                  │
│  Total: $0.0002 to answer the same question 10,000 times        │
│  Saving: 99.98%                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## The Flywheel

```
  More queries
       │
       ▼
  Cache warms ──────────────────────────────────┐
       │                                         │
       ▼                                         │
  Hit rate rises                            Economic moat
       │                                    (competitors
       ▼                                    can't replicate
  Cost per query drops                       without history)
       │
       ▼
  Users query more (it's fast + cheap)
       │
       └──────────────────────────────────────► repeat
```

At **12% hit rate** (Day 1, cold cache): **12% cost savings**  
At **80% hit rate** (mature deployment): **80% cost savings**  
At **90% hit rate**: **90% cost savings** — cost approaches zero

---

## Benchmark Results

| Benchmark | Records | Accuracy | Notes |
|-----------|---------|----------|-------|
| Synthetic (personal memory) | 25 | **96%** | CaSVeM's target use case |
| BEAM kv_retrieval | 5 | **100%** | Pure fact retrieval |
| BEAM longdialogue | 3 | 0% | Chunk coverage limit (known) |
| LoCoMo (conv. memory) | 15 | 4.2% F1 | Date format mismatch (known) |
| LongMemEval | 5 | 20% | Temporal multi-hop reasoning |

### Cache Performance — The Core Metric

| Query type | Latency | LLM tokens | Cost |
|-----------|---------|------------|------|
| Cold (first query) | ~2,500ms | ~151 tokens | paid |
| L2 cached | **47ms** | **0 tokens** | **$0.00** |
| L1 cached | **<1ms** | **0 tokens** | **$0.00** |
| **Speedup** | **54× avg / 680× peak** | **100% savings on hits** | — |

### Cost at Scale (12% hit rate, Day 1)

| Queries/day | With CaSVeM | Without | Monthly saving |
|------------|-------------|---------|----------------|
| 1,000 | $0.018 | $0.020 | $0.07 |
| 10,000 | $0.177 | $0.201 | $0.72 |
| 100,000 | $1.767 | $2.008 | $7.23 |
| 1,000,000 | $17.67 | $20.08 | $72.28 |

> At 80% hit rate (mature): monthly saving on 1M queries/day = **~$484**

Full benchmark report: [casvem-v3/benchmark/result.md](casvem-v3/benchmark/result.md)

---

## Architectural Evolution: v1 → v2 → v3

### v1 — Proof of Concept *(archived)*

```
┌──────────────────────────────────────────────┐
│  v1 Architecture                             │
│                                              │
│  Query → Graph DB + Vector DB → LLM          │
│                                              │
│  ✓ First working memory system               │
│  ✗ No caching (every query hits LLM)         │
│  ✗ Graph + vector dual-write complexity      │
│  ✗ No bitmap filter → full scans             │
│  ✗ hnswlib required C++ compiler             │
└──────────────────────────────────────────────┘
```

### v2 — Improved Retrieval *(archived)*

```
┌──────────────────────────────────────────────┐
│  v2 Architecture                             │
│                                              │
│  Query → Vector DB → Multi-provider LLM      │
│                                              │
│  ✓ Multi-provider LLM (OpenAI / Gemini)      │
│  ✓ Better retrieval pipeline                 │
│  ✗ Cache was naive (exact-match only)        │
│  ✗ No semantic cache → paraphrases missed    │
│  ✗ No reranker → retrieval imprecise         │
│  ✗ No cost tracking                          │
└──────────────────────────────────────────────┘
```

### v3 — Production Architecture *(ACTIVE)*

```
┌──────────────────────────────────────────────────────────────────┐
│  v3 Architecture — 6-stage pipeline                              │
│                                                                  │
│                        ┌─────────────┐                          │
│   Query ──────────────►│  L1 Cache   │──── HIT ──► Answer ~1ms  │
│                        │  (hot LRU)  │                          │
│                        └──────┬──────┘                          │
│                               │ MISS                             │
│                               ▼                                  │
│                        ┌─────────────┐                          │
│                        │  L2 Cache   │──── HIT ──► Answer ~47ms │
│                        │  (warm LRU) │                          │
│                        └──────┬──────┘                          │
│                               │ MISS                             │
│                               ▼                                  │
│                        ┌─────────────┐                          │
│                        │   Roaring   │                          │
│                        │   Bitmap    │ ← O(1) metadata filter   │
│                        └──────┬──────┘                          │
│                               │                                  │
│                               ▼                                  │
│                        ┌─────────────┐                          │
│                        │    HNSW     │ ← ANN vector search      │
│                        │  (usearch)  │   sub-10ms, k=50         │
│                        └──────┬──────┘                          │
│                               │                                  │
│                               ▼                                  │
│                        ┌─────────────┐                          │
│                        │  Cross-enc  │ ← rerank k=50 → k=5      │
│                        │  Reranker   │   precision boost         │
│                        └──────┬──────┘                          │
│                               │                                  │
│                               ▼                                  │
│                        ┌─────────────┐                          │
│                        │   Context   │ ← token-budget aware     │
│                        │   Builder   │                          │
│                        └──────┬──────┘                          │
│                               │                                  │
│                               ▼                                  │
│                        ┌─────────────┐                          │
│                        │    LLM      │ ← Gemini 2.5 Flash       │
│                        │  (cold only)│   or Ollama (1 .env var) │
│                        └──────┬──────┘                          │
│                               │                                  │
│                               ▼                                  │
│                        ┌─────────────┐                          │
│                        │   Cache     │ ← writeback flywheel     │
│                        │  Writeback  │   warms L1+L2 for next   │
│                        └─────────────┘                          │
│                                                                  │
│  ✓ Semantic cache (cosine ≥0.92 similarity check)               │
│  ✓ Two-level LRU (L1 hot / L2 warm)                             │
│  ✓ Roaring Bitmap O(1) metadata pre-filter                      │
│  ✓ HNSW ANN search (usearch — no C++ compiler needed)           │
│  ✓ Cross-encoder reranker (precision retrieval)                  │
│  ✓ Token-budget context builder                                  │
│  ✓ Exact USD cost tracking from API metadata                     │
│  ✓ Ollama support (one .env change, zero API cost)               │
└──────────────────────────────────────────────────────────────────┘
```

---

## Time & Space Complexity

```
┌──────────────────────────────────────────────────────────────┐
│  Operation                   Complexity    Practical          │
├──────────────────────────────────────────────────────────────┤
│  Encode text (384-dim)       O(seq × d)    ~50ms/memory      │
│  HNSW insert                 O(M log n)    ~5ms/memory       │
│  ─────────────────────────────────────────────────────────── │
│  Query — L1 cache hit        O(d)          <1ms   ← fast     │
│  Query — L2 cache hit        O(d)          ~15ms  ← fast     │
│  Query — cold path           O(kd + log n) ~5,000ms + LLM    │
│  ─────────────────────────────────────────────────────────── │
│  Bitmap filter               O(bits/64)    <1ms   ← O(1)     │
│  HNSW search k=50            O(log n + kM) ~10ms             │
│  Cross-encoder rerank k=5    O(k × seq × d)~200ms            │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│  Space at 10,000 memories                                    │
├──────────────────────────────────────────────────────────────┤
│  Vectors (384-dim float32)   384 × 4B × 10K = 15MB          │
│  HNSW index                  ~M × 8B × log(n) = 20MB        │
│  SQLite (text + metadata)    ~500B avg × 10K  =  5MB        │
│  Bitmap index                ~n_bits / 8       = <1MB        │
│  LRU cache (bounded)         2,500 entries max =  5MB        │
│  ─────────────────────────────────────────────────────────── │
│  TOTAL                                          ~47MB        │
│                                                              │
│  GPT-4 context (128K tokens) = ~512KB plain text            │
│  CaSVeM stores 10K memories in 47MB, sub-10ms retrieval     │
└──────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| Language | Python 3.12 | Fastest iteration, SDK ecosystem |
| API | FastAPI + uvicorn | Async, production-ready |
| Vector search | usearch (HNSW) | Pre-built wheels, no C++ compiler |
| Encoder | all-MiniLM-L6-v2 | 384-dim, CPU-fast, no API cost |
| Reranker | ms-marco-MiniLM-L-6-v2 | Precision, free, local |
| Metadata DB | SQLite WAL | Zero-ops, concurrent writes |
| Bitmap filter | pyroaring | O(1) metadata pre-filter |
| Cache | cachetools LRUCache | Two-level L1/L2 |
| LLM | Gemini 2.5 Flash | Cost-efficient, switchable |
| Alt LLM | Ollama | One .env change → open models |
| Phase 2 | scikit-learn MLP | Learned cache predictor |
| Built with | Claude Code (Anthropic) | AI coding assistant |

---

## Quickstart (v3)

```bash
git clone https://github.com/mujahed-dev/casvem.git
cd casvem/casvem-v3
cp .env.example .env
# Add your GEMINI_API_KEY to .env

./run.sh          # starts server on :8000
./test.sh         # unit tests + live API + all benchmarks → result.md
```

### API

```bash
# Store a memory
curl -X POST http://localhost:8000/memory \
  -H "Content-Type: application/json" \
  -d '{"text": "User prefers Python over JavaScript", "memory_type": "preference"}'

# Query (first call: cold ~2500ms, same call again: cached ~47ms, $0.00)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"text": "What language does the user prefer?"}'

# Live stats — cache hit rate, token spend, cost saved
curl http://localhost:8000/stats
```

---

## Why This Matters

Today's AI memory market (Mem0, Zep, MemGPT) charges **flat cost per query forever**.  
At scale that's $60K+/month for 100K users × 10 queries/day.

CaSVeM is the only memory system where **cost decreases as usage increases**:

```
  Day 1   (cache cold):   12% savings  → every dollar saved is a dollar of moat built
  Month 1 (cache warm):   40–60% savings
  Year 1  (cache mature): 80–90% savings → near-zero marginal cost

  Competitors who launch later start at Day 1 cold cache.
  CaSVeM's advantage compounds — it cannot be replicated without the history.
```

---

## Repository Structure

```
casvem/
├── README.md              ← you are here
├── .gitignore
├── casvem-v1/             ← archived: graph+vector PoC
│   └── README.md
├── casvem-v2/             ← archived: improved retrieval
│   └── README.md
└── casvem-v3/             ← ACTIVE: production 6-stage pipeline
    ├── run.sh
    ├── test.sh
    ├── core/
    │   ├── cache/         ← L1/L2 LRU + semantic collision check
    │   ├── memory/        ← ingest pipeline, HNSW, bitmap
    │   └── retrieval/     ← reranker + context builder
    ├── pipeline/          ← ingest.py + query.py (public API)
    ├── api/               ← FastAPI routes
    └── benchmark/
        ├── result.md      ← auto-generated after test.sh
        ├── run_synthetic.py
        ├── run_beam_local.py
        ├── run_locomo_local.py
        └── run_longmemeval_local.py
```

---

*Built by **Aimann**  & **Mujahed**  — [Xunoia Technologies Private Limited](https://xunoia.com)*  
*[casvem-v3](casvem-v3/) is the active codebase.*

© 2026 Xunoia Technologies Private Limited
