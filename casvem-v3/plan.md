# CaSVeM v3 — Implementation Plan

> **USP**: "Every AI memory system gets more expensive as it scales. CaSVeM gets cheaper."
> The cache warms with every query. More usage = lower cost per query. Cost curve inversion.

---

## What This Is (Not v1/v2)

v1 and v2 explored a 5-layer hierarchy (L1-L5) for memory compression and retrieval.

v3 is a different thesis entirely:

```
v1/v2 thesis:  Better memory STRUCTURE → better accuracy
v3 thesis:     Better memory CACHING   → lower cost + better latency → better product
```

v3's core claim is that AI memory systems are wasteful because they re-search storage on every query. 
v3 inserts a Learned Cache Gate between the query and the retrieval pipeline. After warm-up, 
80-85% of queries are served in under 1ms with zero tokens spent. The system learns what gets 
asked repeatedly and promotes those answers to the hottest cache tier.

This is the CDN insight applied to AI memory.

---

## Architecture Overview

```
QUERY
  │
  ▼
┌─────────────────────────────────────────────────────┐
│  LEARNED CACHE GATE                                 │
│  Phase 1, Week 1-2: LRU eviction                    │
│  Phase 1, Week 3-4: MLP predictor replaces LRU      │
│                                                     │
│  L1 (hot):  access_count > 10  → ~0.1ms            │
│  L2 (warm): access_count > 3   → ~0.5ms            │
│                                                     │
│  Cache HIT  → return instantly, zero tokens         │
│  Cache MISS → fall through to retrieval pipeline    │
└──────────────┬──────────────────────────────────────┘
               │ MISS only
               ▼
┌─────────────────────────────────────────────────────┐
│  ROARING BITMAP FILTER                              │
│  Pre-filter candidates by metadata fields:          │
│  date_range, memory_type, project_id, author_id     │
│  ~30 lines of code, eliminates 80-95% of corpus     │
└──────────────┬──────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────┐
│  HNSW DENSE SEARCH                                  │
│  Euclidean-encoded query vector                     │
│  Searches ONLY pre-filtered candidates              │
│  Returns top-K (K=50 default)                       │
└──────────────┬──────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────┐
│  RERANKER                                           │
│  Cross-encoder rescores top-K                       │
│  Early exit: if top result confidence > 0.95 →      │
│  skip remaining candidates                          │
│  Returns top-N (N=5 default)                        │
└──────────────┬──────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────┐
│  CONTEXT BUILDER                                    │
│  Token budget: 2048 tokens max                      │
│  Sort by: relevance score * recency weight          │
│  Assemble: memory blocks for LLM                    │
└──────────────┬──────────────────────────────────────┘
               │
               ▼
           LLM ANSWER
               │
               ▼
┌─────────────────────────────────────────────────────┐
│  MEMORY UPDATE (flywheel)                           │
│  Write result to cache (cache write-back)           │
│  Increment access_count for matched memories        │
│  Check access_count thresholds → promote L2 → L1   │
│  Apply confidence decay to old entries              │
│  Queue MLP retraining if query_log > 1000 entries  │
└─────────────────────────────────────────────────────┘
```

---

## Tech Stack — Full List

### Core Libraries

| Component | Library | Version | Notes |
|-----------|---------|---------|-------|
| Encoder | `sentence-transformers` | ≥3.0 | `all-MiniLM-L6-v2` — 384-dim, fully local, CPU-fast |
| Vector Index | `hnswlib` | ≥0.8 | CPU-native HNSW, no GPU needed |
| Bitmap Filter | `pyroaring` | ≥0.4 | Roaring Bitmap Python bindings |
| Cache Predictor | `scikit-learn` | ≥1.4 | `MLPClassifier` for learned cache predictor |
| Reranker | `sentence-transformers` | ≥3.0 | `cross-encoder/ms-marco-MiniLM-L-6-v2` — local |
| Storage | `sqlite3` | stdlib | metadata, access logs, cache state |
| LLM (default) | `google-generativeai` | ≥0.8 | Gemini 2.0 Flash — answer generation + judging |
| LLM (future) | `ollama` | ≥0.3 | open model backend, pluggable via provider pattern |
| LLM abstraction | `core/llm/` | internal | provider interface — swap backends without touching pipeline |
| API | `fastapi` | ≥0.111 | HTTP API |
| API server | `uvicorn` | ≥0.30 | ASGI server |
| Dashboard | `rich` | ≥13.0 | live terminal dashboard |
| Numpy | `numpy` | ≥1.26 | vector operations |

### Models

| Model | Where | Use | Cost / notes |
|-------|-------|-----|--------------|
| `gemini-2.0-flash` | Google API | LLM answer generation + benchmark judge | ~$0.0001/1k input tokens |
| `all-MiniLM-L6-v2` | local (sentence-transformers) | encoding — never hits an API | free, ~384-dim |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | local (sentence-transformers) | reranking | free, fast on CPU |

> Encoder and reranker stay local permanently — no API cost, no latency added.
> LLM backend is swappable: Gemini now, any open model later via `LLM_BACKEND` env var.

### Benchmark Libraries (benchmark/ only)

| Library | Use |
|---------|-----|
| `datasets` | HuggingFace dataset downloader |
| `huggingface_hub` | auth for gated datasets |
| `tabulate` | results table printing |
| `python-dotenv` | .env loading |

---

## Environment Setup

### System Requirements

```
OS:     Linux (Ubuntu 22.04+ recommended)
Python: 3.10+
RAM:    15GB (fits comfortably)
Disk:   ~5GB (models + indexes + SQLite)
GPU:    NOT required
```

### Gemini API Key Setup

```bash
# Get key: https://aistudio.google.com/apikey (free tier available)
# Add to .env:
GEMINI_API_KEY=your_key_here
```

```bash
# Verify it works before building anything:
python -c "
import google.generativeai as genai, os
genai.configure(api_key=os.environ['GEMINI_API_KEY'])
m = genai.GenerativeModel('gemini-2.0-flash')
print(m.generate_content('say hi').text)
"
```

### Python Environment

```bash
cd casvem-v3/
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Future: Adding an Open Model Backend

When you want to add an open model (Ollama, vLLM, LMStudio, etc.):
1. Install Ollama: `curl -fsSL https://ollama.com/install.sh | sh`
2. Pull model: `ollama pull qwen3:4b`
3. In `.env`: set `LLM_BACKEND=ollama` and `LLM_MODEL=qwen3:4b`
4. No code changes needed — `core/llm/factory.py` handles the swap

The provider abstraction is designed specifically so this is a config change, not a code change.

### .env File

```env
# casvem-v3/.env

# ── LLM Backend ──────────────────────────────────────
LLM_BACKEND=gemini                          # gemini | ollama | openai
LLM_MODEL=gemini-2.0-flash                  # model name for chosen backend
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=1024

# Gemini (active)
GEMINI_API_KEY=your_key_here

# Ollama (future open models — uncomment when switching)
# OLLAMA_BASE_URL=http://localhost:11434

# OpenAI (future — uncomment when switching)
# OPENAI_API_KEY=your_key_here

# ── Benchmark Judge ───────────────────────────────────
JUDGE_BACKEND=gemini                        # uses same backend as LLM by default
JUDGE_MODEL=gemini-2.0-flash

# ── Encoder + Reranker (always local) ────────────────
ENCODER_MODEL=all-MiniLM-L6-v2
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# ── HNSW ─────────────────────────────────────────────
HNSW_M=16
HNSW_EF_CONSTRUCTION=200
HNSW_EF_SEARCH=50
TOP_K=50
TOP_N=5

# ── Cache ─────────────────────────────────────────────
CACHE_L1_THRESHOLD=10
CACHE_L2_THRESHOLD=3
CACHE_CONFIDENCE_DECAY=0.01
MLP_RETRAIN_AFTER=1000

# ── Context Builder ───────────────────────────────────
CONTEXT_TOKEN_BUDGET=2048
RERANKER_EARLY_EXIT_THRESHOLD=0.95

# ── Storage ───────────────────────────────────────────
SQLITE_PATH=data/casvem.db
HNSW_INDEX_PATH=data/hnsw_index.bin

# ── Dashboard ─────────────────────────────────────────
COST_DASHBOARD=true
MEM0_COST_PER_QUERY=0.02              # baseline cost to compare against
GEMINI_COST_PER_1K_INPUT=0.0001       # for real $ savings calculation
GEMINI_COST_PER_1K_OUTPUT=0.0004
```

---

## Folder Structure

```
casvem-v3/
│
├── plan.md                      ← this file
├── requirements.txt
├── .env.example
├── config.py                    ← loads .env, single source of truth for all settings
├── main.py                      ← FastAPI app entry point
│
├── data/                        ← runtime data (gitignored)
│   ├── casvem.db                ← SQLite (memories, access log, cache state)
│   ├── hnsw_index.bin           ← HNSW index (persisted to disk)
│   └── mlp_model.pkl            ← trained MLP predictor (Phase 1 Week 3+)
│
├── core/
│   ├── __init__.py
│   │
│   ├── llm/                     ← LLM Provider Abstraction (swap backends via config)
│   │   ├── __init__.py
│   │   ├── base.py              ← BaseLLMProvider (abstract interface)
│   │   │                           complete(prompt, max_tokens) -> str
│   │   │                           judge(question, ground_truth, answer) -> bool
│   │   ├── gemini.py            ← GeminiProvider  (default, Phase 1)
│   │   │                           uses google-generativeai SDK
│   │   ├── ollama.py            ← OllamaProvider  (future open models)
│   │   │                           uses ollama SDK, any model via LLM_MODEL
│   │   └── factory.py           ← get_llm_provider() -> BaseLLMProvider
│   │                               reads LLM_BACKEND from env, returns correct provider
│   │                               adding a new backend = new file + one line in factory
│   │
│   ├── encoder.py               ← Euclidean Encoder
│   │                               encodes text → 384-dim float32 vector
│   │                               wraps sentence-transformers all-MiniLM-L6-v2
│   │
│   ├── storage.py               ← Storage layer
│   │                               SQLite tables: memories, access_log, cache_entries
│   │                               HNSW index: add/search/save/load
│   │
│   ├── cache/
│   │   ├── __init__.py
│   │   ├── lru_cache.py         ← Phase 1 Week 1-2
│   │   │                           LRU eviction, exact + cosine key matching
│   │   ├── mlp_predictor.py     ← Phase 1 Week 3-4
│   │   │                           MLPClassifier: features(query_vec) → P(cache_hit)
│   │   │                           trains on access_log, retrains every 1000 queries
│   │   ├── cache_gate.py        ← Unified interface (swaps LRU for MLP at week 3)
│   │   │                           check(query_vec) → (hit: bool, result: CacheEntry | None)
│   │   │                           write(query_vec, result, memory_ids)
│   │   └── access_log.py        ← Access tracking
│   │                               log_access(memory_id), get_count(memory_id)
│   │                               promote L2→L1 when count > threshold
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── bitmap_filter.py     ← Roaring Bitmap pre-filter
│   │   │                           build_index(field, values)
│   │   │                           filter(date_range, memory_type, project_id) → candidate_ids
│   │   ├── hnsw_search.py       ← HNSW search wrapper
│   │   │                           search(query_vec, candidate_ids, top_k) → [(id, score)]
│   │   └── reranker.py          ← Cross-encoder reranker
│   │                               rerank(query_text, candidates, top_n) → ranked_candidates
│   │                               early_exit if top confidence > threshold
│   │
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── writer.py            ← Memory ingest
│   │   │                           add(text, metadata) → memory_id
│   │   │                           encodes → stores in SQLite → adds to HNSW
│   │   │                           updates bitmap indexes
│   │   └── updater.py           ← Memory Update (flywheel)
│   │                               cache_writeback(query_vec, result, memory_ids)
│   │                               invalidate(memory_id)  ← call on memory edit/delete
│   │                               decay_old_entries()    ← run on schedule
│   │                               queue_mlp_retrain()    ← when log hits threshold
│   │
│   └── context/
│       ├── __init__.py
│       └── builder.py           ← Context Builder
│                                   build(memories, token_budget) → context_str
│                                   sorts by: relevance_score * recency_weight
│                                   truncates to token budget
│
├── pipeline/
│   ├── __init__.py
│   ├── query.py                 ← Full query pipeline
│   │                               query(text, filters) → (answer_context, metadata)
│   │                               wires: cache_gate → bitmap → hnsw → reranker → context
│   └── ingest.py                ← Full ingest pipeline
│                                   ingest(text, metadata) → memory_id
│                                   wires: encoder → writer → bitmap_update → cache_update
│
├── dashboard/
│   ├── __init__.py
│   └── live.py                  ← Rich live terminal dashboard
│                                   metrics: total_queries, hit_rate, tokens_saved, $ saved
│                                   trend: hit_rate over time (rising graph)
│                                   comparison: CaSVeM vs Mem0 cost today
│
├── api/
│   ├── __init__.py
│   └── routes.py                ← FastAPI routes
│                                   POST /memory          add a memory
│                                   POST /query           query memories
│                                   GET  /stats           cache stats + cost dashboard
│                                   DELETE /memory/{id}   delete + cache invalidate
│
└── benchmark/
    ├── run_longmemeval.py        ← LongMemEval benchmark runner
    ├── run_locomo.py             ← LoCoMo benchmark runner
    ├── scorer.py                 ← LLM judge + Token F1 scoring
    └── results/                  ← JSON result files (gitignored)
```

---

## Three Built-In Optimizations

These are baked into Phase 1 from day one — not added later.

---

### Opt 1 — Semantic Cache Key Collision Detection

**Problem without it**: two queries with similar-but-not-identical vectors can hash to the same quantized key and get each other's cached results. Wrong answer returned silently.

**Fix**: store the original `query_vec` alongside every cache entry. On every cache hit, verify cosine similarity between current query and stored query. If similarity < 0.92, treat as miss.

```python
# core/cache/cache_gate.py  — verify_hit()
def verify_hit(self, query_vec: np.ndarray, entry: dict) -> bool:
    stored_vec = np.frombuffer(entry['query_vec'], dtype=np.float32)
    similarity = float(np.dot(query_vec, stored_vec))  # both L2-normalized → cosine
    return similarity >= 0.92

# cache_entries table gets one extra column:
#   query_vec  BLOB NOT NULL   ← stored alongside memory_ids
```

**Impact**: eliminates silent wrong-cache-hit errors. Essential for correctness.
**Cost**: ~20 lines of code, one extra BLOB column per cache entry.

---

### Opt 2 — Concurrent Gemini Calls for Benchmarks

**Problem without it**: LongMemEval has 500 questions. Sequential judge calls = 500 × ~1s = 8+ minutes just for scoring.

**Fix**: use `asyncio.gather()` with a semaphore to run 20 judge calls concurrently. Gemini 2.0 Flash free tier allows 15 RPM; paid tier allows 2000 RPM. Even at free tier, batching 15 at a time cuts scoring time by ~12×.

```python
# benchmark/scorer.py  — batch_judge()
import asyncio

async def batch_judge(provider, qa_pairs: list[dict], concurrency: int = 15) -> list[bool]:
    sem = asyncio.Semaphore(concurrency)

    async def judge_one(pair):
        async with sem:
            return await provider.judge(pair['question'], pair['ground_truth'], pair['answer'])

    return await asyncio.gather(*[judge_one(p) for p in qa_pairs])
```

**Impact**: LongMemEval scoring drops from 8+ min → ~40s at 15 RPM.
**Cost**: 10 lines. Uses same `judge()` interface already on `BaseLLMProvider`.

---

### Opt 3 — Real Token Counts from Gemini Response

**Problem without it**: the cost dashboard estimates savings by multiplying query count × a fixed cost assumption. This is wrong and unconvincing to investors.

**Fix**: `BaseLLMProvider.complete()` returns a `CompletionResult` dataclass that includes actual `input_tokens` and `output_tokens` from the API response. Gemini's response carries `usage_metadata.prompt_token_count` and `candidates_token_count`. `Storage.log_access()` stores real token counts. Dashboard computes exact USD cost.

```python
# core/llm/base.py
@dataclass
class CompletionResult:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0

# core/llm/gemini.py
async def complete(self, prompt, max_tokens=1024) -> CompletionResult:
    r = await self._model.generate_content_async(prompt, ...)
    return CompletionResult(
        text=r.text,
        input_tokens=r.usage_metadata.prompt_token_count,
        output_tokens=r.usage_metadata.candidates_token_count,
    )

# dashboard/live.py — exact cost
cost = (result.input_tokens / 1000 * cfg.gemini_cost_per_1k_input +
        result.output_tokens / 1000 * cfg.gemini_cost_per_1k_output)
```

**Impact**: the dashboard shows **exact dollars saved vs Mem0**, not estimates. Investor demo becomes airtight.
**Cost**: one dataclass, one extra column in access_log.

---

## LLM Provider Abstraction

The pipeline never imports `google-generativeai` directly. Everything goes through `core/llm/`.
This is the single design decision that makes adding open models a config change, not a refactor.

```python
# core/llm/base.py
from abc import ABC, abstractmethod

class BaseLLMProvider(ABC):

    @abstractmethod
    async def complete(self, prompt: str, max_tokens: int = 1024) -> str:
        """Generate a completion. Used by context builder → answer generation."""
        ...

    @abstractmethod
    async def judge(self, question: str, ground_truth: str, answer: str) -> bool:
        """Return True if answer is correct vs ground_truth. Used by benchmark scorer."""
        ...
```

```python
# core/llm/gemini.py  (Phase 1 default)
import google.generativeai as genai
from .base import BaseLLMProvider

JUDGE_PROMPT = """Question: {question}
Ground truth: {ground_truth}
Answer: {answer}

Is the answer correct or equivalent to the ground truth? Reply only YES or NO."""

class GeminiProvider(BaseLLMProvider):
    def __init__(self, model: str, api_key: str):
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model)

    async def complete(self, prompt, max_tokens=1024):
        r = self._model.generate_content(prompt,
            generation_config={"max_output_tokens": max_tokens, "temperature": 0.1})
        return r.text

    async def judge(self, question, ground_truth, answer):
        prompt = JUDGE_PROMPT.format(question=question,
                                     ground_truth=ground_truth, answer=answer)
        result = await self.complete(prompt, max_tokens=5)
        return "yes" in result.lower()
```

```python
# core/llm/ollama.py  (future open models — Qwen, Llama, Mistral, etc.)
import ollama as ol
from .base import BaseLLMProvider

class OllamaProvider(BaseLLMProvider):
    def __init__(self, model: str, base_url: str):
        self._model = model
        self._client = ol.AsyncClient(host=base_url)

    async def complete(self, prompt, max_tokens=1024):
        r = await self._client.generate(model=self._model, prompt=prompt,
                                         options={"num_predict": max_tokens})
        return r["response"]

    async def judge(self, question, ground_truth, answer):
        prompt = JUDGE_PROMPT.format(...)  # same prompt as Gemini
        result = await self.complete(prompt, max_tokens=5)
        return "yes" in result.lower()
```

```python
# core/llm/factory.py
import os
from .base import BaseLLMProvider

def get_llm_provider() -> BaseLLMProvider:
    backend = os.getenv("LLM_BACKEND", "gemini")
    model   = os.getenv("LLM_MODEL", "gemini-2.0-flash")

    if backend == "gemini":
        from .gemini import GeminiProvider
        return GeminiProvider(model=model, api_key=os.environ["GEMINI_API_KEY"])

    if backend == "ollama":
        from .ollama import OllamaProvider
        return OllamaProvider(model=model,
                              base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))

    raise ValueError(f"Unknown LLM_BACKEND: {backend}. Add a new provider in core/llm/")
```

**To add a new backend later** (e.g., OpenAI, vLLM, LMStudio, Anthropic):
1. Create `core/llm/openai.py` implementing `BaseLLMProvider`
2. Add one `if backend == "openai":` block in `factory.py`
3. Set `LLM_BACKEND=openai` in `.env`
4. Zero changes to pipeline, cache, benchmark, or API code

---

## Phase 1 — Core Innovation (8 Weeks)

**Goal**: Prove the cost curve inversion end-to-end. Full working demo for investors.

**Deliverable**: Working system that demonstrably gets cheaper per query over time.
Beat Mem0 on latency (1ms cached vs 7-10s). Show live cost savings dashboard.

---

### Week 1-2 — Foundation + LRU Cache

**Goal**: Get a working pipeline with basic LRU cache. First latency measurements.

**Build order** (strict sequence — each depends on previous):

#### Day 1-2: Storage Foundation

Build [config.py](config.py) and [core/storage.py](core/storage.py).

SQLite schema:
```sql
CREATE TABLE memories (
    id          TEXT PRIMARY KEY,
    text        TEXT NOT NULL,
    vector      BLOB NOT NULL,        -- float32 array, 384-dim
    memory_type TEXT DEFAULT 'fact',  -- fact | event | preference | decision
    project_id  TEXT,
    author_id   TEXT,
    created_at  INTEGER NOT NULL,     -- unix timestamp
    confidence  REAL DEFAULT 1.0,
    access_count INTEGER DEFAULT 0
);

CREATE TABLE access_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    query_hash  TEXT NOT NULL,        -- SHA256 of query vector quantized
    memory_ids  TEXT NOT NULL,        -- JSON array of memory IDs returned
    hit_type    TEXT NOT NULL,        -- 'L1' | 'L2' | 'cold'
    latency_ms  REAL NOT NULL,
    tokens_used INTEGER NOT NULL,
    created_at  INTEGER NOT NULL
);

CREATE TABLE cache_entries (
    query_hash  TEXT PRIMARY KEY,     -- quantized vector hash
    memory_ids  TEXT NOT NULL,        -- JSON array
    tier        TEXT NOT NULL,        -- 'L1' | 'L2'
    hit_count   INTEGER DEFAULT 0,
    created_at  INTEGER NOT NULL,
    last_hit    INTEGER NOT NULL
);
```

HNSW setup in storage.py:
```python
import hnswlib
# space='cosine', dim=384, max_elements=100_000
# M=16, ef_construction=200
# Persist to disk after every 100 adds
```

#### Day 2-3: Encoder

Build [core/encoder.py](core/encoder.py).

```python
from sentence_transformers import SentenceTransformer

class EuclideanEncoder:
    # model: all-MiniLM-L6-v2
    # encode(text: str) -> np.ndarray[float32, 384]
    # encode_batch(texts: list[str]) -> np.ndarray[N, 384]
    # normalize() before storing — enables cosine via dot product
```

First test: `encoder.encode("hello world")` → vector of shape `(384,)`.

#### Day 3-4: Roaring Bitmap + HNSW Search

Build [core/retrieval/bitmap_filter.py](core/retrieval/bitmap_filter.py):
```python
from pyroaring import BitMap

class BitmapFilter:
    # One BitMap per indexed field per value
    # e.g., date_2026_04 → BitMap of all memory row_ids with that date
    # filter(date_range, memory_type, project_id) → frozenset of candidate row_ids
    # ~30 lines total
```

Build [core/retrieval/hnsw_search.py](core/retrieval/hnsw_search.py):
```python
# search(query_vec, candidate_ids, top_k) -> [(memory_id, score)]
# key: set_ef(ef_search) before searching, NOT during index construction
# if candidate_ids is None → search entire index (Phase 1 fallback)
```

#### Day 4-5: LRU Cache Gate

Build [core/cache/lru_cache.py](core/cache/lru_cache.py):
```python
from cachetools import LRUCache
import hashlib, numpy as np

class LRUCacheGate:
    # Exact key: SHA256(quantize(query_vec))
    # quantize: round to 2 decimal places → reduces near-duplicate queries to same key
    # L1: LRUCache(maxsize=500)  — hottest 500 queries
    # L2: LRUCache(maxsize=2000) — warm 2000 queries
    # check(query_vec) -> (tier: str | None, result: list[MemoryID] | None)
    # write(query_vec, memory_ids, tier='L2')
    # promote(query_hash) -> 'L2' to 'L1' when hit_count > L1_THRESHOLD
```

Build [core/cache/access_log.py](core/access_log.py):
```python
# log_access(memory_id, tier, latency_ms, tokens_used)
# get_hit_count(query_hash) -> int
# get_stats() -> CacheStats(total, hits, misses, hit_rate, tokens_saved, cost_saved)
```

Build [core/cache/cache_gate.py](core/cache/cache_gate.py):
```python
# Unified interface that starts with LRU, swaps to MLP in week 3
# check(query_vec) -> CacheCheckResult
# write(query_vec, result)
# Phase 1 flag: USE_MLP = False  (flip to True in week 3)
```

#### Day 5-6: Reranker + Context Builder

Build [core/retrieval/reranker.py](core/retrieval/reranker.py):
```python
from sentence_transformers import CrossEncoder

class Reranker:
    # model: cross-encoder/ms-marco-MiniLM-L-6-v2
    # rerank(query: str, candidates: list[Memory], top_n: int) -> list[Memory]
    # early_exit: if scores[0] > EARLY_EXIT_THRESHOLD → return immediately
    # This is 10 lines of real logic
```

Build [core/context/builder.py](core/context/builder.py):
```python
# build(query: str, memories: list[Memory], budget: int) -> str
# Sort: combined_score = 0.6 * relevance_score + 0.4 * recency_weight
# recency_weight = exp(-decay * days_since_created)  same decay logic as v1
# Truncate: add memories until token count exceeds budget
```

#### Day 6-7: Memory Writer + Memory Updater

Build [core/memory/writer.py](core/memory/writer.py):
```python
# add(text, metadata) -> memory_id
# pipeline: encode → sqlite insert → hnsw add → bitmap update
# Returns memory_id (UUID)
```

Build [core/memory/updater.py](core/memory/updater.py):
```python
# cache_writeback(query_vec, memory_ids, context_str)
#   → writes result to cache, increments hit counts
# invalidate(memory_id)
#   → removes memory from cache entries that reference it
#   → this is cache correctness: memory changed → stale cache evicted
# decay_old_entries()
#   → UPDATE memories SET confidence = confidence * exp(-decay * days) WHERE old
```

#### Day 7-8: Full Pipeline + First Test

Build [pipeline/query.py](pipeline/query.py):
```python
# query(text, filters={}) -> QueryResult(context, memories, metadata)
# Full flow: encode → cache_check → bitmap_filter → hnsw → rerank → build → update
# metadata includes: hit_type, latency_ms, tokens_used
```

Build [pipeline/ingest.py](pipeline/ingest.py):
```python
# ingest(text, metadata={}) -> memory_id
# Full flow: encode → write → bitmap_update
```

**Week 1-2 milestone check**:
```
[ ] encoder.encode() returns (384,) vector
[ ] ingest("hello") stores in SQLite + HNSW index
[ ] query("hello") returns memory in <50ms (cold path)
[ ] cache write + read roundtrip works
[ ] second query("hello") returns from L2 cache
[ ] hit_rate showing in logs
```

---

### Week 3-4 — MLP Predictor

**Goal**: Replace LRU eviction with a learned predictor. Cache hit rate should start climbing.

**Why now**: After 2 weeks of queries, `access_log` has real training data.

Build [core/cache/mlp_predictor.py](core/cache/mlp_predictor.py):

```python
from sklearn.neural_network import MLPClassifier
import numpy as np, pickle

class MLPCachePredictor:
    # Input features (per query):
    #   query_vector[0:384]    — 384-dim embedding
    #   hour_of_day[0:23]      — one-hot, 24 dims
    #   day_of_week[0:6]       — one-hot, 7 dims
    #   query_length           — int, 1 dim
    # Total: 416 features
    #
    # Output: P(cache_hit) — float 0.0 to 1.0
    #
    # Training data: access_log table
    #   X = feature_vector per query
    #   y = 1 if hit, 0 if miss
    #
    # Architecture: hidden_layer_sizes=(128, 64)
    # Retrain: when access_log grows by 1000 new rows
    # Persist: pickle to data/mlp_model.pkl

    def train(self, access_log_rows):
        ...

    def predict(self, query_vec) -> float:
        # Returns probability of cache hit
        ...

    def should_cache(self, query_vec, threshold=0.6) -> bool:
        # Decides whether to pre-populate cache with this result
        ...
```

Swap in [core/cache/cache_gate.py](core/cache/cache_gate.py):
```python
# When USE_MLP = True:
#   check() uses MLP.predict() instead of LRU lookup
#   write() respects MLP.should_cache() to decide tier
#   retrain() called asynchronously when log threshold hit
```

**Week 3-4 milestone check**:
```
[ ] MLP trains on access_log without error
[ ] MLP.predict() returns float between 0 and 1
[ ] Cache hit rate measurably higher than LRU after 500 queries
[ ] Auto-retrain triggers when log grows by 1000 rows
```

---

### Week 5-6 — Full Pipeline Validation

**Goal**: End-to-end pipeline complete. First accuracy benchmarks.

Build [api/routes.py](api/routes.py) and [main.py](main.py):
```
POST /memory        { text, metadata }        → { id }
POST /query         { text, filters? }        → { context, memories, stats }
GET  /stats                                   → { hit_rate, tokens_saved, cost_saved, queries_total }
DELETE /memory/{id}                           → { ok }
```

Build [dashboard/live.py](dashboard/live.py):
```python
from rich.live import Live
from rich.table import Table
from rich.panel import Panel

# Live terminal display refreshing every 2 seconds:
#
# ┌─────────────────────────────────────────────────────┐
# │              CaSVeM Live Dashboard                  │
# ├────────────────────┬────────────────────────────────┤
# │ Queries served     │ 4,821                          │
# │ Cache hit rate     │ 82.3%  ↑ (was 61% at start)   │
# │ Tokens saved today │ 1,240,000                      │
# │ $ saved vs Mem0    │ $24.80 today                   │
# │ Projected/month    │ $744                           │
# │ Cache L1 entries   │ 487                            │
# │ Cache L2 entries   │ 1,923                          │
# ├────────────────────┴────────────────────────────────┤
# │ Last 5 queries:                                     │
# │ [L1 1ms]  "What did we decide about auth?"          │
# │ [L2 0.5ms] "Who is the product owner?"              │
# │ [cold 22ms] "Show me Q3 decisions"                  │
# └─────────────────────────────────────────────────────┘
```

Build [benchmark/run_longmemeval.py](benchmark/run_longmemeval.py):
```python
# Download: xiaowu0162/LongMemEval from HuggingFace
# For each item: ingest sessions → query each question → judge answer
# Categories: single_hop, multi_hop, temporal, knowledge_update, absent_info
# Output: JSON results file + printed table
```

Build [benchmark/scorer.py](benchmark/scorer.py):
```python
# LLM judge (local, uses qwen3:1.7b):
#   "Given ground truth: X. Given answer: Y. Is the answer correct? YES or NO"
# Token F1 (for LoCoMo):
#   precision, recall, F1 on token overlap
```

**Week 5-6 milestone check**:
```
[ ] FastAPI server starts and all 4 routes work
[ ] Live dashboard shows rising hit rate over 100+ queries
[ ] LongMemEval starts running (first 50 questions)
[ ] Token savings counter incrementing correctly
```

---

### Week 7-8 — Benchmarks + Demo Polish

**Goal**: Full benchmark results. Demo-ready for investors.

Run full benchmarks:
```bash
# LongMemEval (500 questions, ~3-4 hours)
python benchmark/run_longmemeval.py --output results/longmemeval_$(date +%Y-%m-%d).json

# LoCoMo
python benchmark/run_locomo.py --output results/locomo_$(date +%Y-%m-%d).json
```

Target scores:
```
LongMemEval:  beat Mem0 on temporal + knowledge_update categories
              overall accuracy competitive (within 5% of Mem0's 93.4%)
LoCoMo:       F1 > 85 (Mem0 target: 91.6)
Latency:      cached queries < 2ms (Mem0: 7-10s)
Token savings: show 70%+ reduction on warm cache
```

Demo script (investor demo):
```
1. Start with empty cache. Show hit_rate = 0%.
2. Run 10 queries on same topic. Show hit_rate climbing.
3. After 50 queries: show $X saved vs Mem0 today.
4. Show trend line: hit rate rising from 0% → 80%+.
5. Side by side: CaSVeM 1ms vs Mem0 7s for same query.
```

**Phase 1 final deliverable**:
```
[ ] Full end-to-end working system
[ ] LongMemEval benchmark results
[ ] Live dashboard showing cost savings in real time
[ ] Demo script polished and rehearsed
[ ] Publishable latency + cost results
[ ] Investor demo ready
```

---

## Phase 2 — Retrieval Quality (4 Weeks)

**When**: After Phase 1 is benchmarked and working.
**Goal**: Better accuracy on keyword-heavy queries. Higher cache hit rate. Lower latency.

### Hybrid Bridge (Weeks 9-10)

Add BM25 keyword search alongside the existing dense HNSW search.

```
Dense path:   query_vec  →  HNSW  →  dense_results
Keyword path: query_text →  BM25  →  keyword_results
Fusion:       RRF(dense_results, keyword_results) → merged_results
```

Libraries:
- `rank_bm25` — BM25 implementation (~50 lines to integrate)
- RRF (Reciprocal Rank Fusion) — 10 lines to implement

When to use which path:
```python
# Simple query classifier (not ML, just heuristics):
# keyword_weight = 0.7 if query has quotes or exact names
# keyword_weight = 0.3 otherwise (default to dense-heavy)
```

### Two-Tier Cache Keys (Week 10)

Current: one key = quantized vector hash

New: two keys per query:
- **Exact key**: SHA256(raw query vector, 4 decimal places)
- **Semantic key**: SHA256(quantize(query vector, 1 decimal place)) — catches paraphrases

```
"What did we decide about auth?" → exact_key_A + semantic_key_B
"What was decided about authentication?" → exact_key_C + semantic_key_B  ← SAME semantic key
                                                                             → cache hit!
```

Expected impact: cache hit rate +10-15% from paraphrase matching.

### Async Encoding (Week 11)

Current: encode → cache_check → ...

New: encode + cache_check run in parallel:
```python
import asyncio

async def query(text, filters):
    encode_task  = asyncio.create_task(encoder.encode_async(text))
    # While encoding, we can check exact-key cache immediately
    # (exact key doesn't need vector — uses raw text hash)
    ...
```

Expected impact: -15ms on cache miss path (encoding no longer blocks).

### Reranker Optimization (Week 12)

Add partial rerank:
```python
# If top HNSW result score > 0.9 → only rerank top-3 (not top-50)
# If top HNSW result score > 0.95 → skip rerank entirely (early exit already there)
```

Expected impact: -30ms on high-confidence queries.

**Phase 2 deliverable**:
```
[ ] Hybrid bridge integrated + benchmarked
[ ] Two-tier keys show measurable cache hit improvement
[ ] Async encoding shaves 15ms off miss path
[ ] Reranker partial mode working
[ ] Updated LongMemEval + LoCoMo results showing improvement
```

---

## Phase 3 — Research Contributions (Ongoing)

**When**: After Phase 2 is benchmarked.
**Goal**: Paper-worthy results beyond the core claim.

### Predictive Pre-fetching

CDN-style pre-warming the cache:
- After a query lands, predict what the next 3 likely queries are
- Pre-fetch those answers into L2 cache before they're asked
- Requires: pattern data from access_log (available after Phase 1)
- Complexity: HIGH — requires sequence modeling or markov chain on query patterns

### Learned Eviction Policy

Replace LRU with ML-driven eviction:
- Currently: LRU evicts least-recently-used entry
- Proposed: train eviction model on access patterns (similar to learned indexes in Bigtable)
- Predicts future access probability for each cache entry
- Evicts entries with lowest predicted future access

### Composite Bitmap Indexes

Phase 1 bitmaps: single-field indexes (date OR type OR project)

Phase 3: multi-field compound indexes:
```
date_2026_04 AND memory_type_decision AND project_auth → BitMap intersection
```

High-cardinality optimization for enterprise deployments with thousands of projects.

---

## Build Timeline Summary

```
Week 1-2:  Storage + Encoder + Roaring Bitmap + HNSW + LRU Cache + Writer
           → Basic retrieval + first cache hits

Week 3-4:  MLP Predictor + Cache Gate swap
           → Cache hit rate starts climbing

Week 5-6:  Full API + Live Dashboard + Memory Updater + LongMemEval runner
           → First benchmark results

Week 7-8:  Benchmark runs + Polish + Demo script
           → Investor-ready

Week 9-10: Hybrid Bridge + Two-tier cache keys
           → Better accuracy on keyword queries

Week 11:   Async encoding
           → Lower latency on miss path

Week 12:   Reranker optimization + Phase 2 benchmarks
           → Final Phase 2 numbers

Week 13+:  Phase 3 research (ongoing, paper material)
```

---

## Requirements File

```
# requirements.txt

# Core
sentence-transformers>=3.0.0
hnswlib>=0.8.0
pyroaring>=0.4.0
scikit-learn>=1.4.0
numpy>=1.26.0

# LLM backends
google-generativeai>=0.8.0      # Gemini (default)
ollama>=0.3.0                   # open models (optional, install when switching)

# API
fastapi>=0.111.0
uvicorn>=0.30.0
python-multipart>=0.0.9
pydantic>=2.0.0

# Dashboard
rich>=13.0.0

# Utils
python-dotenv>=1.0.0

# Phase 2
rank-bm25>=0.2.2

# Benchmarks
datasets>=2.19.0
huggingface_hub>=0.23.0
tabulate>=0.9.0

# Dev
pytest>=8.0.0
```

---

## What Makes This Different From v1/v2

| | v1/v2 | v3 |
|--|-------|-----|
| Core bet | 5-layer hierarchy = better structure | Learned cache = lower cost |
| Storage | Weaviate (graph+vector) | SQLite + hnswlib (simple, fast) |
| USP | Better accuracy via structure | Lower cost via caching |
| Novelty | Layer compression + retention decay | Learned cache predictor + cost curve inversion |
| Investor pitch | "Better accuracy" (commodity) | "Gets cheaper at scale" (unique) |
| Demo | benchmark scores | live cost dashboard + latency numbers |
| Paper contribution | memory hierarchy for LLMs | learned caching for AI memory systems |

---

## The Investor Demo Path

```
Phase 1 done → demo working system + cost dashboard → seed round pitch
Phase 2 done → accuracy numbers + hybrid retrieval → Series A material
Phase 3 done → full paper + enterprise features → defensible moat
```

The single most important thing to build first is the **cache hit rate rising over time** graph.
That graph IS the pitch. Everything else is supporting evidence.
