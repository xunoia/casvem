# CaSVeM — Cached Smart Vector Memory
### Complete Architecture & Business Summary

---

## What Is CaSVeM?

CaSVeM (Cached Smart Vector Memory) is a hierarchical memory architecture for local LLMs, inspired by CPU cache hierarchy principles (L1/L2/L3/RAM/Storage). It solves the fundamental problem that LLMs are stateless — they forget everything between sessions.

Unlike existing solutions (Mem0, Supermemory, MemGPT), CaSVeM is built **local-first** — all intelligence runs on your own hardware, zero cloud dependency, zero per-query cost after setup.

---

## The Core Problem Being Solved

```
LLMs are stateless.
Every session starts from zero.
Current solutions either:
  - Use flat RAG (dumb vector search, no memory management)
  - Require cloud APIs (expensive, privacy risk, no local option)
  - Have no hierarchy (everything at the same level)

CaSVeM solves all three.
```

---

## How CaSVeM Differs From RAG

| | RAG | CaSVeM |
|---|---|---|
| What it stores | Documents (static) | Knowledge (evolving) |
| Temporal awareness | None | Layer-specific decay |
| Personalisation | Same for everyone | Per-user memory |
| Output | Raw chunks | Query-aware synthesis |
| Learning | None | Updates every session |
| Structure | Flat | 5-layer hierarchy |
| Causal awareness | None | Memory chains |

**In one sentence:** RAG is a search engine. CaSVeM is a memory management system.

---

## The 5-Layer Memory Hierarchy

Modelled directly on CPU cache architecture:

```
L1  Ultra-compressed context        ~500 tokens   | 20 lines
    Always auto-injected into every prompt
    Core facts, active preferences, current projects

L2  Compressed topic summaries      ~5K tokens    | 100 lines
    Searched first after L1
    Per-topic summaries with temporal context

L3  Detailed knowledge              ~50K tokens   | 500 lines
    Followed via pointers from L2
    Context-rich, nuanced, temporally aware

L4  Raw extracted facts             ~500K tokens  | 5000 lines
    Atomic facts from sessions
    Timestamped, tagged, deduplicated

L5  Raw session archive             Unlimited
    Full conversation transcripts
    Append-only, never modified, source of truth
```

### Memory Line Structure (every layer)

Every single memory entry at every layer contains:

```json
{
  "id": "mem_L2_0042",
  "layer": 2,
  "content": "User prefers Vue over React",
  "embedding": [...],
  "source_pointers": ["mem_L3_0091", "mem_L3_0094"],
  "derived_by": ["mem_L1_0008"],
  "retention_score": 0.82,
  "importance": 0.85,
  "recency": 0.76,
  "access_count": 14,
  "last_accessed": "2026-04-14",
  "created_at": "2026-01-15",
  "dirty": false,
  "status": "active",
  "topic_tags": ["frontend", "frameworks", "preferences"],
  "confidence": "high"
}
```

`source_pointers` — points DOWN (where this was derived from)
`derived_by` — points UP (what summarised this)

---

## Retention Score Formula

```
retention_score =
  (importance     × 0.40) +
  (recency        × 0.30) +
  (access_frequency × 0.20) +
  (uniqueness     × 0.10)

recency           = e^(-λ × days_since_access)
                    L1: λ=0.1  L2: λ=0.05  L3: λ=0.02  L4: λ=0.01

access_frequency  = normalised(access_count / age_in_days)

uniqueness        = 1 - max_cosine_similarity_to_neighbours
```

---

## The Write System

Runs after every conversation. Async — never blocks reads.

**Step 1 — Session Capture (→ L5)**
- Full transcript saved as-is
- Permanent, append-only, never modified

**Step 2 — Fact Extraction (L5 → L4)**
- Local LLM reads session, extracts atomic facts as JSON
- Each fact: `{fact, importance, category, topic_tags}`
- Deduplication check before insert (cosine similarity > 0.92)
- New L4 line created with pointer to L5 session

**Step 3 — Consolidation (L4 → L3, async)**
- Group new L4 lines by topic_tags
- Local LLM classifies each new fact:
  - `CONTRADICTS` → archive old L3, create new, keep historical
  - `ADDS` → update L3 content, mark dirty
  - `REINFORCES` → update recency only, no re-summarise
  - `IRRELEVANT` → ignore

**Step 4 — Compression (L3 → L2 → L1, background)**
- Find dirty L3 lines, follow derived_by pointer to parent L2
- Pull ALL L3 lines feeding into that L2
- Local LLM compresses to 2-3 sentences max
- Re-embed, update source_pointers, mark L2 dirty
- Repeat for L2 → L1 (even more aggressive compression)

### Dirty Flag System

```
dirty = False  →  line accurately reflects source layers
dirty = True   →  something below changed, line is stale

Flags set:     immediately when lower layer changes
Flags cleared: by compressor engine after re-summarisation
During reads:  stale lines served with low confidence flag
               + background refresh triggered
```

---

## The Read System

Runs before every prompt to the main LLM.

**Step 1 — L1 Auto-inject**
Always included, no search needed, ~500 tokens.

**Step 2 — Query Analysis**
Local LLM analyses query intent, outputs:
```json
{
  "topics": ["frontend", "frameworks"],
  "memory_type": "preference + project context",
  "time_relevance": "current",
  "search_terms": ["Vue preference", "framework decision", "React vs Vue"]
}
```

**Step 3 — Hierarchical Search**
- Vector search L2 using generated search terms
- Local LLM evaluates: sufficient or go deeper?
- If deeper needed: follow source_pointers into specific L3 lines only
- If still insufficient: follow L3 pointers into L4 (rare)

**Step 4 — Confidence Scoring**
```
Count supporting lines across layers
Count sessions spanned
Check recency of most recent confirmation
→ Assign high / medium / low confidence
```

**Step 5 — Memory Synthesis**
Local LLM synthesises a query-specific memory block:
- Max 1000 tokens
- Organised by relevance to this specific query
- Low-confidence items flagged explicitly
- Not raw chunks — a coherent, tailored briefing

**Step 6 — Output**
`{query + synthesised memory block}` sent to any LLM of user's choice.

---

## The Promotion / Demotion Engine

Runs on a schedule (every 60 minutes). Never blocks reads or writes.

**Promotion** — lines above threshold move up:
```
If retention_score > promotion_threshold
AND access_count > Y in last Z days
→ promote to layer above
→ if target layer full: swap with lowest scoring line there
```

**Demotion** — cold lines move down:
```
If retention_score < demotion_threshold
AND last_accessed > N days ago
→ verify source lines exist in lower layer
→ demote, update pointers
→ never delete — only demote or archive
```

**Merge** — prevents bloat:
```
If cosine_similarity > 0.88 between two lines in same layer
→ Local LLM: "Are these the same fact?"
→ Yes: merge into one, combine all pointers, sum access counts
→ No:  keep both
```

### Layer Thresholds

| Layer | Max Tokens | Max Lines | λ Decay | Promote | Demote |
|---|---|---|---|---|---|
| L1 | 500 | 20 | 0.10 | 0.85 | 0.30 |
| L2 | 5,000 | 100 | 0.05 | 0.75 | 0.20 |
| L3 | 50,000 | 500 | 0.02 | 0.65 | 0.15 |
| L4 | 500,000 | 5,000 | 0.01 | 0.55 | 0.10 |

---

## Memory Chains

Typed directed graphs overlaid on the vector store. Enable causal and temporal retrieval that semantic similarity search is blind to.

### Chain Types

```
TEMPORAL    — what happened in order
              "conversation history", "project timeline"

CAUSAL      — what caused what
              "decision chains", "bug → fix → regression"

DEPARTMENTAL — grouped by domain
              "finance conversations", "HR decisions"

ENTITY      — everything about one thing
              "all memories about [person X]"

CONTRADICTORY — evolution of a belief
              "we thought X → learned Y → now believe Z"
```

### Chain Membership in Memory Line

```json
{
  "chain_memberships": [
    {
      "chain_id": "chain_decisions_2026",
      "chain_type": "decision",
      "position": 4,
      "prev_id": "mem_L3_0087",
      "next_id": "mem_L3_0103",
      "causal_label": "caused_by"
    }
  ]
}
```

One memory line can belong to multiple chains simultaneously.

### Chain vs Vector Search

```
Vector search answers: "What is similar to this?"
Chain search answers:  "What led to this, what came after, why?"

Chain traversal speed:
  Pure vector + SQLite:  ~200ms (20 SQL calls)
  Graph-based chains:    ~20ms  (1 traversal)
```

---

## Intelligence Routing Layer

CaSVeM decides automatically whether to answer locally or route to a cloud LLM.

```
Query + retrieved memory
        ↓
Local LLM classifies:
  "Is this simple or complex?"
        ↓
SIMPLE → Local LLM answers directly
         Zero cloud cost, instant response

COMPLEX → Package:
           - Query
           - Synthesised memory block
           - Local LLM partial analysis
           - Any attached files (PDF, doc, image)
          → Route to cloud LLM of user's choice
```

### Routing Signals

```
Always local:                    Always cloud:
─────────────                    ────────────
Fact retrieval                   External files attached
Single chain lookup              Long-form generation requested
Preference recall                Complex multi-source reasoning
High confidence memory           Deep analysis required
Recent session recall            Low confidence on critical facts

Borderline:
  Local LLM tries first
  Self-evaluates confidence
  Escalates to cloud if unsure
```

### Cost Reality

```
~70% of queries → $0 (local)
~20% of queries → $0 (local, medium complexity)
~10% of queries → cloud API rate (with tiny context)

vs Mem0/Supermemory: 100% of queries pay cloud API
```

---

## The Graph + Tree Data Structure

### Why Graph Over Pure Vector

| Feature | Pure Vector | Graph Vector |
|---|---|---|
| Causal relationships | ❌ | ✅ native edges |
| Temporal ordering | ❌ | ✅ built in |
| Chain traversal | ~200ms (20 SQL calls) | ~20ms (1 traversal) |
| Contradiction detection | Manual | Native edge |
| Cross-topic links | ❌ | ✅ |
| Write + pointer update | Dual system | Atomic |

### CaSVeM Uses Both Tree + Graph

```
Tree structure:   handles parent-child hierarchy
                  (L1 → L2 → L3 → L4 → L5)

Graph structure:  handles cross-layer relationships
                  (chains, contradictions, entity links)

Together:         tree backbone + graph edges
                  = maximum retrieval flexibility
```

### Recommended DB: Weaviate

Replaces Qdrant + SQLite with one unified system:
- Vector search on nodes
- Graph edges native
- Self-hosted via Docker
- Apache 2.0 license

---

## Tech Stack

```
Local LLM:      Qwen 3.5 2B via Ollama (current)
                → upgrade to Gemma 4 26B (target)
                → or Qwen 3.6 Plus 35B-A3B (intermediate)

Embeddings:     qwen3-embedding:0.6b via Ollama

Vector DB:      Qdrant (v1) → Weaviate (v2, graph-native)

Metadata:       SQLite (v1) → replaced by Weaviate (v2)

API:            FastAPI (Python)

Scheduler:      APScheduler (promotion/demotion engine)

Language:       Python, async throughout
```

### Provider Abstraction

All LLM calls go through a provider interface. Swap any model by changing one line in `config.py`. No other code changes required.

```
config.py:
  LLM_FAST   = OllamaLLMProvider("qwen3.5:2b")
  LLM_STRONG = OllamaLLMProvider("qwen3.5:2b")
  EMBEDDER   = OllamaEmbeddingProvider("qwen3-embedding:0.6b")
```

### Model Upgrade Path

```
NOW:    Qwen 3.5 2B          (2.7GB, you have this)
NEXT:   Qwen 3.6 Plus 35B    (3B active, ~8GB VRAM)
IDEAL:  Gemma 4 26B          (4B active, ~8GB VRAM, native tool calling)
FUTURE: DeepSeek V4 Flash    (13B active, MIT, watch for Ollama support)
```

---

## Project Structure

```
scavem/
├── config.py                  ← only file changed to upgrade models
├── models.py                  ← Pydantic data models
├── providers/
│   ├── base.py                ← abstract LLMProvider interface
│   ├── ollama_provider.py     ← Ollama implementation
│   └── router.py              ← task routing + fallback
├── database/
│   ├── sqlite_store.py        ← metadata, pointers, dirty flags
│   └── qdrant_store.py        ← vector storage per layer
├── engines/
│   ├── write/
│   │   ├── extractor.py       ← L5 → L4
│   │   ├── consolidator.py    ← L4 → L3
│   │   └── compressor.py      ← L3 → L2 → L1
│   └── read/
│       ├── analyser.py        ← query intent analysis
│       ├── searcher.py        ← hierarchical vector search
│       └── synthesiser.py     ← query-aware memory block
├── scheduler.py               ← promotion/demotion engine
├── api.py                     ← FastAPI endpoints
├── main.py                    ← entry point
└── requirements.txt
```

### API Endpoints

```
POST /session           → submit conversation (triggers write pipeline)
POST /query             → submit query (runs read pipeline)
GET  /memory/{layer}    → list memory lines at a layer
GET  /memory/{id}       → get specific line with pointers
GET  /status            → system status, counts, scheduler state
POST /admin/consolidate → manually trigger consolidation
POST /admin/promote     → manually trigger promotion/demotion
```

---

## Competitive Comparison

| Feature | Mem0 | Supermemory | CaSVeM |
|---|---|---|---|
| Architecture | Flat + graph variant | Knowledge graph | 5-layer cache hierarchy |
| Retrieval | Algorithmic (3-signal) | Graph + hybrid | Local LLM as cache controller |
| Memory chains | ❌ | Partial (3 types) | ✅ typed directional |
| Local-first | ❌ | ❌ | ✅ fully local |
| Intelligence routing | ❌ | ❌ | ✅ auto local/cloud |
| Query-aware synthesis | ❌ raw chunks | ❌ raw chunks | ✅ synthesised per query |
| Per-query cost | Pays cloud API | Pays cloud API | ~$0 |
| Privacy | Their servers | Cloudflare | Your device only |
| Open source | ✅ | ✅ | ✅ planned |
| Benchmarks | 91.6 LoCoMo | 85.4 LongMemEval | TBD |

### Why Competitors Use Algorithms Not LLMs for Retrieval

```
Mem0, Supermemory, Zep all use algorithmic retrieval
because every LLM call costs money and adds latency.

CaSVeM can use a local LLM as the cache controller
because local inference costs $0 per call.

This is the core architectural advantage that
the cost constraints of cloud systems made
impossible for everyone else to build.
```

---

## Novel Research Contributions

1. **CPU cache hierarchy applied to LLM memory** — 5 layers with configurable compression ratios
2. **Bidirectional semantic pointer graph** — surgical deep retrieval without full layer scans
3. **LLM-as-cache-controller** — local model decides retrieval depth dynamically
4. **Query-aware memory synthesis** — custom memory block per query, not raw chunks
5. **Composite retention scoring** — importance + recency + frequency + uniqueness with layer-specific decay
6. **Typed memory chains** — causal/temporal traversal orthogonal to semantic similarity
7. **Intelligence routing layer** — automatic local vs cloud routing based on query complexity
8. **Fully local pipeline** — zero cloud dependency, privacy preserving, zero marginal cost

### Benchmarks to Beat (Paper Targets)

```
Benchmark       Mem0 score    Target
──────────      ──────────    ──────
LoCoMo          91.6          > 91.6
LongMemEval     93.4          > 93.4
BEAM 1M         64.1          > 64.1
BEAM 10M        48.6          > 48.6
Token/query     ~7,000        < 7,000 (local ≈ free)
```

### Mem0's Stated Unsolved Problems (CaSVeM Solves These)

```
Mem0 unsolved:                   CaSVeM solution:
───────────────────────          ──────────────────
Temporal abstraction             Retention scores with
                                 layer-specific decay

Cross-session structure          Bidirectional pointer graph
                                 connects facts across sessions

Agent-native async memory        Write system is fully async,
                                 never blocks reads
```

---

## Research Paper

**Title:**
"CaSVeM: A Hierarchical Cache-Inspired Vector Memory Architecture for Efficient Long-Term LLM Context Management"

**Core thesis:**
Applying CPU cache hierarchy principles — layered storage, pointer-based lazy loading, and dynamic promotion/demotion — to LLM memory systems, achieving efficient long-term context retrieval without full context window costs.

**Target venues:** ACL, EMNLP, NAACL, or arXiv preprint

**Paper framing:**
> "We present CaSVeM, a hierarchical cache-inspired memory architecture that addresses the unsolved problems in current memory systems — temporal abstraction, cross-session structure, and agent-native asynchronous memory — while achieving zero marginal token cost through fully local inference."

---

## Business Model

### Positioning

```
NOT: "memory for LLMs" (Mem0 owns this)
YES: "the memory layer for local AI"
     "Mem0 for people who won't send data to the cloud"
     "fully private, fully local, zero per-query cost"
```

### Revenue Streams

```
1. Hosted Cloud API (primary)
   Same CaSVeM architecture, you run the servers
   $0 free tier → $29/mo starter → $99/mo growth → $499/mo scale

2. Enterprise Licenses
   Self-hosted with support, SLA, compliance docs
   $2,000-$10,000/month per customer
   Healthcare, legal, finance — can't use cloud AI

3. Consumer App
   Personal AI that remembers everything
   Runs on their device, encrypted, private
   $10-20/month subscription

4. Consulting / Integration (early stage)
   $5,000-$50,000 per engagement
```

### The Open Source Strategy

```
FREE (open source):              PAID (commercial):
──────────────────               ──────────────────
Core CaSVeM engine               Hosted cloud version
Basic memory layers              Enterprise support SLA
Local deployment                 Compliance features
Community support                Advanced analytics
                                 Multi-user features
                                 Custom integrations
```

### Distribution Plan

```
1. Finish CaSVeM + publish arXiv paper
2. GitHub with clean README + architecture diagrams
3. Post on r/LocalLLaMA (exact target audience)
4. Post Show HN on HackerNews
5. Integrate with Ollama ecosystem, Open WebUI
6. Build MCP server (works with Claude, Cursor, etc.)
7. First enterprise customer conversation
8. Apply to YC (Fall 2026 with traction)
```

### MCP / Connector Strategy

CaSVeM as a FastAPI service is trivially wrappable as an MCP server. One MCP build surfaces CaSVeM to every MCP-compatible tool:

```
Claude.ai ✓    Cursor ✓    Windsurf ✓
Continue.dev ✓    Any future MCP app ✓ automatically
```

---

## IP Protection

```
Step 1: File provisional patent (~$2,000-3,000)
        Before publishing anything publicly
        Secures your filing date for 12 months

Step 2: Submit arXiv paper
        Establishes public prior art
        Nobody can patent your ideas after this date

Step 3: Open source on GitHub (Apache 2.0)
        Requires attribution in all derivative works
        GitHub commit history = timestamped proof of invention

Step 4: Be the known inventor publicly
        Paper + GitHub + blog posts + talks = your name
        attached to CaSVeM permanently
```

---

## Known Limitations & Mitigations

| Limitation | Severity | Mitigation |
|---|---|---|
| Routing errors (silent failure) | 🔴 High | Better model = near-zero errors |
| Cold start problem | 🟠 Medium | Pre-populate from documents on setup |
| Local LLM quality ceiling | 🟠 Medium | Upgrade hardware → better model |
| Latency on first query | 🟡 Low-Medium | Graph DB reduces 3-5x |
| Chain maintenance | 🟡 Low | Graph edges make this native |
| Dirty flag cascades | 🟡 Low | Graph traversal speeds propagation |
| Hardware dependency | 🟡 Low | Hosted version covers non-local users |
| System complexity | 🟡 Low | One-command install hides complexity |

---

## Next Steps

```
Immediate (this week):
  ✓ Register casvem.ai and casvem.com (done)
  ✓ Create GitHub repo
  → Finish building in Claude Code
  → Pull qwen3-embedding:0.6b
  → Start Docker + Qdrant

Short term (2-4 weeks):
  → Get system working end to end
  → Write arXiv paper (separate chat started)
  → Write good README with architecture diagram
  → Apply to YC May 4th deadline

Medium term (month 2-3):
  → Post on r/LocalLLaMA + HackerNews
  → Run benchmarks (LongMemEval, LoCoMo)
  → Build MCP server
  → Launch casvem.ai landing page
  → Find first 5 real users

Phase 2 (after working prototype):
  → Migrate to Weaviate (graph-native DB)
  → Replace SQLite pointer logic with graph edges
  → Benchmark speed improvements
  → Add to research paper
```

---

*CaSVeM — Cached Smart Vector Memory*
*Built for the local AI era.*
