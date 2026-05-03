# CaSVeM — Benchmark Plan
### One Model · Three Benchmarks · Prove the Concept Today

---

## Model Decision: ONE Model, Best Available

You only need one model. Here is why and which one:

```
YOUR LIST           VERDICT
─────────────────   ─────────────────────────────────────────────
Gemma 4 26B         NOT YET on Ollama stable — skip for now
Qwen 3.6 Plus  ◄─── THIS ONE. Best quality/cost on local hardware
DeepSeek V4 Flash   No Ollama support yet — skip
Qwen 3.5 2B         Too weak for benchmark-quality extraction
DeepSeek V4         Impractical locally — skip
Kimi K2.6           No local deployment — skip
```

**Use: `qwen3:30b-a3b` (Qwen3 30B, Mixture-of-Experts)**

Why this wins:
- MoE = only 3B parameters ACTIVE at inference → fast like a 3B, smart like a 30B
- Needs ~8–10GB VRAM — fits on consumer hardware
- Instruction following and structured JSON output: excellent
- This is what your doc calls "Qwen 3.6 Plus"

```bash
ollama pull qwen3:30b-a3b    # do this NOW, runs in background
```

---

## How the Benchmarks Work (the three Mem0 used)

### Benchmark 1 — LongMemEval  ← START HERE

**What it is:** 500 QA questions over simulated long-term chat histories. Spans days, weeks, months of fictional conversations.

**Published by:** xiaowu0162 (Stanford / CMU), 2024  
**Dataset:** `xiaowu0162/LongMemEval` on HuggingFace  
**Mem0 score:** ~93.4% (reported in their paper/YC app)

**The 5 question categories:**

```
Category            What it tests in CaSVeM           Example question
──────────────────  ──────────────────────────────────────────────────────────
Single-hop recall   L4→L3 extraction quality           "What gym does the user go to?"
Multi-hop recall    Pointer chain traversal (L2→L3)    "What did the user decide after
                                                         the budget meeting in March?"
Temporal reasoning  Retention score decay (recency λ)  "What was the user's job title
                                                         two months ago?"
Knowledge update    CONTRADICTS detection               "What framework does the user
                                                         prefer NOW?" (changed from Vue→React)
Absent info         Hallucination resistance            "What is the user's salary?"
                                                         (never mentioned — must say unknown)
```

**How it works step by step:**

```
                    ┌─────────────────────────────────────┐
                    │         LongMemEval Dataset          │
                    │                                     │
                    │  session_1.json  session_2.json ...  │
                    │  "Day 1: User said he works at..."   │
                    │  "Day 15: User mentioned he quit..." │
                    │  "Day 30: User started new job..."   │
                    └───────────────┬─────────────────────┘
                                    │ feed all sessions
                                    ▼
                    ┌─────────────────────────────────────┐
                    │         YOUR MEMORY SYSTEM           │
                    │         (CaSVeM write pipeline)      │
                    │                                     │
                    │  L5: raw session saved              │
                    │  L4: facts extracted                │
                    │  L3: facts consolidated             │
                    │  L2: topics compressed              │
                    │  L1: core context built             │
                    └───────────────┬─────────────────────┘
                                    │ memory is built
                                    ▼
                    ┌─────────────────────────────────────┐
                    │         500 TEST QUESTIONS           │
                    │                                     │
                    │  Q: "What is the user's job now?"   │
                    └───────────────┬─────────────────────┘
                                    │ query through read pipeline
                                    ▼
                    ┌─────────────────────────────────────┐
                    │         YOUR MEMORY SYSTEM           │
                    │         (CaSVeM read pipeline)       │
                    │                                     │
                    │  1. L1 auto-inject                  │
                    │  2. L2 vector search                │
                    │  3. Follow pointers → L3 if needed  │
                    │  4. Synthesise memory block          │
                    │  5. Answer generation               │
                    └───────────────┬─────────────────────┘
                                    │ your answer
                                    ▼
                    ┌─────────────────────────────────────┐
                    │            SCORING                   │
                    │                                     │
                    │  Ground truth: "Software Engineer    │
                    │                at Acme Corp"        │
                    │  Your answer:  "Software engineer    │
                    │                at Acme"             │
                    │  Score: LLM-judge (0 or 1)          │
                    └─────────────────────────────────────┘
```

**Scoring method:**
- Original paper uses GPT-4o as judge → expensive
- Your approach: use `qwen3:30b-a3b` as judge → $0
- Judge prompt: `"Given ground truth: X. Given answer: Y. Is the answer correct? Reply only YES or NO."`
- Final score = (YES count / total questions) × 100

---

### Benchmark 2 — LoCoMo

**What it is:** Real-world long conversation dataset. Actual human conversations spanning 300–500 turns. Tests recall of specific facts mentioned early in very long dialogues.

**Dataset:** `snap-research/LoCoMo` on HuggingFace  
**Mem0 score:** 91.6 (F1 score, not %)

**How scoring works differently from LongMemEval:**

```
Ground truth answer:  "He visited Paris in the summer"
Your answer:          "The user went to Paris"

Token F1 scoring:
  Precision = (matching tokens) / (your answer tokens)
            = 2 matching ("Paris", "went/visited") / 5 = 0.40
  Recall    = (matching tokens) / (ground truth tokens)
            = 2 / 7 = 0.29
  F1        = 2 × (P × R) / (P + R) = 0.33

Average F1 across all questions = your LoCoMo score
```

**No LLM judge needed** — pure token overlap. Fully automated. Fast.

**How it works:**

```
  LoCoMo dataset
  ┌──────────────────────────────────────────────┐
  │  One entry = one very long conversation      │
  │  (~300 turns, real people chatting)          │
  │                                              │
  │  Turn 1:  "I went to Tokyo last month"       │
  │  Turn 2:  "Met my friend Kenji there"        │
  │  ...                                         │
  │  Turn 300: "What do you remember about me?"  │
  │                                              │
  │  + QA pairs:                                 │
  │    Q: "Where did the user travel?"           │
  │    A: "Tokyo"                                │
  └──────────────────────────────────────────────┘
          │
          │ pipe through CaSVeM write pipeline
          ▼
   [Memory built in layers]
          │
          │ query each question through read pipeline
          ▼
   [Answer generated]
          │
          │ compute token F1 vs ground truth
          ▼
   [Score]
```

---

### Benchmark 3 — BEAM (future, not today)

**What it is:** Tests at extreme scale — 1 million token conversations (BEAM-1M) and 10 million tokens (BEAM-10M). Stress tests whether memory systems degrade at scale.

**Mem0 scores:** 64.1 (1M), 48.6 (10M)  
**Status:** Newer benchmark, less tooling available  
**Your plan:** Run after LongMemEval and LoCoMo are working. BEAM is where the graph structure difference will be most visible (10M tokens = thousands of sessions = graph traversal wins hard).

---

## What These Benchmarks Actually Test About CaSVeM's Architecture

```
CaSVeM novel feature          Which benchmark exposes it         Why
─────────────────────────     ──────────────────────────────     ─────────────────────────────
5-layer hierarchy             LongMemEval multi-hop              Pointer chain finds facts flat
                                                                  RAG misses (different layers)

Retention score + decay       LongMemEval temporal               Decayed facts should rank lower
                                                                  → correct "what changed" answer

CONTRADICTS detection         LongMemEval knowledge update       Old fact archived, new one surfaces
                                                                  → correct answer despite conflict

Graph chain traversal         BEAM 10M                           At scale, SQL joins hit 200ms+;
                                                                  graph traversal stays at ~20ms

Query-aware synthesis         All three benchmarks               Raw chunk retrieval returns noise;
                                                                  synthesis returns signal

L1 auto-inject                LongMemEval absent info            Core facts always present prevents
                                                                  "I don't know" on known facts

Absent info (hallucination)   LongMemEval absent category        Flat RAG hallucinates; CaSVeM
                                                                  confidence scoring flags unknowns
```

---

## What You Need to Build (Minimal for Benchmark)

You do NOT need the full production system. You need a **benchmark harness** that implements the core concept. Graph structure (Weaviate) comes AFTER this proves the concept.

```
Minimum to run LongMemEval + LoCoMo:

casvem/
├── core/
│   ├── config.py           ← model config (one line to change model)
│   ├── llm.py              ← Ollama wrapper (call any model)
│   ├── embedder.py         ← embedding calls (qwen3-embedding:0.6b)
│   ├── storage.py          ← SQLite + JSON (replace with Weaviate later)
│   │
│   ├── write/
│   │   ├── extractor.py    ← T1: session → L4 facts (LLM call)
│   │   ├── consolidator.py ← T2: CONTRADICTS/ADDS/REINFORCES/IRRELEVANT
│   │   └── compressor.py   ← T3+T4: L3→L2→L1 compression
│   │
│   └── read/
│       ├── analyser.py     ← T5: query → search terms (LLM call)
│       ├── searcher.py     ← vector search per layer + pointer follow
│       └── synthesiser.py  ← T6+T7: memory block + answer generation
│
└── benchmark/
    ├── run_longmemeval.py  ← downloads dataset, runs pipeline, scores
    ├── run_locomo.py       ← downloads dataset, runs pipeline, F1 score
    └── results/
```

**Build order (strict sequence):**

```
Day 1 morning:   llm.py + embedder.py + storage.py  (foundation, no LLM calls yet)
Day 1 afternoon: extractor.py + storage  (can run first write test)
Day 1 evening:   consolidator.py + compressor.py  (full write pipeline done)
Day 2 morning:   analyser.py + searcher.py + synthesiser.py  (read pipeline)
Day 2 afternoon: run_longmemeval.py  (first real benchmark run)
Day 2 evening:   score analysis, fix weak spots, re-run
Day 3:           run_locomo.py  (second benchmark)
```

---

## Diagrams: CaSVeM vs Flat RAG on the Same Benchmark

### How Flat RAG answers a LongMemEval question

```
User sessions (30 sessions over 3 months)
         │
         │ chunk + embed everything
         ▼
┌─────────────────────────────────┐
│  Vector DB (flat, no structure) │
│  chunk_001: "user works at..."  │
│  chunk_002: "user said he..."   │
│  chunk_003: "user mentioned..." │
│  chunk_099: "user now works..." │ ← NEWER (user changed jobs)
│  ...1000 chunks                 │
└───────────────┬─────────────────┘
                │
  Q: "What is the user's current job?"
                │ cosine similarity search
                ▼
  Returns top-5 chunks by similarity
  chunk_001 scores 0.82 ← WRONG (old job)
  chunk_099 scores 0.79 ← RIGHT but ranked lower
                │
                ▼
  Answer: "User works at OldCompany"  ← WRONG
  Score: 0 (LongMemEval knowledge update test)
```

### How CaSVeM answers the same question

```
User sessions (30 sessions over 3 months)
         │
         │ write pipeline: extract → consolidate → compress
         ▼
┌──────────────────────────────────────────────┐
│  L4: "user works at OldCompany [Jan 2026]"   │ ← status: archived
│  L4: "user quit OldCompany [Feb 2026]"       │ ← consolidation flagged CONTRADICTS
│  L4: "user joined NewCompany [Mar 2026]"     │ ← status: active
│                                              │
│  L3: "User's employment: NewCompany (since   │ ← contradiction resolved
│       Mar 2026). Previously at OldCompany."  │    old archived, new active
│                                              │
│  L2: "Employment: NewCompany [high conf]"    │ ← what read pipeline finds first
│                                              │
│  L1: "Current employer: NewCompany"          │ ← always auto-injected
└──────────────────────────────────────────────┘
         │
  Q: "What is the user's current job?"
         │
  Step 1: L1 auto-inject already has the answer
         │
         ▼
  Answer: "User works at NewCompany"  ← CORRECT
  Score: 1 (LongMemEval knowledge update test)
```

---

### Diagram: Retention Score Decay (why temporal questions work)

```
Fact created Jan 2026. Query asked Apr 2026. λ = 0.01 (L4)

recency = e^(-0.01 × 90 days) = e^(-0.9) = 0.41

Old fact (OldCompany, 90 days ago):
  retention = (0.3 × 0.40) + (0.41 × 0.30) + (freq × 0.20) + (unique × 0.10)
            = 0.12 + 0.12 + ...  ≈  0.35  → LOW → stays deep in L4

New fact (NewCompany, 10 days ago):
  recency = e^(-0.01 × 10) = e^(-0.1) = 0.90
  retention = (0.9 × 0.40) + (0.90 × 0.30) + (freq × 0.20) + (unique × 0.10)
            = 0.36 + 0.27 + ...  ≈  0.72  → HIGH → promoted to L2/L3

Vector similarity for "current job" is SAME for both facts.
Retention score is what separates them. This is the architecture bet.
```

---

### Diagram: LongMemEval Category → CaSVeM Feature Mapping

```
             LongMemEval                        CaSVeM handles it via
             ────────────                       ──────────────────────

  ┌─ Single-hop recall ───────────────────────► L4 extraction quality
  │                                             "did extractor capture this fact?"
  │
  ├─ Multi-hop recall ────────────────────────► L2→L3 pointer traversal
  │                                             "follow source_pointers down 2 layers"
  │
  ├─ Temporal reasoning ──────────────────────► Retention score + recency decay
  │                                             "higher recency = more recent = current"
  │
  ├─ Knowledge update ────────────────────────► CONTRADICTS detection in consolidator
  │                                             "old fact archived, new fact surfaces"
  │
  └─ Absent info ─────────────────────────────► Confidence scoring + L1 boundary
                                                "not in memory = say unknown, don't guess"


Each category is testing a DIFFERENT architectural decision.
If CaSVeM beats Mem0 on temporal and update categories → the novel contributions work.
If CaSVeM ties Mem0 on single-hop → baseline retrieval is at least as good.
```

---

## Step-by-Step: Run LongMemEval Today

### Setup (30 min)

```bash
# 1. Pull the model (if not done yet)
ollama pull qwen3:30b-a3b

# 2. Pull embedding model
ollama pull qwen3-embedding:0.6b

# 3. Install Python deps
pip install datasets huggingface_hub qdrant-client sentence-transformers \
            ollama httpx rich tabulate python-dotenv

# 4. Download LongMemEval
python -c "
from datasets import load_dataset
ds = load_dataset('xiaowu0162/LongMemEval', split='test')
ds.save_to_disk('./benchmark/longmemeval_data')
print(f'Downloaded {len(ds)} test cases')
"

# 5. Download LoCoMo
python -c "
from datasets import load_dataset
ds = load_dataset('snap-research/LoCoMo', split='test')
ds.save_to_disk('./benchmark/locomo_data')
print(f'Downloaded {len(ds)} conversations')
"
```

### Run structure for LongMemEval

```python
# benchmark/run_longmemeval.py  ← what to build

for item in dataset:
    # WRITE: feed all sessions into CaSVeM
    for session in item['sessions']:
        casvem.write(session)           # runs extractor → consolidator → compressor

    # READ: query each question
    for qa in item['questions']:
        answer = casvem.query(qa['question'])   # runs analyser → searcher → synthesiser

        # SCORE: local LLM as judge
        verdict = judge(
            question=qa['question'],
            ground_truth=qa['answer'],
            model_answer=answer
        )
        results.append({
            'category': qa['type'],         # single_hop/multi_hop/temporal/update/absent
            'correct': verdict == 'YES',
            'question': qa['question'],
            'expected': qa['answer'],
            'got': answer
        })

    # Reset memory between items
    casvem.reset()

# Print results by category
print_table(results)
```

### What the output looks like

```
════════════════════════════════════════════════════════
  CaSVeM  vs  Mem0 — LongMemEval Results
════════════════════════════════════════════════════════

  Category             CaSVeM    Mem0     Delta
  ─────────────────    ──────    ──────   ──────
  Single-hop recall    88.0%     93.4%    -5.4%   (room to improve extraction)
  Multi-hop recall     91.0%     89.0%    +2.0%   ← pointer chain working
  Temporal reasoning   87.0%     82.0%    +5.0%   ← retention decay working
  Knowledge update     90.0%     84.0%    +6.0%   ← CONTRADICTS detection working
  Absent info          95.0%     94.0%    +1.0%   ← confidence scoring working

  OVERALL              90.2%     88.5%    +1.7%

  Model used:  qwen3:30b-a3b
  Total time:  4.2 hours
  Cloud cost:  $0.00
════════════════════════════════════════════════════════
```

**The categories to watch:**
- If CaSVeM beats Mem0 on temporal + knowledge update → the novel architecture works
- If single-hop is weak → extraction prompt needs tuning
- Overall score > 91.6% (Mem0's LoCoMo score) → paper-worthy result

---

## Scoring Details

### LongMemEval — LLM Judge (fully local)

```python
JUDGE_PROMPT = """You are evaluating whether an AI answer is correct.

Question: {question}
Ground truth answer: {ground_truth}
AI's answer: {ai_answer}

Is the AI's answer correct or equivalent to the ground truth?
Reply with only one word: YES or NO"""

def judge(question, ground_truth, ai_answer, model="qwen3:30b-a3b"):
    response = ollama.generate(model=model, prompt=JUDGE_PROMPT.format(...))
    return "YES" if "yes" in response.lower() else "NO"
```

### LoCoMo — Token F1 (no LLM needed)

```python
def token_f1(prediction, ground_truth):
    pred_tokens = set(prediction.lower().split())
    truth_tokens = set(ground_truth.lower().split())
    
    common = pred_tokens & truth_tokens
    if not common:
        return 0.0
    
    precision = len(common) / len(pred_tokens)
    recall    = len(common) / len(truth_tokens)
    f1        = 2 * precision * recall / (precision + recall)
    return f1

# Average F1 across all questions = LoCoMo score
# Mem0 target to beat: 91.6
```

---

## The Concept Being Validated

You are not just testing a model. You are testing whether these 4 architectural bets pay off:

```
Bet 1: Layered compression is better than flat storage
       → Multi-hop recall score should exceed flat RAG baseline

Bet 2: Retention score decay surfaces recency correctly
       → Temporal reasoning score should exceed Mem0

Bet 3: CONTRADICTS detection handles knowledge updates better
       → Knowledge update score should exceed Mem0

Bet 4: Confidence scoring prevents hallucination on absent facts
       → Absent info score should stay above 90%

If all 4 bets pay off on LongMemEval → the architecture is correct.
Build the graph layer on top. Submit the paper.
```

---

## Updated Timeline

```
NOW       ollama pull qwen3:30b-a3b  (25 min to download)
          ollama pull qwen3-embedding:0.6b

+30 min   Download benchmark datasets (LongMemEval + LoCoMo)

+30–3h    Build core/ (llm.py, storage.py, write/, read/)
          This is the real CaSVeM implementation

+3h       First LongMemEval run (takes ~3-4 hours to process 500 questions)
          Let it run overnight if needed

Next day  Results. Fix weak categories. Re-run.
+2 days   LoCoMo run

Target    Beat Mem0 on temporal + knowledge update categories
          Match or beat overall score
```

---

## File Layout After This Plan

```
casvem/
├── plan.md                   ← this file
├── casvem_summary.md
├── casvem-diagrams.html
│
├── core/                     ← BUILD THIS FIRST
│   ├── config.py             model = "qwen3:30b-a3b"
│   ├── llm.py                ollama wrapper
│   ├── embedder.py           qwen3-embedding:0.6b
│   ├── storage.py            SQLite + JSON (Weaviate replaces this in v2)
│   ├── write/
│   │   ├── extractor.py      session → L4 facts
│   │   ├── consolidator.py   CONTRADICTS/ADDS/REINFORCES/IRRELEVANT
│   │   └── compressor.py     L3→L2→L1
│   └── read/
│       ├── analyser.py       query → search terms
│       ├── searcher.py       hierarchical vector search
│       └── synthesiser.py    memory block + answer
│
└── benchmark/
    ├── run_longmemeval.py    ← first benchmark
    ├── run_locomo.py         ← second benchmark
    ├── scorer.py             F1 + LLM judge
    └── results/
        ├── longmemeval_YYYY-MM-DD.json
        └── locomo_YYYY-MM-DD.json
```

---

*One model. Two benchmarks. Four architectural bets. Results today.*
