#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'

PASSED=0; FAILED=0
fail() { echo -e "${RED}  FAIL: $1${NC}"; FAILED=$((FAILED+1)); }
pass() { echo -e "${GREEN}  PASS: $1${NC}"; PASSED=$((PASSED+1)); }

echo -e "${CYAN}CaSVeM v3 — test suite${NC}"
echo "────────────────────────────────"

# ── Venv check ────────────────────────────────────────────────────────────────
if [ ! -d venv ]; then
    echo "No venv found. Run ./run.sh first to set up the environment."
    exit 1
fi

# ── PART 1: Unit tests (no API key needed) ────────────────────────────────────
echo ""
echo -e "${CYAN}Part 1 — Unit tests (no API key required)${NC}"
echo "────────────────────────────────"

mkdir -p data
# Unit tests use an isolated DB so they never conflict with a running server
SQLITE_PATH=data/test_unit.db HNSW_INDEX_PATH=data/test_unit.usearch \
venv/bin/python3 -c "
import sys, os
sys.path.insert(0, '.')
os.environ.setdefault('GEMINI_API_KEY', 'test-placeholder')
os.environ['SQLITE_PATH'] = 'data/test_unit.db'
os.environ['HNSW_INDEX_PATH'] = 'data/test_unit.usearch'
import numpy as np, time

errors = []

# 1 config
try:
    from config import cfg
    assert cfg.llm_backend == 'gemini'
    assert cfg.encoder_model == 'all-MiniLM-L6-v2'
    print('  [1/10] config                  PASS')
except Exception as e:
    errors.append(('config', str(e)))
    print(f'  [1/10] config                  FAIL: {e}')

# 2 encoder
try:
    from core.encoder import get_encoder
    enc = get_encoder()
    vec = enc.encode('test sentence')
    assert vec.shape == (384,)
    assert abs(np.linalg.norm(vec) - 1.0) < 1e-5
    print('  [2/10] encoder                 PASS  dim=384 norm=1.0')
except Exception as e:
    errors.append(('encoder', str(e)))
    print(f'  [2/10] encoder                 FAIL: {e}')

# 3 storage write + read
try:
    from core.encoder import get_encoder
    from core.storage import get_storage
    enc = get_encoder()
    st = get_storage()
    vec = enc.encode('User lives in Bangalore')
    mid = st.add_memory('User lives in Bangalore', vec, memory_type='fact')
    mem = st.get_memory(mid)
    assert mem['text'] == 'User lives in Bangalore'
    print('  [3/10] storage write+read      PASS')
except Exception as e:
    errors.append(('storage', str(e)))
    print(f'  [3/10] storage write+read      FAIL: {e}')

# 4 HNSW search returns correct memory
try:
    from core.encoder import get_encoder
    from core.storage import get_storage
    enc = get_encoder()
    st = get_storage()
    vec = enc.encode('User lives in Bangalore')
    results = st.search_hnsw(vec, k=5)
    assert len(results) >= 1
    scores = [s for _, s in results]
    assert max(scores) > 0.9, f'Expected high score, got {max(scores)}'
    print(f'  [4/10] HNSW search             PASS  top_score={max(scores):.4f}')
except Exception as e:
    errors.append(('hnsw', str(e)))
    print(f'  [4/10] HNSW search             FAIL: {e}')

# 5 bitmap filter
try:
    from core.retrieval.bitmap_filter import BitmapFilter
    bf = BitmapFilter()
    bf.add(0, memory_type='fact', project_id='p1')
    bf.add(1, memory_type='decision', project_id='p1')
    bf.add(2, memory_type='fact', project_id='p2')
    result = bf.filter(memory_type='fact', project_id='p1')
    assert result == {0}, f'Expected {{0}} got {result}'
    result2 = bf.filter()
    assert result2 is None  # no constraints → search all
    print('  [5/10] bitmap filter           PASS')
except Exception as e:
    errors.append(('bitmap', str(e)))
    print(f'  [5/10] bitmap filter           FAIL: {e}')

# 6 LRU cache write + hit
try:
    from core.cache.lru_cache import LRUCacheGate, make_exact_key
    import numpy as np
    gate = LRUCacheGate()
    vec = np.random.rand(384).astype(np.float32)
    vec /= np.linalg.norm(vec)
    key = make_exact_key(vec)
    gate.write(key, ['mem-id-1', 'mem-id-2'], tier='L2')
    tier, found_key = gate.check(vec)
    assert tier == 'L2', f'Expected L2 hit, got {tier}'
    ids = gate.get_memory_ids(found_key)
    assert ids == ['mem-id-1', 'mem-id-2']
    print('  [6/10] LRU cache gate          PASS')
except Exception as e:
    errors.append(('lru', str(e)))
    print(f'  [6/10] LRU cache gate          FAIL: {e}')

# 7 collision detection (Opt 1)
try:
    from core.cache.lru_cache import LRUCacheGate
    import numpy as np
    gate = LRUCacheGate()
    v1 = np.random.rand(384).astype(np.float32); v1 /= np.linalg.norm(v1)
    v2 = np.random.rand(384).astype(np.float32); v2 /= np.linalg.norm(v2)
    assert gate.verify_hit(v1, v1.tobytes()), 'Same vector should pass'
    assert not gate.verify_hit(v1, v2.tobytes()), 'Random vector should fail'
    print('  [7/10] collision detection     PASS')
except Exception as e:
    errors.append(('collision', str(e)))
    print(f'  [7/10] collision detection     FAIL: {e}')

# 8 reranker
try:
    from core.retrieval.reranker import get_reranker
    rr = get_reranker()
    mems = [
        {'id': '1', 'text': 'User uses Python for all ML work', 'created_at': 0},
        {'id': '2', 'text': 'User had pizza for lunch yesterday', 'created_at': 0},
    ]
    ranked = rr.rerank('What language does the user code in?', mems, top_n=2)
    assert ranked[0]['id'] == '1', f'Python memory should rank first, got id={ranked[0][\"id\"]}'
    print(f'  [8/10] reranker                PASS  correct order, top_score={ranked[0][\"rerank_score\"]:.3f}')
except Exception as e:
    errors.append(('reranker', str(e)))
    print(f'  [8/10] reranker                FAIL: {e}')

# 9 context builder
try:
    from core.context.builder import build_context, build_prompt
    mems = [{'id': '1', 'text': 'User is based in Bangalore', 'rerank_score': 0.9, 'created_at': 0}]
    ctx = build_context('Where does user live?', mems)
    assert '[Memory]' in ctx
    assert 'Bangalore' in ctx
    prompt = build_prompt('Where does user live?', ctx)
    assert 'Question:' in prompt
    print('  [9/10] context builder         PASS')
except Exception as e:
    errors.append(('context', str(e)))
    print(f'  [9/10] context builder         FAIL: {e}')

# 10 ingest pipeline (full write path)
try:
    from pipeline.ingest import ingest
    mid = ingest('User prefers dark mode in all editors', memory_type='preference', project_id='settings')
    assert mid and len(mid) == 36  # UUID format
    print('  [10/10] ingest pipeline        PASS  id=' + mid[:12] + '...')
except Exception as e:
    errors.append(('ingest', str(e)))
    print(f'  [10/10] ingest pipeline        FAIL: {e}')

total = 10
passed = total - len(errors)
print()
if errors:
    print(f'  {passed}/{total} passed  |  {len(errors)} failed: {[e[0] for e in errors]}')
    sys.exit(1)
else:
    print(f'  {passed}/{total} passed')
    sys.exit(0)
" && pass "Unit tests (10/10)" || fail "Unit tests"
rm -f data/test_unit.db data/test_unit.usearch

# ── PART 2: Live API test (needs GEMINI_API_KEY + server running) ─────────────
echo ""
echo -e "${CYAN}Part 2 — Live API test (needs server running + GEMINI_API_KEY)${NC}"
echo "────────────────────────────────"

source .env 2>/dev/null || true

if [ -z "$GEMINI_API_KEY" ] || [ "$GEMINI_API_KEY" = "your_gemini_api_key_here" ]; then
    echo -e "${YELLOW}  Skipped — GEMINI_API_KEY not set in .env${NC}"
else
    # Wait up to 10s for server to be ready (it may be reloading after file changes)
    SERVER_READY=0
    for _i in 1 2 3 4 5; do
        if curl -sf http://localhost:8000/stats > /dev/null 2>&1; then
            SERVER_READY=1; break
        fi
        sleep 2
    done
    if [ "$SERVER_READY" = "0" ]; then
        echo -e "${YELLOW}  Skipped — server not running. Start with ./run.sh first.${NC}"
    else
        echo "  Server is up. Running live tests..."

        # Test 1: ingest a memory
        INGEST_RESP=$(curl -sf -X POST http://localhost:8000/memory \
            -H "Content-Type: application/json" \
            -d '{"text":"The user is a solo developer building CaSVeM as a research project","memory_type":"fact"}')
        MEM_ID=$(echo "$INGEST_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])" 2>/dev/null)
        if [ -n "$MEM_ID" ]; then
            pass "POST /memory  → id=${MEM_ID:0:12}..."
        else
            fail "POST /memory  → unexpected response: $INGEST_RESP"
        fi

        # Test 2: cold query (cache miss, hits Gemini)
        QUERY_RESP=$(curl -sf -X POST http://localhost:8000/query \
            -H "Content-Type: application/json" \
            -d '{"text":"What is the user working on?"}')
        HIT_TYPE=$(echo "$QUERY_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['hit_type'])" 2>/dev/null)
        LATENCY=$(echo "$QUERY_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['latency_ms'])" 2>/dev/null)
        if [ "$HIT_TYPE" = "cold" ]; then
            pass "POST /query (cold)  → hit_type=cold  latency=${LATENCY}ms"
        else
            fail "POST /query (cold)  → expected cold got '$HIT_TYPE'"
        fi

        # Test 3: same query again (should be cache hit)
        sleep 0.5
        QUERY_RESP2=$(curl -sf -X POST http://localhost:8000/query \
            -H "Content-Type: application/json" \
            -d '{"text":"What is the user working on?"}')
        HIT_TYPE2=$(echo "$QUERY_RESP2" | python3 -c "import sys,json; print(json.load(sys.stdin)['hit_type'])" 2>/dev/null)
        LATENCY2=$(echo "$QUERY_RESP2" | python3 -c "import sys,json; print(json.load(sys.stdin)['latency_ms'])" 2>/dev/null)
        if [ "$HIT_TYPE2" = "L2" ] || [ "$HIT_TYPE2" = "L1" ]; then
            pass "POST /query (cached)  → hit_type=${HIT_TYPE2}  latency=${LATENCY2}ms"
        else
            fail "POST /query (cached)  → expected L1/L2 hit, got '$HIT_TYPE2'"
        fi

        # Test 4: GET /stats
        STATS_RESP=$(curl -sf http://localhost:8000/stats)
        HIT_RATE=$(echo "$STATS_RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"{d['hit_rate']*100:.0f}%\")" 2>/dev/null)
        TOTAL=$(echo "$STATS_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['total_queries'])" 2>/dev/null)
        if [ -n "$HIT_RATE" ]; then
            pass "GET /stats  → total=${TOTAL}  hit_rate=${HIT_RATE}"
        else
            fail "GET /stats  → unexpected response"
        fi

        # Test 5: DELETE /memory
        DEL_RESP=$(curl -sf -X DELETE "http://localhost:8000/memory/${MEM_ID}")
        DEL_OK=$(echo "$DEL_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('ok',''))" 2>/dev/null)
        if [ "$DEL_OK" = "True" ]; then
            pass "DELETE /memory/${MEM_ID:0:8}...  → ok"
        else
            fail "DELETE /memory  → unexpected: $DEL_RESP"
        fi
    fi
fi

# ── PART 3: Dataset Benchmarks (needs server running + GEMINI_API_KEY) ────────
echo ""
echo -e "${CYAN}Part 3 — Dataset Benchmarks (needs server running + GEMINI_API_KEY)${NC}"
echo "────────────────────────────────"

source .env 2>/dev/null || true

BENCH_SKIPPED=0
if [ -z "$GEMINI_API_KEY" ] || [ "$GEMINI_API_KEY" = "your_gemini_api_key_here" ]; then
    echo -e "${YELLOW}  Skipped — GEMINI_API_KEY not set in .env${NC}"
    BENCH_SKIPPED=1
else
    # Wait up to 10s for server (may be reloading)
    BENCH_READY=0
    for _i in 1 2 3 4 5; do
        if curl -sf http://localhost:8000/stats > /dev/null 2>&1; then
            BENCH_READY=1; break
        fi
        sleep 2
    done
    if [ "$BENCH_READY" = "0" ]; then
        echo -e "${YELLOW}  Skipped — server not running. Start with ./run.sh first.${NC}"
        BENCH_SKIPPED=1
    fi
fi

if [ "$BENCH_SKIPPED" = "0" ]; then
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M')
    RESULT_FILE="benchmark/result.md"
    mkdir -p benchmark/results

    # ── Synthetic benchmark (fast, no LLM judge, shows cache warmup) ─────────
    echo ""
    echo "  Running Synthetic benchmark (20 memories, 25 queries, keyword match)..."
    SYNTH_OUT=$(venv/bin/python3 benchmark/run_synthetic.py 2>&1)
    echo "$SYNTH_OUT"
    if echo "$SYNTH_OUT" | grep -q "CaSVeM Synthetic Benchmark — Results"; then
        pass "Synthetic benchmark completed"
    else
        fail "Synthetic benchmark"
    fi

    # ── LoCoMo (no LLM judge, fast) ───────────────────────────────────────────
    echo ""
    echo "  Running LoCoMo (local, 3 records × 5 QA, Token F1)..."
    LOCOMO_OUT=$(venv/bin/python3 benchmark/run_locomo_local.py --limit 3 --qa-per-record 5 2>&1)
    echo "$LOCOMO_OUT"
    if echo "$LOCOMO_OUT" | grep -q "LoCoMo (local) Results"; then
        pass "LoCoMo benchmark completed"
    else
        fail "LoCoMo benchmark"
    fi

    # ── BEAM (kv_retrieval + longdialogue, no LLM judge) ──────────────────────
    echo ""
    echo "  Running BEAM (local, kv×5 + dlg×3, exact/substr match)..."
    BEAM_OUT=$(venv/bin/python3 benchmark/run_beam_local.py --kv-limit 5 --dlg-limit 3 2>&1)
    echo "$BEAM_OUT"
    if echo "$BEAM_OUT" | grep -q "BEAM Summary"; then
        pass "BEAM benchmark completed"
    else
        fail "BEAM benchmark"
    fi

    # ── LongMemEval (LLM judge — costs API tokens, temporal fix applied) ──────
    echo ""
    echo "  Running LongMemEval (local, 5 records, LLM judge, date-tagged sessions)..."
    LME_OUT=$(venv/bin/python3 benchmark/run_longmemeval_local.py --limit 5 2>&1)
    echo "$LME_OUT"
    if echo "$LME_OUT" | grep -q "LongMemEval (local) Results"; then
        pass "LongMemEval benchmark completed"
    else
        fail "LongMemEval benchmark"
    fi

    # ── Write result.md ───────────────────────────────────────────────────────
    venv/bin/python3 benchmark/write_result_md.py

    if [ -f "benchmark/result.md" ]; then
        pass "benchmark/result.md written"
    else
        fail "benchmark/result.md not created"
    fi
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "────────────────────────────────"
TOTAL=$((PASSED+FAILED))
if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}All ${TOTAL} tests passed${NC}"
    exit 0
else
    echo -e "${RED}${FAILED}/${TOTAL} tests failed${NC}"
    exit 1
fi
