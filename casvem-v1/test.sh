#!/usr/bin/env bash
# CaSVeM End-to-End Test Suite + Benchmark
# Runs all tests, saves responses, then prints a full analysis with timing.

set -euo pipefail

API="http://localhost:8000"
RUN_ID="test_$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="./benchmark/results/$RUN_ID"
mkdir -p "$RESULTS_DIR"

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

pass() { echo -e "${GREEN}  ✓  $1${RESET}"; }
fail() { echo -e "${RED}  ✗  $1${RESET}"; }
info() { echo -e "${CYAN}  →  $1${RESET}"; }
head() { echo -e "\n${BOLD}${YELLOW}══ $1 ══${RESET}"; }

# ── Helper: POST with saved response + latency ───────────────────────────────
post() {
  local label="$1" path="$2" body="$3"
  local file="$RESULTS_DIR/${label}.json"
  local http_code t_start t_end latency
  t_start=$(date +%s%3N)
  http_code=$(curl -s -o "$file" -w "%{http_code}" \
    -X POST "$API$path" \
    -H "Content-Type: application/json" \
    -d "$body" 2>/dev/null)
  t_end=$(date +%s%3N)
  latency=$(( t_end - t_start ))
  # save latency alongside the file
  echo "$latency" > "$RESULTS_DIR/${label}.ms"
  echo "$http_code"
}

# ── Helper: GET with saved response + latency ─────────────────────────────────
get() {
  local label="$1" path="$2"
  local file="$RESULTS_DIR/${label}.json"
  local http_code t_start t_end latency
  t_start=$(date +%s%3N)
  http_code=$(curl -s -o "$file" -w "%{http_code}" "$API$path" 2>/dev/null)
  t_end=$(date +%s%3N)
  latency=$(( t_end - t_start ))
  echo "$latency" > "$RESULTS_DIR/${label}.ms"
  echo "$http_code"
}

# ── Helper: extract JSON field ────────────────────────────────────────────────
jq_get() {
  local file="$RESULTS_DIR/$1.json" key="$2"
  python3 -c "
import json, sys
try:
    d = json.load(open('$file'))
    keys = '$key'.split('.')
    v = d
    for k in keys:
        if k.isdigit(): v = v[int(k)]
        else: v = v[k]
    print(v)
except: print('N/A')
" 2>/dev/null
}

# ── Track results ─────────────────────────────────────────────────────────────
PASS=0; FAIL=0
check() {
  local label="$1" code="$2" expected="$3" desc="$4"
  if [ "$code" = "$expected" ]; then
    pass "$desc (HTTP $code)"
    PASS=$((PASS+1))
  else
    fail "$desc (expected $expected, got $code)"
    FAIL=$((FAIL+1))
  fi
}

SUITE_START=$(date +%s%3N)

# ═════════════════════════════════════════════════════════════════════════════
echo -e "\n${BOLD}${CYAN}  CaSVeM Test Suite + Benchmark  |  $(date)${RESET}"
echo -e "${CYAN}  Run ID: $RUN_ID${RESET}"
echo -e "${CYAN}  Results saved to: $RESULTS_DIR${RESET}"
# ═════════════════════════════════════════════════════════════════════════════

# ── TEST 1: Health check ──────────────────────────────────────────────────────
head "1. HEALTH CHECK"
code=$(get "status" "/status")
check "status" "$code" "200" "GET /status"
weaviate_ok=$(jq_get "status" "weaviate_ok")
ollama_ok=$(jq_get   "status" "ollama_ok")
sched_ok=$(jq_get    "status" "scheduler_running")
info "Weaviate: $weaviate_ok | Ollama: $ollama_ok | Scheduler: $sched_ok"

# ── TEST 2: Write pipeline — Session 1 (work/tech facts) ─────────────────────
head "2. WRITE PIPELINE — SESSION 1 (tech decisions)"
info "Submitting session... (LLM extraction runs in background)"
SESSION1='{
  "session_id": "test_s1",
  "transcript": "I decided to build CaSVeM using FastAPI for the backend. We chose Weaviate as the graph and vector database because it handles both embeddings and graph edges natively. My team lead Sarah approved the architecture on April 28th. The project deadline is end of May 2026. I prefer Python over Go for this project because of the ML ecosystem."
}'
code=$(post "session1" "/session" "$SESSION1")
check "session1" "$code" "200" "POST /session (tech decisions)"
session1_id=$(jq_get "session1" "session_id")
info "Session ID: $session1_id"

# ── TEST 3: Write pipeline — Session 2 (contradicting fact) ──────────────────
head "3. WRITE PIPELINE — SESSION 2 (contradiction test)"
info "Submitting session with a fact that contradicts session 1..."
SESSION2='{
  "session_id": "test_s2",
  "transcript": "After more research I changed my mind — we are switching from Weaviate to Qdrant for the database. The reason is that Qdrant has better documentation and a simpler API. Sarah also agreed with this change."
}'
code=$(post "session2" "/session" "$SESSION2")
check "session2" "$code" "200" "POST /session (contradiction: Weaviate→Qdrant)"

# ── TEST 4: Write pipeline — Session 3 (personal/temporal facts) ─────────────
head "4. WRITE PIPELINE — SESSION 3 (personal facts)"
SESSION3='{
  "session_id": "test_s3",
  "transcript": "My name is Mujahed. I am 24 years old. I am based in Bangalore. I have been coding for 6 years mainly in Python and JavaScript. I am currently working on CaSVeM as a solo project with plans to launch it as open source in June 2026."
}'
code=$(post "session3" "/session" "$SESSION3")
check "session3" "$code" "200" "POST /session (personal facts)"

# ── Wait for write pipeline ────────────────────────────────────────────────────
head "5. WAITING FOR WRITE PIPELINE"
info "Write pipeline runs async — waiting for all 3 sessions to be extracted..."
info "This takes ~2–5 min on CPU (3 sessions × LLM extraction + consolidation)"

L5_BEFORE=$(python3 -c "
import urllib.request, json
try:
    r = urllib.request.urlopen('$API/memory/5', timeout=5)
    d = json.loads(r.read())
    print(d.get('count', 0))
except: print(0)
" 2>/dev/null)
info "L5 count before test sessions: $L5_BEFORE — waiting for 3 new sessions..."

MAX_WAIT=600
INTERVAL=10
elapsed=0
while [ $elapsed -lt $MAX_WAIT ]; do
  l5_count=$(python3 -c "
import urllib.request, json
try:
    r = urllib.request.urlopen('$API/memory/5', timeout=5)
    d = json.loads(r.read())
    print(d.get('count', 0))
except: print(0)
" 2>/dev/null)
  l4_count=$(python3 -c "
import urllib.request, json
try:
    r = urllib.request.urlopen('$API/memory/4', timeout=5)
    d = json.loads(r.read())
    print(d.get('count', 0))
except: print(0)
" 2>/dev/null)
  l5_new=$(( ${l5_count:-0} - ${L5_BEFORE:-0} ))
  if [ "$l5_new" -ge "3" ] 2>/dev/null; then
    pass "All 3 sessions saved (L5=$l5_count, L4=$l4_count facts total) — pipeline complete"
    break
  fi
  echo -ne "\r${CYAN}  →  Waiting... ${elapsed}s elapsed, L5 new: $l5_new/3, L4 facts: $l4_count${RESET}   "
  sleep $INTERVAL
  elapsed=$((elapsed+INTERVAL))
done
if [ "$elapsed" -ge "$MAX_WAIT" ]; then
  fail "Timed out waiting for write pipeline"
fi
echo ""

WRITE_ELAPSED=$elapsed

# ── TEST 5: Memory inspection — all layers ────────────────────────────────────
head "6. MEMORY INSPECTION — ALL LAYERS"
for layer in 1 2 3 4 5; do
  code=$(get "memory_l${layer}" "/memory/${layer}")
  check "memory_l${layer}" "$code" "200" "GET /memory/${layer}"
  count=$(jq_get "memory_l${layer}" "count")
  info "L${layer}: $count nodes"
done

# ── TEST 6: Query benchmark ────────────────────────────────────────────────────
head "7. READ PIPELINE — QUERY BENCHMARK"

run_query() {
  local num="$1" label="$2" qtext="$3"
  info "Q${num}: $label"
  local t_start t_end latency
  t_start=$(date +%s%3N)
  code=$(post "query${num}" "/query" "{\"query\": \"$qtext\"}")
  t_end=$(date +%s%3N)
  latency=$(( t_end - t_start ))
  check "query${num}" "$code" "200" "POST /query ($label)"
  local answer conf layers
  answer=$(jq_get "query${num}" "answer")
  conf=$(jq_get   "query${num}" "confidence")
  layers=$(jq_get "query${num}" "layers_hit")
  info "Answer:  $answer"
  info "Conf: $conf  |  Layers: $layers  |  Latency: ${latency}ms"
}

run_query 1 "simple fact recall"       "What backend framework am I using for CaSVeM?"
run_query 2 "contradiction detection"  "Which database are we using for the project?"
run_query 3 "personal fact recall"     "Where is the developer based and how old are they?"
run_query 4 "temporal/deadline"        "When is the CaSVeM project deadline?"
run_query 5 "absent info (hallucination guard)" "What is my salary?"

# ── TEST 7: Admin endpoints ────────────────────────────────────────────────────
head "8. ADMIN ENDPOINTS"
code=$(post "admin_consolidate" "/admin/consolidate" '{}')
check "admin_consolidate" "$code" "200" "POST /admin/consolidate"
code=$(post "admin_promote"    "/admin/promote"     '{}')
check "admin_promote"    "$code" "200" "POST /admin/promote"

# ── Final status check ─────────────────────────────────────────────────────────
head "9. FINAL STATUS"
code=$(get "status_final" "/status")
check "status_final" "$code" "200" "GET /status (final)"

SUITE_END=$(date +%s%3N)
SUITE_TOTAL=$(( SUITE_END - SUITE_START ))

# ═════════════════════════════════════════════════════════════════════════════
# ANALYSIS + BENCHMARK REPORT
# ═════════════════════════════════════════════════════════════════════════════
head "ANALYSIS + BENCHMARK"

python3 - "$RESULTS_DIR" "$RUN_ID" "$SUITE_TOTAL" "$WRITE_ELAPSED" <<'PYEOF'
import json, os, sys
from pathlib import Path

d         = Path(sys.argv[1])
run_id    = sys.argv[2]
suite_ms  = int(sys.argv[3])
write_sec = int(sys.argv[4])

def load(name):
    f = d / f"{name}.json"
    if not f.exists(): return {}
    try: return json.loads(f.read_text())
    except: return {}

def ms(name):
    f = d / f"{name}.ms"
    if not f.exists(): return 0
    try: return int(f.read_text().strip())
    except: return 0

def yn(v): return "✓" if str(v).lower() in ("true","1","yes") else "✗"

print()
print("┌──────────────────────────────────────────────────────────────┐")
print("│                 CaSVeM Test + Benchmark Report               │")
print(f"│  Run: {run_id:<55}│")
print("├──────────────────────────────────────────────────────────────┤")

# Infrastructure
status = load("status_final")
print(f"│  Weaviate running   {yn(status.get('weaviate_ok'))}                                       │")
print(f"│  Ollama running     {yn(status.get('ollama_ok'))}                                       │")
print(f"│  Scheduler running  {yn(status.get('scheduler_running'))}                                       │")
counts = status.get("layer_counts", {})
print(f"│  Layer counts  L1:{counts.get('L1',0):>4}  L2:{counts.get('L2',0):>4}  L3:{counts.get('L3',0):>4}  L4:{counts.get('L4',0):>4}  L5:{counts.get('L5',0):>4}       │")
print("├──────────────────────────────────────────────────────────────┤")

# Query benchmark table
queries = [
    ("query1", "Simple recall      (FastAPI backend)"),
    ("query2", "Contradiction      (Weaviate→Qdrant)"),
    ("query3", "Personal fact      (location/age)"),
    ("query4", "Temporal fact      (deadline)"),
    ("query5", "Absent info        (salary)"),
]

print("│  QUERY RESULTS                                               │")
latencies = []
for fname, label in queries:
    q       = load(fname)
    lat     = ms(fname)
    latencies.append(lat)
    answer  = str(q.get("answer", "no response"))[:46]
    conf    = q.get("confidence", "?")
    layers  = q.get("layers_hit", [])
    routed  = q.get("routed_to", "?")
    print(f"│                                                              │")
    print(f"│  {label:<60}│")
    print(f"│    Answer:  {answer:<48}  │")
    print(f"│    Conf: {str(conf):<8} Layers: {str(layers):<12} Latency: {lat:>5}ms       │")

print("├──────────────────────────────────────────────────────────────┤")

# Architecture validation
q2 = load("query2")
q2_ans = q2.get("answer","").lower()
qdrant_found  = "qdrant" in q2_ans
weaviate_only = "weaviate" in q2_ans and "qdrant" not in q2_ans

q5 = load("query5")
q5_ans = q5.get("answer","").lower()
no_hallucination = any(x in q5_ans for x in ["don't","do not","no information","not","unknown","n/a","i don"])

q3 = load("query3")
q3_ans = q3.get("answer","").lower()
personal_ok = "bangalore" in q3_ans or "24" in q3_ans

l4 = load("memory_l4")
facts_extracted = l4.get("count", 0)
pipeline_ok = facts_extracted > 0

print("│  ARCHITECTURE VALIDATION                                     │")
print(f"│  Contradiction detection  {'✓ returned Qdrant (correct)' if qdrant_found else '✗ returned stale Weaviate':<38}│")
print(f"│  Hallucination guard      {'✓ refused to guess' if no_hallucination else '✗ may have hallucinated':<38}│")
print(f"│  Personal fact recall     {'✓ Bangalore/24 found' if personal_ok else '✗ personal facts missing':<38}│")
print(f"│  Write pipeline           {'✓' if pipeline_ok else '✗'} {facts_extracted} facts in L4{'':<29}│")
print("├──────────────────────────────────────────────────────────────┤")

# Benchmark numbers
avg_lat  = int(sum(latencies) / len(latencies)) if latencies else 0
min_lat  = min(latencies) if latencies else 0
max_lat  = max(latencies) if latencies else 0
p95_lat  = sorted(latencies)[int(len(latencies)*0.95)-1] if latencies else 0
print("│  BENCHMARK                                                   │")
print(f"│  Write pipeline time      {write_sec:>4}s                                   │")
print(f"│  Query latency avg        {avg_lat:>5}ms                                  │")
print(f"│  Query latency min        {min_lat:>5}ms                                  │")
print(f"│  Query latency max        {max_lat:>5}ms                                  │")
print(f"│  Total suite time         {suite_ms//1000:>4}s                                   │")
print("└──────────────────────────────────────────────────────────────┘")
print()

# Verdict
if pipeline_ok and qdrant_found and no_hallucination and personal_ok:
    print("  VERDICT: ✓ ALL CHECKS PASSED — Core architecture working correctly.")
    print("  Contradiction ✓  Hallucination guard ✓  Write pipeline ✓  Personal facts ✓")
elif pipeline_ok and qdrant_found and no_hallucination:
    print("  VERDICT: MOSTLY PASSING — personal fact recall needs attention.")
elif pipeline_ok:
    print("  VERDICT: Write pipeline works. Read pipeline needs tuning.")
else:
    print("  VERDICT: Write pipeline still processing — wait 2–3 min and re-run.")
print()

# Save machine-readable summary alongside results
summary = {
    "run_id": run_id,
    "layer_counts": counts,
    "write_pipeline_sec": write_sec,
    "query_latency_ms": {
        "avg": avg_lat, "min": min_lat, "max": max_lat,
        "per_query": dict(zip([q[0] for q in queries], latencies)),
    },
    "validation": {
        "contradiction_detection": qdrant_found,
        "hallucination_guard": no_hallucination,
        "personal_fact_recall": personal_ok,
        "write_pipeline_ok": pipeline_ok,
    },
    "suite_total_ms": suite_ms,
}
import json as _json
(Path(sys.argv[1]) / "summary.json").write_text(_json.dumps(summary, indent=2))
print(f"  Machine-readable summary saved to: {sys.argv[1]}/summary.json")
PYEOF

# ═════════════════════════════════════════════════════════════════════════════
echo ""
echo -e "${BOLD}  Tests passed: ${GREEN}$PASS${RESET}  ${BOLD}Failed: ${RED}$FAIL${RESET}  ${BOLD}Total time: $((SUITE_TOTAL/1000))s${RESET}"
echo -e "  Full responses saved to: ${CYAN}$RESULTS_DIR/${RESET}"
echo ""
