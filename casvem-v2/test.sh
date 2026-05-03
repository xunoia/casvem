#!/usr/bin/env bash
# CaSVeM End-to-End Test Suite
#
# Usage:
#   bash test.sh                   # run on port 8001 (default)
#   API=http://localhost:8000 bash test.sh
#
# Before running, start the server:
#   .venv/bin/uvicorn main:app --port 8001

set -euo pipefail

API="${API:-http://localhost:8001}"
RESULTS_DIR="./benchmark/results/test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

pass() { echo -e "${GREEN}  ✓  $1${RESET}"; }
fail() { echo -e "${RED}  ✗  $1${RESET}"; }
info() { echo -e "${CYAN}  →  $1${RESET}"; }
hdr()  { echo -e "\n${BOLD}${YELLOW}══ $1 ══${RESET}"; }

# ── Helper: POST ──────────────────────────────────────────────────────────────
post() {
  local label="$1" path="$2" body="$3"
  local file="$RESULTS_DIR/${label}.json"
  local code
  code=$(curl -s -o "$file" -w "%{http_code}" \
    -X POST "$API$path" \
    -H "Content-Type: application/json" \
    -d "$body" 2>/dev/null)
  echo "$code"
}

# ── Helper: GET ───────────────────────────────────────────────────────────────
get() {
  local label="$1" path="$2"
  local file="$RESULTS_DIR/${label}.json"
  local code
  code=$(curl -s -o "$file" -w "%{http_code}" "$API$path" 2>/dev/null)
  echo "$code"
}

# ── Helper: extract a JSON field ──────────────────────────────────────────────
jqg() {
  local file="$RESULTS_DIR/$1.json" key="$2"
  python3 -c "
import json, sys
try:
    d = json.load(open('$file'))
    keys = '$key'.split('.')
    v = d
    for k in keys:
        v = v[int(k)] if k.isdigit() else v[k]
    print(v)
except: print('N/A')
" 2>/dev/null
}

# ── Track pass/fail ───────────────────────────────────────────────────────────
PASS=0; FAIL=0
check() {
  local label="$1" code="$2" expected="$3" desc="$4"
  if [ "$code" = "$expected" ]; then
    pass "$desc (HTTP $code)"; PASS=$((PASS+1))
  else
    fail "$desc (expected $expected, got $code)"; FAIL=$((FAIL+1))
  fi
}

# =============================================================================
echo -e "\n${BOLD}${CYAN}  CaSVeM E2E Test Suite  |  $(date)${RESET}"
echo -e "${CYAN}  API:     $API${RESET}"
echo -e "${CYAN}  Results: $RESULTS_DIR${RESET}"
# =============================================================================

# ── TEST 0: Server health ─────────────────────────────────────────────────────
hdr "0. SERVER HEALTH"
code=$(get "status_pre" "/status")
check "status_pre" "$code" "200" "GET /status"
vs_ok=$(jqg  "status_pre" "vector_store_ok")
ol_ok=$(jqg  "status_pre" "ollama_ok")
info "VectorStore: $vs_ok  |  Ollama: $ol_ok"
if [ "$vs_ok" != "True" ] || [ "$ol_ok" != "True" ]; then
  echo -e "${RED}  ✗  Required backends not healthy — aborting.${RESET}"
  exit 1
fi

# ── TEST 1: Reset ─────────────────────────────────────────────────────────────
hdr "1. RESET (clean slate)"
code=$(post "reset" "/admin/reset" '{}')
check "reset" "$code" "200" "POST /admin/reset"
info "Pinecone index cleared"

# ── TEST 2: Write — Session 1 (tech/project facts) ───────────────────────────
hdr "2. WRITE — SESSION 1  (tech decisions)"
SESSION1='{
  "session_id": "test_s1",
  "transcript": "I decided to build CaSVeM using FastAPI for the backend. We chose Weaviate as the graph and vector database because it handles both embeddings and graph edges natively. My team lead Sarah approved the architecture on April 28th. The project deadline is end of May 2026. I prefer Python over Go for this project because of the ML ecosystem."
}'
code=$(post "session1" "/session" "$SESSION1")
check "session1" "$code" "200" "POST /session (tech facts)"
info "Session ID: $(jqg session1 session_id)"

# ── TEST 3: Write — Session 2 (contradiction) ────────────────────────────────
hdr "3. WRITE — SESSION 2  (contradiction: Weaviate → Qdrant)"
SESSION2='{
  "session_id": "test_s2",
  "transcript": "After more research I changed my mind — we are switching from Weaviate to Qdrant for the database. The reason is that Qdrant has better documentation and a simpler API. Sarah also agreed with this change."
}'
code=$(post "session2" "/session" "$SESSION2")
check "session2" "$code" "200" "POST /session (contradiction)"

# ── TEST 4: Write — Session 3 (personal facts) ───────────────────────────────
hdr "4. WRITE — SESSION 3  (personal facts)"
SESSION3='{
  "session_id": "test_s3",
  "transcript": "My name is Mujahed. I am 24 years old and based in Bangalore. I have been coding for 6 years mainly in Python and JavaScript. I am currently working on CaSVeM as a solo project with plans to launch it as open source in June 2026."
}'
code=$(post "session3" "/session" "$SESSION3")
check "session3" "$code" "200" "POST /session (personal facts)"

# ── Wait for write pipeline (extractor + consolidator) ────────────────────────
hdr "5. WAITING FOR WRITE PIPELINE (LLM extraction in background)"
info "Polling until L4 facts appear ..."
MAX_WAIT=300; INTERVAL=5; elapsed=0
while [ $elapsed -lt $MAX_WAIT ]; do
  l4=$(python3 -c "
import urllib.request, json
try:
    r=urllib.request.urlopen('$API/memory/4',timeout=5)
    print(json.loads(r.read()).get('count',0))
except: print(0)" 2>/dev/null)
  [ "$l4" -gt "0" ] 2>/dev/null && pass "L4 has $l4 facts — write complete" && break
  echo -ne "\r${CYAN}  →  ${elapsed}s  L4=${l4}${RESET}   "
  sleep $INTERVAL; elapsed=$((elapsed+INTERVAL))
done
echo ""

# ── TEST 5: Consolidate L3 → L2 → L1 ─────────────────────────────────────────
hdr "6. CONSOLIDATE  (L3 → L2 → L1)"
code=$(post "consolidate" "/admin/consolidate" '{}')
check "consolidate" "$code" "200" "POST /admin/consolidate  (L3→L2)"

# Wait for L2 to appear in Pinecone before promoting
info "Waiting for L2 nodes to become queryable ..."
for i in $(seq 1 30); do
  l2=$(python3 -c "
import urllib.request, json
try:
    r=urllib.request.urlopen('$API/memory/2',timeout=5)
    print(json.loads(r.read()).get('count',0))
except: print(0)" 2>/dev/null)
  [ "$l2" -gt "0" ] 2>/dev/null && info "L2 ready ($l2 nodes)" && break
  sleep 2
done

code=$(post "promote" "/admin/promote" '{}')
check "promote" "$code" "200" "POST /admin/promote      (L2→L1)"

# Wait for L1 to appear before querying
info "Waiting for L1 nodes to become queryable ..."
for i in $(seq 1 30); do
  l1=$(python3 -c "
import urllib.request, json
try:
    r=urllib.request.urlopen('$API/memory/1',timeout=5)
    print(json.loads(r.read()).get('count',0))
except: print(0)" 2>/dev/null)
  [ "$l1" -gt "0" ] 2>/dev/null && info "L1 ready ($l1 nodes)" && break
  sleep 2
done

# ── TEST 6: Memory snapshot (all layers) ──────────────────────────────────────
hdr "7. MEMORY SNAPSHOT"
for layer in 1 2 3 4 5; do
  code=$(get "mem_l${layer}" "/memory/${layer}")
  check "mem_l${layer}" "$code" "200" "GET /memory/${layer}"
  cnt=$(jqg "mem_l${layer}" "count")
  info "L${layer}: $cnt nodes"
done

# ── TEST 7: Query tests ────────────────────────────────────────────────────────
hdr "8. READ PIPELINE — QUERY TESTS"

run_query() {
  local label="$1" qtext="$2" desc="$3"
  code=$(post "$label" "/query" "{\"query\":\"$qtext\"}")
  check "$label" "$code" "200" "POST /query — $desc"
  ans=$(jqg "$label" "answer")
  conf=$(jqg "$label" "confidence")
  layers=$(jqg "$label" "layers_hit")
  info "Q: $qtext"
  info "A: $ans"
  info "Confidence: $conf  |  Layers hit: $layers"
  echo ""
}

run_query "q1" "What backend framework are we using for CaSVeM?" "simple fact recall"
run_query "q2" "Which database are we using for the project?"    "contradiction recall (Weaviate→Qdrant)"
run_query "q3" "Where is Mujahed based and how old is he?"       "personal fact recall"
run_query "q4" "When is the CaSVeM project deadline?"            "temporal fact recall"
run_query "q5" "What is my salary?"                              "absent info (hallucination guard)"
run_query "q6" "Who approved the CaSVeM architecture?"           "named entity recall"
run_query "q7" "What programming languages does Mujahed use?"    "multi-fact recall"

# ── TEST 8: Final status ───────────────────────────────────────────────────────
hdr "9. FINAL STATUS"
code=$(get "status_post" "/status")
check "status_post" "$code" "200" "GET /status (final)"

# =============================================================================
# ANALYSIS
# =============================================================================
hdr "ANALYSIS"

python3 - "$RESULTS_DIR" <<'PYEOF'
import json, sys
from pathlib import Path

d = Path(sys.argv[1])

def load(name):
    f = d / f"{name}.json"
    if not f.exists(): return {}
    try: return json.loads(f.read_text())
    except: return {}

def yn(v): return "✓" if str(v).lower() in ("true","1","yes") else "✗"

status  = load("status_post")
counts  = status.get("layer_counts", {})

print()
print("┌─────────────────────────────────────────────────────────────┐")
print("│              CaSVeM v2  ·  Test Results                    │")
print("├─────────────────────────────────────────────────────────────┤")
print(f"│  VectorStore  {yn(status.get('vector_store_ok'))}   Ollama  {yn(status.get('ollama_ok'))}   Scheduler  {yn(status.get('scheduler_running'))}          │")
print(f"│  Layer counts   L1:{counts.get('L1',0):>4}  L2:{counts.get('L2',0):>4}  L3:{counts.get('L3',0):>4}  L4:{counts.get('L4',0):>4}  L5:{counts.get('L5',0):>4}   │")
print("├─────────────────────────────────────────────────────────────┤")
print("│  QUERY RESULTS                                              │")

queries = [
    ("q1", "Simple recall    (FastAPI backend)  "),
    ("q2", "Contradiction    (Weaviate → Qdrant)"),
    ("q3", "Personal fact    (location / age)   "),
    ("q4", "Temporal fact    (deadline)         "),
    ("q5", "Absent info      (salary)           "),
    ("q6", "Named entity     (who approved?)    "),
    ("q7", "Multi-fact       (languages)        "),
]

for fname, label in queries:
    q      = load(fname)
    answer = str(q.get("answer","—"))[:46]
    conf   = q.get("confidence","?")
    layers = q.get("layers_hit", [])
    print(f"│                                                             │")
    print(f"│  {label}                │")
    print(f"│    → {answer:<52}  │")
    print(f"│    Conf: {conf:<8} Layers: {str(layers):<12}                  │")

print("├─────────────────────────────────────────────────────────────┤")

# Key architecture checks
q2_ans = load("q2").get("answer","").lower()
q5_ans = load("q5").get("answer","").lower()
l4_n   = load("status_post").get("layer_counts",{}).get("L4",0)

contradiction_ok  = "qdrant" in q2_ans and "weaviate" not in q2_ans.replace("from weaviate","")
no_hallucination  = any(x in q5_ans for x in ["don't","do not","no information","unknown","not ","n/a","i don"])
pipeline_ok       = l4_n > 0

print("│  ARCHITECTURE VALIDATION                                    │")
print(f"│  Write pipeline         {'✓' if pipeline_ok else '✗'} {l4_n} facts extracted into L4{'':>14}│")
print(f"│  Contradiction detect   {'✓ Qdrant surfaced correctly' if contradiction_ok else '✗ stale / wrong answer':<36} │")
print(f"│  Hallucination guard    {'✓ refused to guess salary' if no_hallucination else '✗ may have hallucinated':<36} │")
print("└─────────────────────────────────────────────────────────────┘")
print()

if pipeline_ok and contradiction_ok and no_hallucination:
    print("  VERDICT ✓  All core architectural bets validated.")
elif pipeline_ok:
    print("  VERDICT ~  Write pipeline OK. Read pipeline needs tuning.")
else:
    print("  VERDICT ✗  Write pipeline incomplete — check server logs.")
print()
PYEOF

# =============================================================================
echo ""
echo -e "${BOLD}  Passed: ${GREEN}$PASS${RESET}  ${BOLD}Failed: ${RED}$FAIL${RESET}"
echo -e "  Results: ${CYAN}$RESULTS_DIR/${RESET}"
echo ""
