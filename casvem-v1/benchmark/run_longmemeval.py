#!/usr/bin/env python3
"""
CaSVeM — LongMemEval Benchmark Runner

Dataset:   xiaowu0162/LongMemEval (500 QA items, 5 question categories)
Scoring:   Local LLM judge (qwen3:4b) → YES/NO per question
Judge:     no cloud cost

Usage:
    python benchmark/run_longmemeval.py              # run 5 items (quick validation)
    python benchmark/run_longmemeval.py --limit 50   # larger run
    python benchmark/run_longmemeval.py --limit 0    # full 500-item run (takes ~40h CPU)
    python benchmark/run_longmemeval.py --resume     # continue from checkpoint

Mem0 published score: 93.4%
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import httpx
from tabulate import tabulate

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
DATA_FILE  = ROOT / "benchmark" / "longmemeval_data" / "longmemeval_oracle"
RESULTS_DIR= ROOT / "benchmark" / "results"
RESULTS_DIR.mkdir(exist_ok=True)

API = "http://localhost:8000"

# ── Load scorer ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(ROOT))
from benchmark.scorer import llm_judge

# ── Question type grouping (for Mem0 comparison) ──────────────────────────────
TYPE_GROUPS = {
    "single-session-user":       "single-hop",
    "single-session-assistant":  "single-hop",
    "single-session-preference": "single-hop",
    "multi-session":             "multi-hop",
    "temporal-reasoning":        "temporal",
    "knowledge-update":          "knowledge-update",
    "no-answer":                 "absent-info",
}
MEM0_SCORES = {
    "single-hop":       93.4,
    "multi-hop":        92.1,
    "temporal":         90.5,
    "knowledge-update": 94.8,
    "absent-info":      None,
    "OVERALL":          93.4,
}

# ─────────────────────────────────────────────────────────────────────────────

def sessions_to_transcripts(item: dict) -> list[str]:
    """Convert haystack_sessions (list of turn lists) into plain text strings."""
    transcripts = []
    for session_turns in item["haystack_sessions"]:
        lines = []
        for turn in session_turns:
            role    = turn.get("role", "user").capitalize()
            content = turn.get("content", "").strip()
            if content:
                lines.append(f"{role}: {content}")
        if lines:
            transcripts.append("\n".join(lines))
    return transcripts


def reset_memory() -> bool:
    try:
        r = httpx.post(f"{API}/admin/reset", timeout=30.0)
        return r.status_code == 200
    except Exception as e:
        print(f"  [WARN] reset failed: {e}")
        return False


def submit_session(session_id: str, transcript: str) -> bool:
    try:
        r = httpx.post(
            f"{API}/session",
            json={"session_id": session_id, "transcript": transcript},
            timeout=15.0,
        )
        return r.status_code == 200
    except Exception:
        return False


def wait_for_pipeline(n_sessions: int, timeout_sec: int = 600) -> bool:
    """Wait until L5 has n_sessions new entries AND L4 has stabilised (no growth for 30s)."""
    # Phase 1: Wait for all L5 sessions to be saved
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            l5 = httpx.get(f"{API}/memory/5", timeout=10.0).json().get("count", 0)
            if l5 >= n_sessions:
                break
        except Exception:
            pass
        time.sleep(5)
    else:
        print(f"  [WARN] L5 timeout")
        return False

    print(f"  L5 sessions saved. Waiting for L4 extraction (CPU: ~{n_sessions*5} min)...")

    # Phase 2: Wait for L4 to grow and then stabilise
    # On CPU: LongMemEval sessions are 12-18 turns (~900 words).
    # qwen3:1.7b generates ~8-15 tok/s → ~5 min per session, queued sequentially.
    min_wait    = n_sessions * 300  # 5 min per session minimum
    stable_wait = 60                # must be stable for 60s before querying
    t_start     = time.time()
    deadline2   = time.time() + max(min_wait + 300, timeout_sec)
    prev_l4     = 0
    stable_since: float | None = None

    time.sleep(min_wait)  # guaranteed minimum wait

    while time.time() < deadline2:
        try:
            l4 = httpx.get(f"{API}/memory/4", timeout=10.0).json().get("count", 0)
            elapsed = int(time.time() - t_start)
            print(f"  L4={l4}  elapsed={elapsed}s", end="\r")
            if l4 > prev_l4:
                prev_l4 = l4
                stable_since = None
            elif l4 > 0:
                if stable_since is None:
                    stable_since = time.time()
                elif time.time() - stable_since >= stable_wait:
                    print(f"\n  L4 stable at {l4} facts after {elapsed}s")
                    return True
        except Exception:
            pass
        time.sleep(10)

    print(f"\n  L4 stabilised or timeout")
    return prev_l4 > 0


def query_memory(question: str) -> dict:
    try:
        r = httpx.post(
            f"{API}/query",
            json={"query": question},
            timeout=180.0,
        )
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {"answer": "", "confidence": "low", "layers_hit": []}


def run_item(item: dict, item_idx: int) -> dict:
    """Run write + read pipeline for one LongMemEval item."""
    qid      = item["question_id"]
    question = item["question"]
    answer   = item["answer"]
    qtype    = item["question_type"]

    reset_memory()

    transcripts = sessions_to_transcripts(item)
    for i, transcript in enumerate(transcripts):
        submit_session(f"{qid}_s{i}", transcript)
    print(f"  Submitted {len(transcripts)} sessions")

    wait_for_pipeline(n_sessions=len(transcripts))

    # Query
    t0     = time.time()
    result = query_memory(question)
    latency_ms = int((time.time() - t0) * 1000)

    ai_answer = result.get("answer", "")
    correct   = llm_judge(question, answer, ai_answer)

    return {
        "question_id":  qid,
        "question_type": qtype,
        "group":        TYPE_GROUPS.get(qtype, qtype),
        "question":     question,
        "expected":     answer,
        "got":          ai_answer,
        "correct":      correct,
        "confidence":   result.get("confidence", "?"),
        "layers_hit":   result.get("layers_hit", []),
        "latency_ms":   latency_ms,
    }


def print_results(results: list[dict], run_id: str):
    by_group: dict[str, list] = defaultdict(list)
    for r in results:
        by_group[r["group"]].append(r["correct"])

    rows = []
    total_c, total_n = 0, 0
    for group in ["single-hop", "multi-hop", "temporal", "knowledge-update", "absent-info"]:
        items = by_group.get(group, [])
        if not items:
            continue
        c = sum(items)
        n = len(items)
        score = 100.0 * c / n
        mem0  = MEM0_SCORES.get(group)
        delta = f"+{score-mem0:.1f}" if mem0 else "—"
        rows.append([group, f"{score:.1f}%", f"{mem0}%" if mem0 else "—", delta, f"{c}/{n}"])
        total_c += c
        total_n += n

    overall = 100.0 * total_c / total_n if total_n else 0
    mem0_overall = MEM0_SCORES["OVERALL"]
    delta_overall = f"+{overall-mem0_overall:.1f}" if mem0_overall else "—"
    rows.append(["OVERALL", f"{overall:.1f}%", f"{mem0_overall}%", delta_overall, f"{total_c}/{total_n}"])

    print()
    print("═" * 66)
    print(f"  CaSVeM vs Mem0 — LongMemEval Results  [{run_id}]")
    print("═" * 66)
    print(tabulate(rows, headers=["Category", "CaSVeM", "Mem0", "Delta", "n"], tablefmt="simple"))
    print()
    avg_lat = int(sum(r["latency_ms"] for r in results) / len(results)) if results else 0
    print(f"  Model:  qwen3:1.7b (write) + qwen3:4b (judge)")
    print(f"  Items evaluated: {len(results)}")
    print(f"  Avg query latency: {avg_lat}ms")
    print("═" * 66)
    print()


def main():
    parser = argparse.ArgumentParser(description="LongMemEval benchmark for CaSVeM")
    parser.add_argument("--limit",  type=int, default=5,
                        help="Number of items to evaluate (0=all 500, default=5)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing checkpoint")
    parser.add_argument("--offset", type=int, default=0,
                        help="Start from this item index")
    args = parser.parse_args()

    # Load dataset
    if not DATA_FILE.exists():
        print(f"ERROR: Dataset not found at {DATA_FILE}")
        print("Run: python -c \"from huggingface_hub import hf_hub_download; "
              "hf_hub_download(repo_id='xiaowu0162/LongMemEval', "
              "filename='longmemeval_oracle', repo_type='dataset', "
              "local_dir='./benchmark/longmemeval_data')\"")
        sys.exit(1)

    with open(DATA_FILE) as f:
        dataset = json.load(f)

    limit  = args.limit if args.limit > 0 else len(dataset)
    subset = dataset[args.offset : args.offset + limit]

    run_id    = f"longmemeval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ckpt_file = RESULTS_DIR / f"{run_id}_checkpoint.json"

    # Load checkpoint if resuming
    done_ids: set[str] = set()
    results:  list[dict] = []
    if args.resume:
        existing = sorted(RESULTS_DIR.glob("longmemeval_*_checkpoint.json"))
        if existing:
            ckpt_file = existing[-1]
            with open(ckpt_file) as f:
                results = json.load(f)
            done_ids = {r["question_id"] for r in results}
            print(f"Resuming from {ckpt_file} — {len(done_ids)} items already done")

    print(f"\n{'═'*66}")
    print(f"  CaSVeM — LongMemEval Benchmark")
    print(f"  Items: {len(subset)}  |  Judge: qwen3:4b  |  Run: {run_id}")
    cpu_est = len(subset) * 13
    print(f"  Estimated time on CPU: ~{cpu_est} min (~{cpu_est//60}h {cpu_est%60}m)")
    print(f"{'═'*66}\n")

    for idx, item in enumerate(subset):
        qid = item["question_id"]
        if qid in done_ids:
            print(f"  [{idx+1}/{len(subset)}] {qid} — SKIP (already done)")
            continue

        group = TYPE_GROUPS.get(item["question_type"], item["question_type"])
        print(f"  [{idx+1}/{len(subset)}] {qid}  type={group}")
        t_start = time.time()
        try:
            result = run_item(item, idx)
        except Exception as e:
            print(f"    ERROR: {e}")
            result = {
                "question_id": qid, "question_type": item["question_type"],
                "group": group, "question": item["question"],
                "expected": item["answer"], "got": "", "correct": False,
                "confidence": "error", "layers_hit": [], "latency_ms": 0,
            }

        elapsed = int(time.time() - t_start)
        verdict = "✓" if result["correct"] else "✗"
        print(f"    {verdict} correct={result['correct']}  conf={result['confidence']}  "
              f"layers={result['layers_hit']}  time={elapsed}s")
        print(f"    Expected: {result['expected'][:70]}")
        print(f"    Got:      {result['got'][:70]}")

        results.append(result)
        done_ids.add(qid)

        # Save checkpoint after every item
        with open(ckpt_file, "w") as f:
            json.dump(results, f, indent=2)

    # Save final results
    final_file = RESULTS_DIR / f"{run_id}.json"
    with open(final_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {final_file}")

    print_results(results, run_id)


if __name__ == "__main__":
    main()
