#!/usr/bin/env python3
"""
CaSVeM — LoCoMo Benchmark Runner

Dataset:   snap-research/LoCoMo (gated on HuggingFace — requires HF token)
           Real multi-turn human conversations, ~300 turns each.
Scoring:   Token F1 (no LLM judge needed — fast)

Mem0 published score: 91.6 F1

Setup (if dataset not downloaded yet):
    export HF_TOKEN=your_token_here
    python -c "
    from huggingface_hub import hf_hub_download
    import os
    hf_hub_download(
        repo_id='snap-research/LoCoMo',
        filename='locomo_data.json',
        repo_type='dataset',
        token=os.environ['HF_TOKEN'],
        local_dir='./benchmark/locomo_data'
    )"

Usage:
    python benchmark/run_locomo.py             # 3 conversations (quick)
    python benchmark/run_locomo.py --limit 20
    python benchmark/run_locomo.py --limit 0   # all (takes ~8h CPU)
    python benchmark/run_locomo.py --resume
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

ROOT       = Path(__file__).parent.parent
DATA_DIR   = ROOT / "benchmark" / "locomo_data"
RESULTS_DIR= ROOT / "benchmark" / "results"
RESULTS_DIR.mkdir(exist_ok=True)

API = "http://localhost:8000"

sys.path.insert(0, str(ROOT))
from benchmark.scorer import token_f1

MEM0_F1 = 91.6

# ── Dataset loader ─────────────────────────────────────────────────────────────

def load_locomo() -> list[dict]:
    """Load LoCoMo dataset — supports locomo10.json (locomo-mc10 format)."""
    # Try both the raw file and top-level json files
    candidates = (
        list(DATA_DIR.rglob("locomo10.json")) +
        list(DATA_DIR.rglob("locomo_mc10*.json")) +
        list(DATA_DIR.glob("*.json")) +
        list(DATA_DIR.glob("*.jsonl"))
    )
    if not candidates:
        print(f"ERROR: No LoCoMo data found in {DATA_DIR}")
        print()
        print("Download with:")
        print("  python -c \"")
        print("  from huggingface_hub import hf_hub_download")
        print("  hf_hub_download(repo_id='Percena/locomo-mc10',")
        print("    filename='raw/locomo10.json', repo_type='dataset',")
        print("    local_dir='./benchmark/locomo_data')\"")
        sys.exit(1)

    data_file = candidates[0]
    print(f"Loading LoCoMo from {data_file}")
    with open(data_file) as f:
        raw = json.load(f)
    if isinstance(raw, list):
        return raw
    for key in ("data", "conversations", "items"):
        if key in raw:
            return raw[key]
    return list(raw.values())


def conversation_to_sessions(item: dict) -> list[str]:
    """Convert locomo-mc10 conversation dict into per-session transcripts."""
    conv = item["conversation"]
    speaker_a = conv.get("speaker_a", "Person A")
    speaker_b = conv.get("speaker_b", "Person B")

    transcripts = []
    session_num = 1
    while True:
        sess_key = f"session_{session_num}"
        date_key = f"session_{session_num}_date_time"
        if sess_key not in conv:
            break
        turns = conv[sess_key]
        date  = conv.get(date_key, "")
        lines = [f"[Date: {date}]"] if date else []
        for turn in turns:
            speaker = turn.get("speaker", "")
            text    = turn.get("text", "").strip()
            if text:
                lines.append(f"{speaker}: {text}")
        if lines:
            transcripts.append("\n".join(lines))
        session_num += 1

    return transcripts


def get_qa_pairs(item: dict) -> list[dict]:
    """Extract QA pairs from a locomo-mc10 item."""
    return item.get("qa", [])


# ── Pipeline helpers (same as LongMemEval) ────────────────────────────────────

def reset_memory() -> bool:
    try:
        r = httpx.post(f"{API}/admin/reset", timeout=30.0)
        return r.status_code == 200
    except Exception:
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
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            if httpx.get(f"{API}/memory/5", timeout=10.0).json().get("count", 0) >= n_sessions:
                break
        except Exception:
            pass
        time.sleep(5)
    else:
        return False
    # Wait for L4 extraction to stabilise
    min_wait   = n_sessions * 300  # 5 min per session on CPU
    stable_sec = 60
    time.sleep(min_wait)
    prev_l4, stable_since = 0, None
    deadline2 = time.time() + 300
    while time.time() < deadline2:
        try:
            l4 = httpx.get(f"{API}/memory/4", timeout=10.0).json().get("count", 0)
            if l4 > prev_l4:
                prev_l4 = l4; stable_since = None
            elif l4 > 0:
                if stable_since is None: stable_since = time.time()
                elif time.time() - stable_since >= stable_sec:
                    return True
        except Exception:
            pass
        time.sleep(10)
    return prev_l4 > 0


def query_memory(question: str) -> dict:
    try:
        r = httpx.post(f"{API}/query", json={"query": question}, timeout=180.0)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {"answer": "", "confidence": "low"}


def run_conversation(conv: dict, conv_idx: int) -> list[dict]:
    """Process one LoCoMo conversation and evaluate all its QA pairs."""
    conv_id   = conv.get("sample_id", conv.get("id", f"conv_{conv_idx}"))
    qa_pairs  = get_qa_pairs(conv)
    if not qa_pairs:
        return []

    reset_memory()

    transcripts = conversation_to_sessions(conv)
    for i, t in enumerate(transcripts):
        submit_session(f"{conv_id}_s{i}", t)

    wait_for_pipeline(n_sessions=len(transcripts))

    results = []
    for qa in qa_pairs:
        question = qa.get("question", qa.get("q", ""))
        expected = qa.get("answer", qa.get("a", ""))
        if isinstance(expected, list):
            expected = "; ".join(str(e) for e in expected)
        if not question or not expected:
            continue

        r      = query_memory(question)
        answer = r.get("answer", "")
        f1     = token_f1(answer, expected)

        results.append({
            "conv_id":    conv_id,
            "question":   question,
            "expected":   expected,
            "got":        answer,
            "f1":         f1,
            "confidence": r.get("confidence", "?"),
            "layers_hit": r.get("layers_hit", []),
        })

    return results


def print_results(all_results: list[dict], run_id: str):
    if not all_results:
        print("  No results to display.")
        return

    f1_scores = [r["f1"] for r in all_results]
    avg_f1    = sum(f1_scores) / len(f1_scores) * 100
    delta     = avg_f1 - MEM0_F1

    rows = [
        ["CaSVeM F1", f"{avg_f1:.1f}"],
        ["Mem0 F1",   f"{MEM0_F1:.1f}"],
        ["Delta",     f"{delta:+.1f}"],
        ["Questions", str(len(all_results))],
    ]
    print()
    print("═" * 50)
    print(f"  CaSVeM vs Mem0 — LoCoMo Results  [{run_id}]")
    print("═" * 50)
    print(tabulate(rows, tablefmt="simple"))
    print()

    # F1 distribution
    buckets = defaultdict(int)
    for r in all_results:
        b = int(r["f1"] * 10) / 10
        buckets[b] += 1
    print("  F1 distribution:")
    for b in sorted(buckets.keys(), reverse=True):
        bar = "█" * buckets[b]
        print(f"    {b:.1f}: {bar} ({buckets[b]})")
    print("═" * 50)
    print()


def main():
    parser = argparse.ArgumentParser(description="LoCoMo benchmark for CaSVeM")
    parser.add_argument("--limit",  type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    dataset = load_locomo()
    limit   = args.limit if args.limit > 0 else len(dataset)
    subset  = dataset[:limit]

    run_id    = f"locomo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ckpt_file = RESULTS_DIR / f"{run_id}_checkpoint.json"

    done_ids: set[str] = set()
    all_results: list[dict] = []
    if args.resume:
        existing = sorted(RESULTS_DIR.glob("locomo_*_checkpoint.json"))
        if existing:
            ckpt_file = existing[-1]
            with open(ckpt_file) as f:
                all_results = json.load(f)
            done_ids = {r["conv_id"] for r in all_results}

    print(f"\n{'═'*55}")
    print(f"  CaSVeM — LoCoMo Benchmark")
    print(f"  Conversations: {len(subset)}  |  Run: {run_id}")
    print(f"  Estimated time on CPU: ~{len(subset) * 20} min")
    print(f"{'═'*55}\n")

    for idx, conv in enumerate(subset):
        conv_id = conv.get("sample_id", conv.get("id", f"conv_{idx}"))
        if conv_id in done_ids:
            print(f"  [{idx+1}/{len(subset)}] {conv_id} — SKIP")
            continue

        print(f"  [{idx+1}/{len(subset)}] {conv_id}")
        t_start = time.time()
        try:
            results = run_conversation(conv, idx)
        except Exception as e:
            print(f"    ERROR: {e}")
            results = []

        avg = sum(r["f1"] for r in results) / len(results) if results else 0
        elapsed = int(time.time() - t_start)
        print(f"    F1={avg:.3f}  questions={len(results)}  time={elapsed}s")

        all_results.extend(results)
        done_ids.add(conv_id)
        with open(ckpt_file, "w") as f:
            json.dump(all_results, f, indent=2)

    final_file = RESULTS_DIR / f"{run_id}.json"
    with open(final_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {final_file}")

    print_results(all_results, run_id)


if __name__ == "__main__":
    main()
