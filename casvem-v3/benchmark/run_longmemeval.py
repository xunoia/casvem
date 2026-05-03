"""
LongMemEval benchmark runner.

Downloads xiaowu0162/LongMemEval from HuggingFace on first run.
Ingests sessions → queries each question → judges answers concurrently (Opt 2).
Prints results by category vs Mem0 baseline.

Usage:
  python benchmark/run_longmemeval.py
  python benchmark/run_longmemeval.py --limit 50   # quick test on first 50 items
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_PATH = "benchmark/longmemeval_data"
RESULTS_DIR = "benchmark/results"
MEM0_BASELINE = 93.4  # Mem0's reported LongMemEval score


def load_dataset():
    if os.path.exists(DATA_PATH):
        from datasets import load_from_disk
        return load_from_disk(DATA_PATH)

    print("Downloading LongMemEval from HuggingFace...")
    from datasets import load_dataset
    ds = load_dataset("xiaowu0162/LongMemEval", split="test")
    ds.save_to_disk(DATA_PATH)
    print(f"Downloaded {len(ds)} test cases → {DATA_PATH}")
    return ds


async def run(limit: int = None):
    from pipeline.ingest import ingest
    from pipeline.query import query as casvem_query
    from benchmark.scorer import batch_judge, print_results_table

    ds = load_dataset()
    items = list(ds)
    if limit:
        items = items[:limit]

    print(f"\nRunning LongMemEval on {len(items)} items...")
    all_results = []

    for i, item in enumerate(items):
        print(f"  [{i+1}/{len(items)}] ingesting sessions...", end="\r")

        # Ingest all sessions for this item
        for session in item.get("sessions", []):
            text = session if isinstance(session, str) else session.get("content", "")
            if text.strip():
                ingest(text=text, memory_type="session")

        # Query each question
        qa_pairs = []
        query_results = []
        for qa in item.get("questions", []):
            q = qa.get("question", "") or qa.get("input", "")
            gt = qa.get("answer", "") or qa.get("output", "")
            if not q or not gt:
                continue

            t0 = time.perf_counter()
            result = await casvem_query(text=q)
            latency = (time.perf_counter() - t0) * 1000

            qa_pairs.append({
                "question": q,
                "ground_truth": gt,
                "answer": result.answer,
            })
            query_results.append({
                "category": qa.get("type", "unknown"),
                "hit_type": result.hit_type,
                "latency_ms": latency,
            })

        # Judge all answers for this item concurrently
        verdicts = await batch_judge(qa_pairs, concurrency=15)

        for j, (verdict, meta) in enumerate(zip(verdicts, query_results)):
            all_results.append({
                **meta,
                "correct": verdict,
                "question": qa_pairs[j]["question"],
                "expected": qa_pairs[j]["ground_truth"],
                "got": qa_pairs[j]["answer"],
            })

        # Reset memory between items so items don't bleed into each other
        from core.storage import get_storage
        import core.memory.writer as mw
        get_storage().reset_for_benchmark()
        mw._bitmap = None

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = f"{RESULTS_DIR}/longmemeval_{time.strftime('%Y-%m-%d_%H%M')}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved → {out_path}")

    print_results_table(all_results, "LongMemEval", MEM0_BASELINE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of dataset items (for quick testing)")
    args = parser.parse_args()
    asyncio.run(run(limit=args.limit))
