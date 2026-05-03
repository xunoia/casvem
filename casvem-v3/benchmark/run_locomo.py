"""
LoCoMo benchmark runner.

Downloads snap-research/LoCoMo from HuggingFace on first run.
Ingests conversation turns → queries each QA pair → scores with Token F1 (no LLM judge).

Usage:
  python benchmark/run_locomo.py
  python benchmark/run_locomo.py --limit 20
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_PATH = "benchmark/locomo_data"
RESULTS_DIR = "benchmark/results"
MEM0_BASELINE = 91.6  # Mem0's reported LoCoMo F1 score


def load_dataset():
    if os.path.exists(DATA_PATH):
        from datasets import load_from_disk
        return load_from_disk(DATA_PATH)

    print("Downloading LoCoMo from HuggingFace...")
    from datasets import load_dataset
    ds = load_dataset("snap-research/LoCoMo", split="test")
    ds.save_to_disk(DATA_PATH)
    print(f"Downloaded {len(ds)} conversations → {DATA_PATH}")
    return ds


async def run(limit: int = None):
    from pipeline.ingest import ingest
    from pipeline.query import query as casvem_query
    from benchmark.scorer import token_f1, print_results_table

    ds = load_dataset()
    items = list(ds)
    if limit:
        items = items[:limit]

    print(f"\nRunning LoCoMo on {len(items)} conversations...")
    all_results = []

    for i, item in enumerate(items):
        print(f"  [{i+1}/{len(items)}] processing...", end="\r")

        # Ingest conversation turns
        turns = item.get("conversation", []) or item.get("turns", [])
        for turn in turns:
            text = turn if isinstance(turn, str) else (
                turn.get("utterance") or turn.get("text") or ""
            )
            if text.strip():
                ingest(text=text, memory_type="conversation")

        # Score each QA pair with Token F1
        for qa in item.get("qa", []) or item.get("questions", []):
            q = qa.get("question") or qa.get("input", "")
            gt = qa.get("answer") or qa.get("output", "")
            if not q or not gt:
                continue

            t0 = time.perf_counter()
            result = await casvem_query(text=q)
            latency = (time.perf_counter() - t0) * 1000

            f1 = token_f1(result.answer, gt)
            all_results.append({
                "category": "f1",
                "correct": f1 >= 0.5,   # treat F1 >= 0.5 as "correct" for table
                "f1_score": f1,
                "hit_type": result.hit_type,
                "latency_ms": latency,
                "question": q,
                "expected": gt,
                "got": result.answer,
            })

        # Reset between conversations
        from core.storage import get_storage
        import core.memory.writer as mw
        get_storage().reset_for_benchmark()
        mw._bitmap = None

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = f"{RESULTS_DIR}/locomo_{time.strftime('%Y-%m-%d_%H%M')}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved → {out_path}")

    # Compute average F1 (the actual LoCoMo metric)
    avg_f1 = sum(r["f1_score"] for r in all_results) / len(all_results) * 100 if all_results else 0
    print(f"\n  Average Token F1:  {avg_f1:.1f}  (Mem0 target: {MEM0_BASELINE})")
    delta = avg_f1 - MEM0_BASELINE
    print(f"  vs Mem0:           {'+' if delta >= 0 else ''}{delta:.1f}")
    print_results_table(all_results, "LoCoMo", MEM0_BASELINE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    asyncio.run(run(limit=args.limit))
