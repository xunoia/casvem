"""
LoCoMo benchmark — local dataset variant.

Dataset: /media/mujahed/CS-Disk/XUNOIA/casvem/casvem-v1/benchmark/locomo_data/raw/locomo10.json
Format:  JSON array, 10 records
Fields:  sample_id, conversation (dict with session_N keys), qa (list of QA pairs)

Each session is a list of {speaker, dia_id, text} utterances.
We ingest each session as one combined memory, then query each QA pair using Token F1.

Usage:
  python benchmark/run_locomo_local.py
  python benchmark/run_locomo_local.py --limit 3 --qa-per-record 10
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

DATASET_PATH = "/media/mujahed/CS-Disk/XUNOIA/casvem/casvem-v1/benchmark/locomo_data/raw/locomo10.json"
RESULTS_DIR = str(Path(__file__).parent / "results")
MEM0_BASELINE = 91.6


def load_dataset():
    with open(DATASET_PATH) as f:
        return json.load(f)


def _reset_item_state():
    from core.storage import get_storage
    from core.cache import cache_gate
    from core.memory.writer import reset_bitmap
    get_storage().reset_for_benchmark()
    cache_gate.reset_for_benchmark()
    reset_bitmap()


def _extract_sessions(conv: dict) -> list[str]:
    """Return list of combined session texts from conversation dict."""
    sessions = []
    session_keys = sorted(
        [k for k in conv if k.startswith("session_") and not k.endswith("_date_time")],
        key=lambda k: int(k.split("_")[1])
    )
    for key in session_keys:
        turns = conv[key]
        if isinstance(turns, list):
            parts = [f"{t.get('speaker', '')}: {t.get('text', '')}"
                     for t in turns if t.get("text", "").strip()]
            combined = "\n".join(parts)
        else:
            combined = str(turns)
        if combined.strip():
            sessions.append(combined)
    return sessions


async def run(limit: int = 10, qa_per_record: int = 20):
    from pipeline.ingest import ingest
    from pipeline.query import query as casvem_query
    from benchmark.scorer import token_f1

    data = load_dataset()
    data = data[:limit]
    print(f"\nLoCoMo (local) — {len(data)} records  (QA cap: {qa_per_record}/record)")
    print("─" * 60)

    all_results = []

    for i, item in enumerate(data):
        sample_id = item.get("sample_id", str(i))
        conv = item.get("conversation", {})
        qa_list = item.get("qa", [])

        _reset_item_state()

        # Ingest each session as one combined memory
        sessions = _extract_sessions(conv)
        for session_text in sessions:
            ingest(text=session_text, memory_type="conversation")

        # Score each QA pair (capped per record)
        item_results = []
        for qa in qa_list[:qa_per_record]:
            question = qa.get("question", "")
            answer = str(qa.get("answer", ""))
            category = qa.get("category", "unknown")
            if not question or not answer:
                continue

            t0 = time.perf_counter()
            result = await casvem_query(text=question)
            latency = (time.perf_counter() - t0) * 1000

            f1 = token_f1(result.answer, answer)
            item_results.append({
                "sample_id": sample_id,
                "category": str(category),
                "question": question,
                "expected": answer,
                "got": result.answer,
                "f1_score": round(f1, 4),
                "correct": f1 >= 0.5,
                "hit_type": result.hit_type,
                "latency_ms": round(latency, 1),
            })

        all_results.extend(item_results)
        avg_f1 = sum(r["f1_score"] for r in item_results) / len(item_results) if item_results else 0
        print(f"  [{i+1:2d}/{len(data)}] {sample_id}  {len(sessions)} sessions  "
              f"{len(item_results)} QAs  avg_f1={avg_f1:.3f}")

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d_%H%M")
    out_path = os.path.join(RESULTS_DIR, f"locomo_local_{timestamp}.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary
    total = len(all_results)
    avg_f1_overall = sum(r["f1_score"] for r in all_results) / total * 100 if total else 0
    correct = sum(1 for r in all_results if r["correct"])
    accuracy = correct / total * 100 if total else 0
    avg_latency = sum(r["latency_ms"] for r in all_results) / total if total else 0
    cold_count = sum(1 for r in all_results if r["hit_type"] == "cold")
    cache_hits = total - cold_count

    # By category
    cats: dict[str, list] = {}
    for r in all_results:
        cats.setdefault(r["category"], []).append(r["f1_score"])

    print(f"\n{'═' * 60}")
    print(f"  LoCoMo (local) Results")
    print(f"{'═' * 60}")
    print(f"  {'Category':<20} {'Avg F1':>8}  {'N':>4}")
    print(f"  {'─' * 36}")
    for cat, f1s in sorted(cats.items(), key=lambda x: x[0]):
        cat_f1 = sum(f1s) / len(f1s) * 100
        print(f"  {cat:<20} {cat_f1:>7.1f}%  {len(f1s):>4}")
    print(f"  {'─' * 36}")
    delta = avg_f1_overall - MEM0_BASELINE
    delta_str = f"+{delta:.1f}%" if delta >= 0 else f"{delta:.1f}%"
    print(f"  {'OVERALL Avg F1':<20} {avg_f1_overall:>7.1f}%  {total:>4}")
    print(f"  {'OVERALL Accuracy':<20} {accuracy:>7.1f}%  (F1≥0.5)")
    print(f"\n  vs Mem0 ({MEM0_BASELINE}%):  {delta_str}")
    print(f"  Avg latency:      {avg_latency:.0f}ms")
    print(f"  Cache hits:       {cache_hits}/{total}  ({cache_hits/total*100:.0f}%)")
    print(f"  Results saved →   {out_path}")
    print(f"{'═' * 60}\n")

    return {
        "benchmark": "locomo_local",
        "records": total,
        "avg_f1": round(avg_f1_overall, 1),
        "accuracy_at_0_5": round(accuracy, 1),
        "mem0_baseline": MEM0_BASELINE,
        "delta_vs_mem0": round(delta, 1),
        "avg_latency_ms": round(avg_latency, 1),
        "cache_hits": cache_hits,
        "cache_hit_rate": round(cache_hits / total * 100, 1) if total else 0,
        "by_category": {cat: round(sum(f1s) / len(f1s) * 100, 1) for cat, f1s in cats.items()},
        "results_file": out_path,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--qa-per-record", type=int, default=20)
    args = parser.parse_args()
    asyncio.run(run(limit=args.limit, qa_per_record=args.qa_per_record))
