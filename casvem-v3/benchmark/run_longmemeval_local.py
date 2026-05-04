"""
LongMemEval benchmark — local dataset variant.

Dataset: /media/mujahed/CS-Disk/XUNOIA/casvem/casvem-v1/benchmark/longmemeval_data/longmemeval_oracle
Format:  JSON array, 500 records
Fields:  question_id, question_type, question, answer, haystack_sessions (list of sessions)

Each session is a list of {role, content} messages.
We combine all messages in a session into one memory text, then query with the question
and judge the answer against ground truth using Gemini (LLM judge).

Usage:
  python benchmark/run_longmemeval_local.py
  python benchmark/run_longmemeval_local.py --limit 10
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

DATASET_PATH = "/media/mujahed/CS-Disk/XUNOIA/casvem/casvem-v1/benchmark/longmemeval_data/longmemeval_oracle"
RESULTS_DIR = str(Path(__file__).parent / "results")
MEM0_BASELINE = 93.4


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


async def run(limit: int = 20):
    from pipeline.ingest import ingest
    from pipeline.query import query as casvem_query
    from benchmark.scorer import batch_judge

    data = load_dataset()
    items = data[:limit]
    print(f"\nLongMemEval (local) — {len(items)} records  [full dataset: {len(data)}]")
    print("─" * 60)

    all_results = []
    total_input_tokens = 0
    total_output_tokens = 0

    for i, item in enumerate(items):
        sessions = item.get("haystack_sessions", [])
        session_dates = item.get("haystack_dates", [])
        question = item.get("question", "")
        answer = item.get("answer", "")
        q_type = item.get("question_type", "unknown")

        if not question or not answer:
            continue

        _reset_item_state()

        # Chunk each session into 500-char pieces with date prefix on every chunk.
        # Smaller chunks = each memory is more focused = cross-encoder picks the right one.
        CHUNK_SIZE = 500
        for idx, session in enumerate(sessions):
            date_str = session_dates[idx] if idx < len(session_dates) else ""
            if isinstance(session, list):
                parts = []
                for msg in session:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if content.strip():
                        parts.append(f"{role}: {content}")
                combined = "\n".join(parts)
            else:
                combined = str(session)
            if not combined.strip():
                continue
            date_prefix = f"[Date: {date_str}] " if date_str else ""
            for start in range(0, len(combined), CHUNK_SIZE):
                chunk = combined[start:start + CHUNK_SIZE].strip()
                if chunk:
                    ingest(text=f"{date_prefix}{chunk}", memory_type="session")

        # top_k=300 searches all chunks; top_n=12 + token_budget=6000 + no early exit
        # so all 12 chunks reach the LLM (needed for multi-session temporal reasoning)
        t0 = time.perf_counter()
        result = await casvem_query(text=question, top_k=300, top_n=12, token_budget=6000, early_exit=False)
        latency = (time.perf_counter() - t0) * 1000

        total_input_tokens += result.input_tokens
        total_output_tokens += result.output_tokens

        all_results.append({
            "idx": i,
            "question_id": item.get("question_id", ""),
            "category": q_type,
            "question": question,
            "expected": answer,
            "got": result.answer,
            "hit_type": result.hit_type,
            "latency_ms": round(latency, 1),
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
        })

        print(f"  [{i+1:2d}/{len(items)}] {q_type:<30} latency={latency:6.0f}ms  hit={result.hit_type}")

    # Judge all answers concurrently (Opt 2)
    print("\nJudging answers...")
    qa_pairs = [{"question": r["question"], "ground_truth": r["expected"], "answer": r["got"]}
                for r in all_results]
    verdicts = await batch_judge(qa_pairs, concurrency=10)
    for r, verdict in zip(all_results, verdicts):
        r["correct"] = verdict

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d_%H%M")
    out_path = os.path.join(RESULTS_DIR, f"longmemeval_local_{timestamp}.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    total = len(all_results)
    correct = sum(1 for r in all_results if r["correct"])
    accuracy = correct / total * 100 if total > 0 else 0
    avg_latency = sum(r["latency_ms"] for r in all_results) / total if total else 0
    cold_count = sum(1 for r in all_results if r["hit_type"] == "cold")
    cache_hits = total - cold_count

    # By category
    cats: dict[str, list] = {}
    for r in all_results:
        cats.setdefault(r["category"], []).append(r["correct"])

    print(f"\n{'═' * 60}")
    print(f"  LongMemEval (local) Results")
    print(f"{'═' * 60}")
    print(f"  {'Category':<35} {'Acc':>6}  {'N':>4}")
    print(f"  {'─' * 48}")
    for cat, verdicts_list in sorted(cats.items()):
        cat_acc = sum(verdicts_list) / len(verdicts_list) * 100
        print(f"  {cat:<35} {cat_acc:>5.1f}%  {len(verdicts_list):>4}")
    print(f"  {'─' * 48}")
    delta = accuracy - MEM0_BASELINE
    delta_str = f"+{delta:.1f}%" if delta >= 0 else f"{delta:.1f}%"
    print(f"  {'OVERALL':<35} {accuracy:>5.1f}%  {total:>4}")
    print(f"\n  vs Mem0 ({MEM0_BASELINE}%):      {delta_str}")
    print(f"  Avg latency:            {avg_latency:.0f}ms")
    print(f"  Cache hits:             {cache_hits}/{total}  ({cache_hits/total*100:.0f}%)")
    print(f"  Total input tokens:     {total_input_tokens:,}")
    print(f"  Total output tokens:    {total_output_tokens:,}")
    print(f"  Results saved →         {out_path}")
    print(f"{'═' * 60}\n")

    return {
        "benchmark": "longmemeval_local",
        "records": total,
        "accuracy": round(accuracy, 1),
        "mem0_baseline": MEM0_BASELINE,
        "delta_vs_mem0": round(delta, 1),
        "avg_latency_ms": round(avg_latency, 1),
        "cache_hits": cache_hits,
        "cache_hit_rate": round(cache_hits / total * 100, 1) if total else 0,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "by_category": {cat: round(sum(v) / len(v) * 100, 1) for cat, v in cats.items()},
        "results_file": out_path,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    asyncio.run(run(limit=args.limit))
