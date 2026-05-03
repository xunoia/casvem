"""
BEAM benchmark — local dataset variant (kv_retrieval + longdialogue_qa_eng).

Datasets:
  kv_retrieval.jsonl     — 500 records: UUID→UUID key-value lookup
  longdialogue_qa_eng.jsonl — 200 records: screenplay dialogue fill-in-the-blank

kv_retrieval task:
  Ingest target key-value + N random distractors as memories.
  Query: ask for the value of the target key.
  Score: exact match (answer[0] in response, case insensitive).

longdialogue task:
  Split screenplay context into ~2000-char chunks, ingest first MAX_CHUNKS chunks.
  Query: ask the $$MASK$$ question.
  Score: any answer option in response (case insensitive).

Usage:
  python benchmark/run_beam_local.py
  python benchmark/run_beam_local.py --kv-limit 20 --dlg-limit 10
"""

import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

KV_PATH = "/media/mujahed/CS-Disk/XUNOIA/casvem/casvem-v1/benchmark/beam_data/kv_retrieval.jsonl"
DLG_PATH = "/media/mujahed/CS-Disk/XUNOIA/casvem/casvem-v1/benchmark/beam_data/longdialogue_qa_eng.jsonl"
RESULTS_DIR = str(Path(__file__).parent / "results")

KV_DISTRACTORS = 49      # target key + this many random k/v distractors
DLG_MAX_CHUNKS = 40      # max 2000-char chunks to ingest per dialogue
DLG_CHUNK_SIZE = 2000


def _reset_item_state():
    from core.storage import get_storage
    from core.cache import cache_gate
    from core.memory.writer import reset_bitmap
    get_storage().reset_for_benchmark()
    cache_gate.reset_for_benchmark()
    reset_bitmap()


def _load_jsonl(path: str, limit: int) -> list[dict]:
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
            if len(items) >= limit:
                break
    return items


def _chunk_text(text: str, size: int) -> list[str]:
    chunks = []
    for start in range(0, len(text), size):
        chunk = text[start:start + size].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


async def run_kv(limit: int = 20):
    from pipeline.ingest import ingest
    from pipeline.query import query as casvem_query

    items = _load_jsonl(KV_PATH, limit)
    print(f"\nBEAM kv_retrieval (local) — {len(items)} records  [full: 500]")
    print("─" * 60)

    all_results = []

    for i, item in enumerate(items):
        ctx_str = item["context"].replace("JSON data:\n", "").strip()
        try:
            kv_dict = json.loads(ctx_str)
        except json.JSONDecodeError:
            continue

        # Extract target key from input
        m = re.search(r'Key:\s*"([^"]+)"', item["input"])
        if not m:
            continue
        target_key = m.group(1)
        if target_key not in kv_dict:
            continue
        target_value = kv_dict[target_key]
        expected_answers = [a.lower() for a in item["answer"]]

        _reset_item_state()

        # Build subset: target + random distractors
        other_keys = [k for k in kv_dict if k != target_key]
        random.seed(i)
        distractors = random.sample(other_keys, min(KV_DISTRACTORS, len(other_keys)))
        subset = {target_key: target_value}
        for k in distractors:
            subset[k] = kv_dict[k]

        # Ingest each k/v as one memory
        for k, v in subset.items():
            ingest(text=f"Key: {k} — Value: {v}", memory_type="fact")

        # Query
        query_text = item["input"].strip()
        t0 = time.perf_counter()
        result = await casvem_query(text=query_text)
        latency = (time.perf_counter() - t0) * 1000

        got_lower = result.answer.lower()
        correct = any(ans in got_lower for ans in expected_answers)

        all_results.append({
            "id": item["id"],
            "correct": correct,
            "expected": item["answer"],
            "got": result.answer[:200],
            "hit_type": result.hit_type,
            "latency_ms": round(latency, 1),
            "n_ingested": len(subset),
        })

        status = "✓" if correct else "✗"
        print(f"  [{i+1:2d}/{len(items)}] id={item['id']:4d}  {status}  "
              f"latency={latency:5.0f}ms  n={len(subset)}")

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d_%H%M")
    out_path = os.path.join(RESULTS_DIR, f"beam_kv_{timestamp}.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    total = len(all_results)
    correct_count = sum(1 for r in all_results if r["correct"])
    accuracy = correct_count / total * 100 if total else 0
    avg_latency = sum(r["latency_ms"] for r in all_results) / total if total else 0
    cold_count = sum(1 for r in all_results if r["hit_type"] == "cold")
    cache_hits = total - cold_count

    print(f"\n  Accuracy:       {accuracy:.1f}%  ({correct_count}/{total})")
    print(f"  Avg latency:    {avg_latency:.0f}ms")
    print(f"  Cache hits:     {cache_hits}/{total}")
    print(f"  Results saved → {out_path}")

    return {
        "benchmark": "beam_kv_local",
        "records": total,
        "accuracy": round(accuracy, 1),
        "avg_latency_ms": round(avg_latency, 1),
        "cache_hits": cache_hits,
        "results_file": out_path,
    }


async def run_dialogue(limit: int = 10):
    from pipeline.ingest import ingest
    from pipeline.query import query as casvem_query

    items = _load_jsonl(DLG_PATH, limit)
    print(f"\nBEAM longdialogue (local) — {len(items)} records  [full: 200]")
    print("─" * 60)

    all_results = []

    for i, item in enumerate(items):
        context = item["context"]
        # Transform fill-in-blank into a direct question the LLM can answer
        raw_q = item["input"].replace("$$MASK$$", "___")
        question = (
            f"Based on the screenplay/dialogue context provided, fill in the blank: "
            f"{raw_q}  Answer with ONLY the character name or word, nothing else."
        )
        expected_answers = [a.lower() for a in item["answer"]]

        _reset_item_state()

        # Split context into chunks and ingest first DLG_MAX_CHUNKS
        chunks = _chunk_text(context, DLG_CHUNK_SIZE)[:DLG_MAX_CHUNKS]
        for chunk in chunks:
            ingest(text=chunk, memory_type="document")

        # Query
        t0 = time.perf_counter()
        result = await casvem_query(text=question)
        latency = (time.perf_counter() - t0) * 1000

        got_lower = result.answer.lower()
        correct = any(ans in got_lower for ans in expected_answers)

        all_results.append({
            "id": item["id"],
            "correct": correct,
            "expected": item["answer"],
            "got": result.answer[:200],
            "hit_type": result.hit_type,
            "latency_ms": round(latency, 1),
            "chunks_ingested": len(chunks),
        })

        status = "✓" if correct else "✗"
        print(f"  [{i+1:2d}/{len(items)}] id={item['id']:4d}  {status}  "
              f"latency={latency:5.0f}ms  chunks={len(chunks)}")

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d_%H%M")
    out_path = os.path.join(RESULTS_DIR, f"beam_dlg_{timestamp}.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    total = len(all_results)
    correct_count = sum(1 for r in all_results if r["correct"])
    accuracy = correct_count / total * 100 if total else 0
    avg_latency = sum(r["latency_ms"] for r in all_results) / total if total else 0
    cold_count = sum(1 for r in all_results if r["hit_type"] == "cold")
    cache_hits = total - cold_count

    print(f"\n  Accuracy:       {accuracy:.1f}%  ({correct_count}/{total})")
    print(f"  Avg latency:    {avg_latency:.0f}ms")
    print(f"  Cache hits:     {cache_hits}/{total}")
    print(f"  Results saved → {out_path}")

    return {
        "benchmark": "beam_dlg_local",
        "records": total,
        "accuracy": round(accuracy, 1),
        "avg_latency_ms": round(avg_latency, 1),
        "cache_hits": cache_hits,
        "results_file": out_path,
    }


async def run(kv_limit: int = 20, dlg_limit: int = 10):
    kv_summary = await run_kv(limit=kv_limit)
    dlg_summary = await run_dialogue(limit=dlg_limit)

    print(f"\n{'═' * 60}")
    print("  BEAM Summary")
    print(f"{'═' * 60}")
    print(f"  {'Dataset':<30} {'Accuracy':>10}  {'N':>5}")
    print(f"  {'─' * 48}")
    print(f"  {'kv_retrieval':<30} {kv_summary['accuracy']:>9.1f}%  {kv_summary['records']:>5}")
    print(f"  {'longdialogue_qa_eng':<30} {dlg_summary['accuracy']:>9.1f}%  {dlg_summary['records']:>5}")
    print(f"{'═' * 60}\n")

    return {"kv": kv_summary, "dlg": dlg_summary}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kv-limit", type=int, default=20)
    parser.add_argument("--dlg-limit", type=int, default=10)
    args = parser.parse_args()
    asyncio.run(run(kv_limit=args.kv_limit, dlg_limit=args.dlg_limit))
