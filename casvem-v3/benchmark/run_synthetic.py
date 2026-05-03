"""
Synthetic benchmark — CaSVeM v3.

Dataset: benchmark/synthetic/dataset.json
  20 personal-memory facts about a fictional user (Arjun Sharma)
  25 QA pairs across 6 categories:
    single_fact, preference, multi_fact, routine, work, goal, activity,
    repeat_cache_test (same question twice), paraphrase_cache_test (similar wording)

Scoring:
  - Keyword match (≥1 expected keyword in answer = correct)
  - No LLM judge needed → fast, free, reproducible

Output:
  - Per-query: latency, hit_type, correct, tokens used
  - Aggregate: accuracy by category, cache hit rate, token savings
  - Hypothetical comparison: what same queries would cost without CaSVeM

Usage:
  python benchmark/run_synthetic.py
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

DATASET_PATH = str(Path(__file__).parent / "synthetic" / "dataset.json")
RESULTS_DIR = str(Path(__file__).parent / "results")


def _reset_state():
    from core.storage import get_storage
    from core.cache import cache_gate
    from core.memory.writer import reset_bitmap
    get_storage().reset_for_benchmark()
    cache_gate.reset_for_benchmark()
    reset_bitmap()


def _keyword_correct(answer: str, keywords: list[str]) -> bool:
    answer_lower = answer.lower()
    return any(kw.lower() in answer_lower for kw in keywords)


async def run():
    from pipeline.ingest import ingest
    from pipeline.query import query as casvem_query

    with open(DATASET_PATH) as f:
        dataset = json.load(f)

    memories = dataset["memories"]
    qa_list = dataset["qa"]

    print(f"\nCaSVeM Synthetic Benchmark")
    print(f"  {len(memories)} memories  |  {len(qa_list)} queries")
    print("─" * 70)

    _reset_state()

    # ── Phase 1: Ingest all memories ──────────────────────────────────────────
    print("\nPhase 1 — Ingesting memories...")
    ingest_start = time.perf_counter()
    for mem in memories:
        ingest(text=mem["text"], memory_type=mem["type"])
    ingest_total = (time.perf_counter() - ingest_start) * 1000
    print(f"  Ingested {len(memories)} memories in {ingest_total:.0f}ms "
          f"({ingest_total/len(memories):.1f}ms each)")

    # ── Phase 2: Run queries ───────────────────────────────────────────────────
    print(f"\nPhase 2 — Running {len(qa_list)} queries...")
    all_results = []
    total_input_tokens = 0
    total_output_tokens = 0

    for i, qa in enumerate(qa_list):
        q = qa["question"]
        expected_keywords = qa["expected_keywords"]
        category = qa["category"]

        t0 = time.perf_counter()
        result = await casvem_query(text=q)
        latency = (time.perf_counter() - t0) * 1000

        correct = _keyword_correct(result.answer, expected_keywords)
        total_input_tokens += result.input_tokens
        total_output_tokens += result.output_tokens

        hit_icon = {"cold": "❄", "L1": "🔥", "L2": "♨"}.get(result.hit_type, "?")
        status = "✓" if correct else "✗"
        print(f"  [{i+1:2d}/{len(qa_list)}] {category:<25} {hit_icon} {result.hit_type:<5} "
              f"{latency:6.0f}ms  {status}  {q[:40]}")

        all_results.append({
            "id": qa["id"],
            "category": category,
            "question": q,
            "expected_answer": qa["answer"],
            "got": result.answer[:300],
            "correct": correct,
            "hit_type": result.hit_type,
            "latency_ms": round(latency, 1),
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "note": qa.get("note", ""),
        })

    # ── Save results ──────────────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d_%H%M")
    out_path = os.path.join(RESULTS_DIR, f"synthetic_{timestamp}.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # ── Analysis ──────────────────────────────────────────────────────────────
    total = len(all_results)
    correct_count = sum(1 for r in all_results if r["correct"])
    accuracy = correct_count / total * 100

    cold_results = [r for r in all_results if r["hit_type"] == "cold"]
    cached_results = [r for r in all_results if r["hit_type"] in ("L1", "L2")]
    l1_results = [r for r in all_results if r["hit_type"] == "L1"]
    l2_results = [r for r in all_results if r["hit_type"] == "L2"]

    cold_latencies = [r["latency_ms"] for r in cold_results]
    cached_latencies = [r["latency_ms"] for r in cached_results]

    avg_cold_lat = sum(cold_latencies) / len(cold_latencies) if cold_latencies else 0
    avg_cached_lat = sum(cached_latencies) / len(cached_latencies) if cached_latencies else 0
    speedup = avg_cold_lat / avg_cached_lat if avg_cached_lat > 0 else 0
    hit_rate = len(cached_results) / total * 100

    # Per-category accuracy
    cats = {}
    for r in all_results:
        cats.setdefault(r["category"], []).append(r["correct"])

    # ── Token cost comparison ─────────────────────────────────────────────────
    # Actual CaSVeM: cold queries paid tokens, cached paid 0
    casvem_tokens_in = total_input_tokens
    casvem_tokens_out = total_output_tokens

    # Hypothetical without CaSVeM: every query hits LLM
    # Estimate avg tokens per cold query
    if cold_results:
        avg_in_per_cold = sum(r["input_tokens"] for r in cold_results) / len(cold_results)
        avg_out_per_cold = sum(r["output_tokens"] for r in cold_results) / len(cold_results)
    else:
        avg_in_per_cold = 90
        avg_out_per_cold = 30
    no_cache_tokens_in = total * avg_in_per_cold
    no_cache_tokens_out = total * avg_out_per_cold

    # Cost in USD (Gemini 2.5 Flash rates from config)
    cost_per_1k_in = 0.0001
    cost_per_1k_out = 0.0004
    casvem_cost = (casvem_tokens_in * cost_per_1k_in + casvem_tokens_out * cost_per_1k_out) / 1000
    no_cache_cost = (no_cache_tokens_in * cost_per_1k_in + no_cache_tokens_out * cost_per_1k_out) / 1000

    # Scale projections
    def project_cost(n_queries, hit_rate_pct):
        cold = n_queries * (1 - hit_rate_pct / 100)
        return cold * (avg_in_per_cold * cost_per_1k_in + avg_out_per_cold * cost_per_1k_out) / 1000

    # ── Print report ──────────────────────────────────────────────────────────
    print(f"\n{'═' * 70}")
    print("  CaSVeM Synthetic Benchmark — Results")
    print(f"{'═' * 70}")

    print(f"\n  Accuracy by Category")
    print(f"  {'Category':<28} {'Accuracy':>10}  {'N':>4}")
    print(f"  {'─' * 46}")
    for cat, verdicts in sorted(cats.items()):
        acc = sum(verdicts) / len(verdicts) * 100
        print(f"  {cat:<28} {acc:>9.1f}%  {len(verdicts):>4}")
    print(f"  {'─' * 46}")
    print(f"  {'OVERALL':<28} {accuracy:>9.1f}%  {total:>4}")

    print(f"\n  Latency")
    print(f"  {'─' * 46}")
    print(f"  Cold queries:         {len(cold_results):>4}   avg {avg_cold_lat:>7.0f}ms")
    print(f"  L2 cached queries:    {len(l2_results):>4}   avg {(sum(r['latency_ms'] for r in l2_results)/max(1,len(l2_results))):>7.1f}ms")
    print(f"  L1 cached queries:    {len(l1_results):>4}   avg {(sum(r['latency_ms'] for r in l1_results)/max(1,len(l1_results))):>7.1f}ms")
    print(f"  Cache hit rate:       {hit_rate:>7.1f}%")
    if speedup > 1:
        print(f"  Cache speedup:        {speedup:>7.0f}×  (vs cold avg)")

    print(f"\n  Token Usage & Cost  (this benchmark run, {total} queries)")
    print(f"  {'─' * 60}")
    print(f"  {'':30} {'Tokens In':>12}  {'Tokens Out':>11}  {'USD Cost':>10}")
    print(f"  {'CaSVeM actual':<30} {casvem_tokens_in:>12,}  {casvem_tokens_out:>11,}  ${casvem_cost:>9.6f}")
    print(f"  {'Without CaSVeM (est.)':<30} {int(no_cache_tokens_in):>12,}  {int(no_cache_tokens_out):>11,}  ${no_cache_cost:>9.6f}")
    saved_tokens = int(no_cache_tokens_in - casvem_tokens_in) + int(no_cache_tokens_out - casvem_tokens_out)
    saved_pct = (1 - casvem_cost / max(no_cache_cost, 1e-9)) * 100
    print(f"  {'Tokens saved':<30} {saved_tokens:>12,}  {'':>11}  {saved_pct:>9.1f}% saved")

    print(f"\n  Scale Projection — Daily Cost at {hit_rate:.0f}% hit rate")
    print(f"  {'─' * 60}")
    print(f"  {'Queries/day':>15}  {'CaSVeM':>12}  {'No cache':>12}  {'Saving':>12}")
    print(f"  {'─' * 56}")
    for n in [100, 1_000, 10_000, 100_000]:
        casvem_d = project_cost(n, hit_rate)
        naive_d = project_cost(n, 0)
        saving_d = naive_d - casvem_d
        print(f"  {n:>15,}  ${casvem_d:>11.4f}  ${naive_d:>11.4f}  ${saving_d:>11.4f}")

    print(f"\n  Time & Space Complexity")
    print(f"  {'─' * 60}")
    print(f"  Ingest per memory:    O(seq_len × d) encode + O(M log n) HNSW = O(d + log n)")
    print(f"  Query cache hit:      O(d) encode + O(1) LRU lookup          = O(d)")
    print(f"  Query cold path:      O(d) + O(log n) HNSW + O(k×d) rerank  = O(kd + log n)")
    print(f"  Bitmap filter:        O(bits/64) per field ≈ O(1) in practice")
    print(f"  Space per memory:     ~{384*4 + 200} bytes (vector 1.5KB + metadata ~200B)")
    print(f"  HNSW index:           O(n × M × 8B) = {len(memories)} × 16 × 8 = {len(memories)*16*8/1024:.1f}KB")
    print(f"  LRU cache:            O(min(queries, 2500)) ≈ bounded constant")
    print(f"  Ingestion actual:     {ingest_total/len(memories):.1f}ms/memory on CPU (i5-10210U, no GPU)")

    print(f"\n  Results saved → {out_path}")
    print(f"{'═' * 70}\n")

    return {
        "benchmark": "synthetic",
        "records": total,
        "accuracy": round(accuracy, 1),
        "avg_cold_latency_ms": round(avg_cold_lat, 1),
        "avg_cached_latency_ms": round(avg_cached_lat, 1),
        "cache_speedup_x": round(speedup, 0),
        "cache_hit_rate": round(hit_rate, 1),
        "casvem_tokens_in": casvem_tokens_in,
        "casvem_tokens_out": casvem_tokens_out,
        "no_cache_tokens_in": int(no_cache_tokens_in),
        "no_cache_tokens_out": int(no_cache_tokens_out),
        "token_savings_pct": round(saved_pct, 1),
        "cost_casvem_usd": round(casvem_cost, 6),
        "cost_no_cache_usd": round(no_cache_cost, 6),
        "by_category": {cat: round(sum(v)/len(v)*100, 1) for cat, v in cats.items()},
        "results_file": out_path,
    }


if __name__ == "__main__":
    asyncio.run(run())
