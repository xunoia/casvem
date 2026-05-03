"""
Benchmark scoring utilities.

LLM judge:  used for LongMemEval (YES/NO per question)
Token F1:   used for LoCoMo (no LLM needed)

Opt 2: batch_judge() uses asyncio.gather() with a semaphore to run
       multiple judge calls concurrently — ~12x faster than sequential.
"""

import asyncio
from typing import Optional


# ── LLM Judge ─────────────────────────────────────────────────────────────────

async def batch_judge(
    qa_pairs: list[dict],
    concurrency: int = 15,
    provider=None,
) -> list[bool]:
    """
    Opt 2: run all judge calls concurrently under a semaphore.

    qa_pairs: list of {"question": str, "ground_truth": str, "answer": str}
    Returns: list of bool (True = correct)
    """
    if provider is None:
        from core.llm import get_judge_provider
        provider = get_judge_provider()

    sem = asyncio.Semaphore(concurrency)

    async def judge_one(pair: dict) -> bool:
        async with sem:
            return await provider.judge(
                question=pair["question"],
                ground_truth=pair["ground_truth"],
                answer=pair["answer"],
            )

    return await asyncio.gather(*[judge_one(p) for p in qa_pairs])


# ── Token F1 (LoCoMo) ─────────────────────────────────────────────────────────

def token_f1(prediction: str, ground_truth: str) -> float:
    """
    Compute token-level F1 between prediction and ground truth.
    Standard LoCoMo scoring — no LLM needed.
    """
    pred_tokens = set(prediction.lower().split())
    truth_tokens = set(ground_truth.lower().split())

    # Remove stop words that inflate score
    stop = {"a", "an", "the", "is", "was", "are", "were", "in", "on", "at", "of", "to"}
    pred_tokens -= stop
    truth_tokens -= stop

    common = pred_tokens & truth_tokens
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens) if pred_tokens else 0.0
    recall = len(common) / len(truth_tokens) if truth_tokens else 0.0
    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


# ── Results formatting ────────────────────────────────────────────────────────

def print_results_table(results: list[dict], benchmark_name: str, mem0_score: float):
    """
    results: list of {"category": str, "correct": bool, "latency_ms": float, ...}
    """
    from tabulate import tabulate

    # Group by category
    categories: dict[str, list[bool]] = {}
    for r in results:
        cat = r.get("category", "overall")
        categories.setdefault(cat, []).append(r["correct"])

    rows = []
    for cat, verdicts in sorted(categories.items()):
        acc = sum(verdicts) / len(verdicts) * 100
        rows.append([cat, f"{acc:.1f}%", len(verdicts)])

    all_correct = [r["correct"] for r in results]
    overall = sum(all_correct) / len(all_correct) * 100 if all_correct else 0

    rows.append(["─" * 20, "─" * 8, "─" * 6])
    delta = overall - mem0_score
    delta_str = f"+{delta:.1f}%" if delta >= 0 else f"{delta:.1f}%"
    rows.append(["OVERALL", f"{overall:.1f}%", len(results)])

    avg_latency = sum(r.get("latency_ms", 0) for r in results) / len(results) if results else 0
    cold_count = sum(1 for r in results if r.get("hit_type") == "cold")
    hit_count = len(results) - cold_count

    print(f"\n{'═' * 60}")
    print(f"  {benchmark_name} Results  (Mem0 baseline: {mem0_score}%)")
    print(f"{'═' * 60}")
    print(tabulate(rows, headers=["Category", "Accuracy", "N"], tablefmt="simple"))
    print(f"\n  vs Mem0:          {delta_str}")
    print(f"  Avg latency:      {avg_latency:.1f}ms")
    print(f"  Cache hits:       {hit_count} / {len(results)}")
    print(f"  Cache hit rate:   {hit_count / len(results) * 100:.1f}%")
    print(f"{'═' * 60}\n")
