"""
Shared scoring utilities for CaSVeM benchmarks.

  token_f1(pred, truth)  — LoCoMo-style token-overlap F1
  llm_judge(...)         — LongMemEval-style YES/NO judge via local Ollama
"""

from __future__ import annotations
import re
import httpx

JUDGE_MODEL = "qwen3:4b"
OLLAMA_URL  = "http://localhost:11434"

_JUDGE_PROMPT = """\
You are evaluating whether an AI answer is correct.

Question: {question}
Ground truth answer: {ground_truth}
AI's answer: {ai_answer}

Is the AI's answer correct or semantically equivalent to the ground truth?
Reply with ONLY one word: YES or NO"""

_STOPWORDS = {
    "a","an","the","is","are","was","were","be","been","being",
    "have","has","had","do","does","did","will","would","could",
    "should","may","might","shall","can","i","my","your","their",
    "our","in","on","at","to","for","of","and","or","but","not",
    "it","its","this","that","these","those","what","which","who",
    "how","when","where","why",
}


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"\b\w+\b", text.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


def token_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-overlap F1 between prediction and ground truth."""
    pred_tokens  = _tokenize(prediction)
    truth_tokens = _tokenize(ground_truth)
    if not pred_tokens or not truth_tokens:
        return 1.0 if not pred_tokens and not truth_tokens else 0.0

    pred_set  = set(pred_tokens)
    truth_set = set(truth_tokens)
    common    = pred_set & truth_set
    if not common:
        return 0.0

    precision = len(common) / len(pred_set)
    recall    = len(common) / len(truth_set)
    return 2 * precision * recall / (precision + recall)


def llm_judge(
    question: str,
    ground_truth: str,
    ai_answer: str,
    model: str = JUDGE_MODEL,
) -> bool:
    """Ask a local LLM to judge whether ai_answer is correct. Returns True/False."""
    prompt = _JUDGE_PROMPT.format(
        question=question,
        ground_truth=ground_truth,
        ai_answer=ai_answer,
    )
    try:
        resp = httpx.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "think": False,
                "options": {"num_predict": 8, "temperature": 0.0},
            },
            timeout=120.0,
        )
        resp.raise_for_status()
        text = resp.json()["message"]["content"].strip().upper()
        return text.startswith("YES")
    except Exception:
        # Conservative: if judge fails, count as wrong
        return False
