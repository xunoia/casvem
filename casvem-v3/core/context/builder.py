import math
import time
from config import cfg


def build_context(query: str, memories: list[dict], token_budget: int = None) -> str:
    """
    Assemble a context string from reranked memories for the LLM prompt.

    Sorting: combined_score = 0.6 * relevance + 0.4 * recency_weight
    Recency weight: exp(-decay * days_since_created)
    Truncation: add memories until token budget is exceeded.

    Returns a formatted string ready to inject into the LLM prompt.
    """
    budget = token_budget or cfg.context_token_budget
    now = int(time.time())
    decay = cfg.cache_confidence_decay

    scored = []
    for mem in memories:
        relevance = mem.get("rerank_score", 0.5)
        days_old = max(0, (now - mem.get("created_at", now)) / 86400)
        recency_weight = math.exp(-decay * days_old)
        combined = 0.6 * relevance + 0.4 * recency_weight
        scored.append((combined, mem))

    scored.sort(key=lambda x: x[0], reverse=True)

    context_parts = []
    token_count = 0

    for _, mem in scored:
        text = mem["text"]
        # Rough token estimate: ~0.75 tokens per char
        est_tokens = int(len(text) * 0.75)
        if token_count + est_tokens > budget:
            break
        context_parts.append(text)
        token_count += est_tokens

    if not context_parts:
        return ""

    return "\n\n".join(f"[Memory]: {part}" for part in context_parts)


ANSWER_PROMPT = """You are an AI assistant with access to the following memory context.
Use ONLY the provided memories to answer the question.
If the answer is not in the memories, say "I don't have that information."

{context}

Question: {question}

Answer:"""


def build_prompt(query: str, context: str) -> str:
    return ANSWER_PROMPT.format(context=context, question=query)
