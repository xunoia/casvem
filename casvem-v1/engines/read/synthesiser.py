"""
Read pipeline — Steps 5–7
  5. Synthesise and answer in one LLM call (fast path)
  6. Routing decision (deterministic — no LLM needed for LOCAL)
"""

from __future__ import annotations
from engines.read.searcher import SearchResult
from providers.router import strong
from models import RoutingDecision

_ANSWER_PROMPT = """\
You are a personal memory assistant. Answer the user's query using ONLY the memory facts below.
Each fact is prefixed with [Lx DATE] where Lx is the memory layer and DATE is when it was recorded.

STRICT RULES:
- Answer ONLY from the facts listed. Do NOT guess, infer, or use outside knowledge.
- CONTRADICTION RULE: If two facts contradict (e.g., "using X" vs "switching from X to Y"), the LATER-dated fact is the current truth. Always use the more recent fact.
- For each part of the query: if the fact is present, state it. If not present, say "I don't know [that specific detail]" for that part only.
- Never fill in missing details with assumptions.
- Keep the answer to 1-2 sentences.

Memory Facts:
{memories}

User Query: {query}

Answer:"""


async def synthesise(query: str, result: SearchResult) -> str:
    """Build a flat list of memory facts as a context string, newest first."""
    all_nodes = result.l1_nodes + result.l2_nodes + result.l3_nodes + result.l4_nodes
    l5_nodes  = result.l5_nodes
    if not all_nodes and not l5_nodes:
        return "No relevant memories found."
    # Sort newest first so the LLM sees updates before stale facts
    all_nodes.sort(key=lambda n: n.created_at, reverse=True)
    lines = []
    for n in all_nodes[:20]:
        date_str = n.created_at.strftime("%Y-%m-%d %H:%M")
        lines.append(f"[L{n.layer} {date_str}] {n.content}")
    for n in l5_nodes:
        date_str = n.created_at.strftime("%Y-%m-%d %H:%M")
        lines.append(f"[L5-RAW {date_str}] {n.content[:400]}")
    return "\n".join(lines)


async def route(query: str, memory_block: str, confidence: str, node_count: int) -> str:
    """Simple deterministic routing — no extra LLM call needed."""
    if node_count == 0 or confidence == "low":
        return RoutingDecision.CLOUD
    return RoutingDecision.LOCAL


async def answer(query: str, memory_block: str) -> str:
    """Answer the query directly from the memory block in one LLM call."""
    if not memory_block or memory_block == "No relevant memories found.":
        return "I don't have that information."
    prompt = _ANSWER_PROMPT.format(memories=memory_block, query=query)
    return await strong().generate(prompt)
