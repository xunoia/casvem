"""
Read pipeline — Steps 5–7
  5. Synthesise a query-aware memory block from retrieved nodes
  6. Routing decision (LOCAL vs CLOUD)
  7. Generate the final answer
"""

from __future__ import annotations
from engines.read.searcher import SearchResult
from providers.router import fast, strong
from models import RoutingDecision

_SYNTHESISE_PROMPT = """\
You have retrieved memory entries relevant to the user's query.
Synthesise them into a concise, coherent memory brief.
Maximum 500 tokens. Focus on what directly answers the query.
Explicitly flag any low-confidence items.
Return ONLY the memory brief, no preamble.

QUERY: {query}

MEMORY ENTRIES (most recent / highest confidence first):
{memories}

Memory brief:"""

_ROUTE_PROMPT = """\
Should this query be answered locally or sent to a cloud LLM?

LOCAL  — simple fact recall, preference check, single-session event, high confidence
CLOUD  — complex multi-step reasoning, deep analysis, file attached, low confidence on critical facts

Reply with ONLY one word: LOCAL or CLOUD

QUERY: {query}
MEMORY CONFIDENCE: {confidence}
MEMORY LINES FOUND: {count}

Reply:"""

_ANSWER_PROMPT = """\
Using ONLY the information in the Memory Block below, answer the user's query.
If the answer is not in the Memory Block, say exactly: "I don't have that information."
Do not use any knowledge outside the Memory Block.

Memory Block:
{memory_block}

Query: {query}

Answer:"""


async def synthesise(query: str, result: SearchResult) -> str:
    all_nodes = result.l1_nodes + result.l2_nodes + result.l3_nodes + result.l4_nodes

    if not all_nodes:
        return "No relevant memories found."

    # Build memories text — highest retention first
    all_nodes.sort(key=lambda n: n.retention_score, reverse=True)
    memories_text = []
    for n in all_nodes[:20]:
        conf_tag = f"[{n.confidence.upper()}]" if n.confidence != "high" else ""
        memories_text.append(f"[L{n.layer}]{conf_tag} {n.content}")

    prompt = _SYNTHESISE_PROMPT.format(
        query    = query,
        memories = "\n".join(memories_text),
    )
    return await strong().generate(prompt)


async def route(query: str, memory_block: str, confidence: str, node_count: int) -> str:
    prompt = _ROUTE_PROMPT.format(
        query      = query,
        confidence = confidence,
        count      = node_count,
    )
    raw = await fast().generate(prompt, temperature=0.0)
    word = raw.strip().upper().split()[0] if raw.strip() else "LOCAL"
    return RoutingDecision.CLOUD if "CLOUD" in word else RoutingDecision.LOCAL


async def answer(query: str, memory_block: str) -> str:
    prompt = _ANSWER_PROMPT.format(memory_block=memory_block, query=query)
    return await strong().generate(prompt)
