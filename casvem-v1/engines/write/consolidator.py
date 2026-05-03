"""
Write pipeline — Step 3
  L4 → L3 consolidation.

For each new L4 fact, find the most relevant L3 entry on the same topic, then
classify the relationship and act on it:

  CONTRADICTS → archive old L3, create new L3 with contradiction graph edge
  ADDS        → update L3 content (mark dirty), add pointer
  REINFORCES  → update recency only — no re-summarise needed
  IRRELEVANT  → ignore (no matching L3 topic) → create new L3 entry
"""

from __future__ import annotations
import json

from models import MemoryNode, ConsolidationLabel
from database.weaviate_store import get_store
from providers.router import fast, strong, embedder

_CLASSIFY_PROMPT = """\
Classify how this new fact relates to the existing memory entry.
Reply with ONLY one word — nothing else.

NEW FACT:
{new_fact}

EXISTING MEMORY:
{existing}

Choose one:
  CONTRADICTS  — the new fact directly contradicts or replaces the existing memory
  ADDS         — the new fact adds new information not already in the existing memory
  REINFORCES   — the new fact confirms or repeats what is already in the existing memory
  IRRELEVANT   — the new fact is about a completely different topic

Reply:"""

_CREATE_L3_PROMPT = """\
Write a concise knowledge entry (1–2 sentences, max 80 tokens) that captures this fact
for long-term memory. Be specific, include names and context.
Return ONLY the entry text, no preamble.

FACT: {fact}"""

_UPDATE_L3_PROMPT = """\
Update this memory entry by incorporating the new fact.
Return ONLY the updated entry (1–3 sentences, max 120 tokens). No preamble.

EXISTING ENTRY:
{existing}

NEW FACT TO ADD:
{new_fact}

Updated entry:"""


async def run(new_l4_nodes: list[MemoryNode]) -> list[MemoryNode]:
    """
    Consolidate each new L4 node into L3.
    Returns list of new or updated L3 nodes.
    """
    store    = get_store()
    affected: list[MemoryNode] = []

    for l4_node in new_l4_nodes:
        if not l4_node.embedding:
            continue

        # Find best matching active L3 node for the same topic
        candidates = await store.asearch_layer(
            l4_node.embedding, layer=3, limit=3
        )

        if not candidates:
            # No existing L3 — create a new one
            l3 = await _create_l3(l4_node)
            affected.append(l3)
            continue

        best = candidates[0]
        label = await _classify(l4_node.content, best.content)

        if label == ConsolidationLabel.CONTRADICTS:
            l3 = await _handle_contradicts(l4_node, best)
            affected.append(l3)

        elif label == ConsolidationLabel.ADDS:
            l3 = await _handle_adds(l4_node, best)
            affected.append(l3)

        elif label == ConsolidationLabel.REINFORCES:
            await _handle_reinforces(best)

        else:  # IRRELEVANT — create independent L3
            l3 = await _create_l3(l4_node)
            affected.append(l3)

    return affected


async def _classify(new_fact: str, existing: str) -> str:
    prompt = _CLASSIFY_PROMPT.format(new_fact=new_fact, existing=existing)
    raw    = await fast().generate(prompt, temperature=0.0)
    word   = raw.strip().upper().split()[0] if raw.strip() else "IRRELEVANT"
    valid  = {ConsolidationLabel.CONTRADICTS, ConsolidationLabel.ADDS,
              ConsolidationLabel.REINFORCES, ConsolidationLabel.IRRELEVANT}
    return word if word in valid else ConsolidationLabel.IRRELEVANT


async def _create_l3(l4_node: MemoryNode) -> MemoryNode:
    store  = get_store()
    prompt = _CREATE_L3_PROMPT.format(fact=l4_node.content)
    text   = await strong().generate(prompt)
    embed  = await embedder().embed(text)

    l3 = MemoryNode(
        layer           = 3,
        content         = text,
        embedding       = embed,
        importance      = l4_node.importance,
        topic_tags      = l4_node.topic_tags,
        confidence      = "high",
        source_pointers = [l4_node.node_id],
        dirty           = False,
    )
    l3.compute_retention()
    wid = await store.ainsert(l3)

    # graph edges
    await store.aadd_edge(wid,        "sourcedFrom",  l4_node.id)
    await store.aadd_edge(l4_node.id, "summarizedBy", wid)

    return l3


async def _handle_contradicts(l4_node: MemoryNode, old_l3: MemoryNode) -> MemoryNode:
    store = get_store()

    # Archive old L3
    await store.aupdate_props(old_l3.id, status="archived", dirty=False)

    # Create new L3 with updated content
    prompt = _CREATE_L3_PROMPT.format(fact=l4_node.content)
    text   = await strong().generate(prompt)
    embed  = await embedder().embed(text)

    new_l3 = MemoryNode(
        layer           = 3,
        content         = text,
        embedding       = embed,
        importance      = max(l4_node.importance, old_l3.importance),
        topic_tags      = list(set(l4_node.topic_tags + old_l3.topic_tags)),
        confidence      = "high",
        source_pointers = [l4_node.node_id],
        dirty           = True,    # propagate upward — L2 needs refresh
    )
    new_l3.compute_retention()
    wid = await store.ainsert(new_l3)

    # graph edges: new contradicts old, sourced from L4
    await store.aadd_edge(wid,      "contradicts", old_l3.id)
    await store.aadd_edge(wid,      "sourcedFrom", l4_node.id)

    # carry over pointer from old L3 to its L2 parent so compressor knows what to refresh
    for pid in old_l3.derived_by:
        await store.aadd_edge(wid, "summarizedBy", pid)

    return new_l3


async def _handle_adds(l4_node: MemoryNode, l3_node: MemoryNode) -> MemoryNode:
    store  = get_store()
    prompt = _UPDATE_L3_PROMPT.format(
        existing=l3_node.content, new_fact=l4_node.content
    )
    new_text = await strong().generate(prompt)
    new_embed = await embedder().embed(new_text)

    await store.aupdate_props(
        l3_node.id,
        content        = new_text,
        dirty          = True,
        importance     = max(l3_node.importance, l4_node.importance),
        topicTags      = list(set(l3_node.topic_tags + l4_node.topic_tags)),
    )
    # update vector
    # Weaviate v4: update vector by re-inserting is not ideal; use update with vector
    # We'll store the updated embedding via a helper
    await _update_vector(l3_node.id, new_embed)

    await store.aadd_edge(l3_node.id, "sourcedFrom", l4_node.id)

    l3_node.content  = new_text
    l3_node.dirty    = True
    return l3_node


async def _handle_reinforces(l3_node: MemoryNode):
    store = get_store()
    from datetime import datetime, timezone
    await store.aupdate_props(
        l3_node.id,
        lastAccessed = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        accessCount  = l3_node.access_count + 1,
    )


async def _update_vector(weaviate_id: str, vector: list[float]):
    """Update just the vector of an existing node."""
    import asyncio
    store = get_store()
    def _do():
        store._col().data.update(uuid=weaviate_id, vector=vector)
    await asyncio.to_thread(_do)
