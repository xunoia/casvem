"""
Write pipeline — Step 4
  L3 → L2 compression  (dirty L3 nodes trigger L2 re-summarisation)
  L2 → L1 compression  (dirty L2 nodes trigger L1 re-summarisation)

Runs async in background — never blocks reads.
"""

from __future__ import annotations
from itertools import groupby

from models import MemoryNode
from database.weaviate_store import get_store
from providers.router import strong, embedder
from config import LAYER_CONFIG

_L3_TO_L2_PROMPT = """\
Compress the following related facts into a concise topic summary.
Write exactly 2–3 sentences. Maximum 100 tokens.
Be specific — keep names, numbers, and decisions.
Return ONLY the summary text, no preamble.

TOPIC: {topic}

FACTS:
{facts}

Summary:"""

_L2_TO_L1_PROMPT = """\
Compress this summary into a single key fact. Maximum 20 words.
Return ONLY the compressed fact, no preamble.

SUMMARY: {summary}

Key fact:"""


async def run_l3_to_l2():
    """Find all dirty L3 nodes and compress their topic groups into L2."""
    store       = get_store()
    dirty_l3    = await store.aget_dirty(layer=3)

    if not dirty_l3:
        return

    # Group dirty L3 nodes by their primary topic tag
    def _primary_tag(node: MemoryNode) -> str:
        return node.topic_tags[0] if node.topic_tags else "general"

    dirty_l3.sort(key=_primary_tag)
    for topic, group in groupby(dirty_l3, key=_primary_tag):
        group_nodes = list(group)
        await _compress_topic_to_l2(topic, group_nodes)


async def run_l2_to_l1():
    """Find all dirty L2 nodes and compress them into L1."""
    store    = get_store()
    dirty_l2 = await store.aget_dirty(layer=2)

    if not dirty_l2:
        return

    for l2_node in dirty_l2:
        await _compress_l2_to_l1(l2_node)


async def _compress_topic_to_l2(topic: str, l3_nodes: list[MemoryNode]):
    store = get_store()
    cfg   = LAYER_CONFIG[2]

    facts_text = "\n".join(f"- {n.content}" for n in l3_nodes)
    prompt     = _L3_TO_L2_PROMPT.format(topic=topic, facts=facts_text)
    summary    = await strong().generate(prompt)
    embed      = await embedder().embed(summary)

    avg_importance = sum(n.importance for n in l3_nodes) / len(l3_nodes)
    all_tags       = list({t for n in l3_nodes for t in n.topic_tags})

    # Check if an L2 node for this topic already exists (via pointer from first L3 node)
    existing_l2: MemoryNode | None = None
    if l3_nodes[0].derived_by:
        existing_l2 = await store.aget(l3_nodes[0].derived_by[0])

    if existing_l2 and existing_l2.status == "active":
        # Update existing L2
        await store.aupdate_props(
            existing_l2.id,
            content        = summary,
            dirty          = True,          # will cascade to L1
            importance     = avg_importance,
            topicTags      = all_tags,
        )
        await _update_vector(existing_l2.id, embed)
        l2_id = existing_l2.id
    else:
        # Create new L2
        l2_node = MemoryNode(
            layer           = 2,
            content         = summary,
            embedding       = embed,
            importance      = avg_importance,
            topic_tags      = all_tags,
            confidence      = "high",
            source_pointers = [n.node_id for n in l3_nodes],
            dirty           = True,
        )
        l2_node.compute_retention()
        l2_id = await store.ainsert(l2_node)

        # graph edges: L2 ──sourcedFrom──► each L3
        for l3 in l3_nodes:
            await store.aadd_edge(l2_id, "sourcedFrom", l3.id)

    # Mark all L3 nodes as clean and point them at L2
    for l3 in l3_nodes:
        await store.aupdate_props(l3.id, dirty=False)
        await store.aadd_edge(l3.id, "summarizedBy", l2_id)

    # Enforce L2 capacity limit
    await _enforce_layer_limit(layer=2, max_lines=cfg["max_lines"])


async def _compress_l2_to_l1(l2_node: MemoryNode):
    store = get_store()
    cfg   = LAYER_CONFIG[1]

    prompt   = _L2_TO_L1_PROMPT.format(summary=l2_node.content)
    key_fact = await strong().generate(prompt)
    embed    = await embedder().embed(key_fact)

    # Check if a matching L1 node exists
    existing_l1 = await store.asearch_layer(embed, layer=1, limit=1)

    if existing_l1:
        l1 = existing_l1[0]
        await store.aupdate_props(
            l1.id,
            content    = key_fact,
            dirty      = False,
            importance = max(l1.importance, l2_node.importance),
        )
        await _update_vector(l1.id, embed)
    else:
        l1_node = MemoryNode(
            layer           = 1,
            content         = key_fact,
            embedding       = embed,
            importance      = l2_node.importance,
            topic_tags      = l2_node.topic_tags,
            confidence      = "high",
            source_pointers = [l2_node.node_id],
            dirty           = False,
        )
        l1_node.compute_retention()
        l1_id = await store.ainsert(l1_node)
        await store.aadd_edge(l1_id, "sourcedFrom", l2_node.id)

    # Mark L2 clean
    await store.aupdate_props(l2_node.id, dirty=False)

    # Enforce L1 capacity
    await _enforce_layer_limit(layer=1, max_lines=cfg["max_lines"])


async def _enforce_layer_limit(layer: int, max_lines: int):
    """Demote lowest-retention nodes when a layer exceeds its line limit."""
    store = get_store()
    nodes = await store.aget_layer(layer)
    if len(nodes) <= max_lines:
        return

    for node in nodes:
        node.compute_retention()

    nodes.sort(key=lambda n: n.retention_score)
    to_demote = nodes[:len(nodes) - max_lines]

    for node in to_demote:
        await store.aupdate_props(node.id, status="archived")


async def _update_vector(weaviate_id: str, vector: list[float]):
    import asyncio
    store = get_store()
    def _do():
        store._col().data.update(uuid=weaviate_id, vector=vector)
    await asyncio.to_thread(_do)
