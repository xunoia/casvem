"""
Read pipeline — Steps 2–4
Hierarchical vector search across all layers:
  1. L1 auto-inject (always)
  2. L2 vector search (compressed summaries)
  3. L3 vector search (consolidated facts)
  4. Sufficiency check
  5. If not sufficient → L4 vector search (raw extracted facts)
"""

from __future__ import annotations
from dataclasses import dataclass

from models import MemoryNode
from database.weaviate_store import get_store
from providers.router import embedder
from engines.read.analyser import QueryIntent
from config import L2_SEARCH_LIMIT, L3_POINTER_LIMIT


@dataclass
class SearchResult:
    l1_nodes: list[MemoryNode]
    l2_nodes: list[MemoryNode]
    l3_nodes: list[MemoryNode]
    l4_nodes: list[MemoryNode]
    l5_nodes: list[MemoryNode]
    layers_hit: list[int]
    confidence: str


async def run(query: str, intent: QueryIntent) -> SearchResult:
    store = get_store()

    # ── Step 1: L1 auto-inject ─────────────────────────────────────────────────
    l1_nodes = await store.aget_layer(layer=1, status="active", limit=20)
    layers_hit = [1] if l1_nodes else []

    # Build query embedding once for all vector searches
    query_embed = await embedder().embed(query)

    # Also embed each search term for broader coverage
    term_embeds = []
    for term in intent.search_terms[:3]:
        term_embeds.append(await embedder().embed(term))

    # ── Step 2: L2 vector search ───────────────────────────────────────────────
    l2_nodes = await _multi_search(store, [query_embed] + term_embeds, layer=2, limit=L2_SEARCH_LIMIT)
    l2_nodes = _rank_by_topic(l2_nodes, intent.topics)
    if l2_nodes:
        layers_hit.append(2)

    # ── Step 3: L3 direct vector search ───────────────────────────────────────
    l3_nodes = await _multi_search(store, [query_embed] + term_embeds, layer=3, limit=L3_POINTER_LIMIT * 3)
    l3_nodes = _rank_by_topic(l3_nodes, intent.topics)
    if l3_nodes:
        layers_hit.append(3)

    # ── Step 4: L4 direct vector search (always — facts may not have compressed up yet) ──
    l4_nodes = await _multi_search(store, [query_embed] + term_embeds, layer=4, limit=10)
    l4_nodes = _rank_by_topic(l4_nodes, intent.topics)
    if l4_nodes:
        layers_hit.append(4)

    # ── Step 5: L5 fallback — search raw sessions if L2–L4 found nothing ──────
    # L5 transcripts may contain info not yet extracted by the write pipeline.
    l5_nodes: list[MemoryNode] = []
    content_found = len(l2_nodes) + len(l3_nodes) + len(l4_nodes)
    if content_found == 0:
        l5_nodes = await _multi_search(store, [query_embed] + term_embeds, layer=5, limit=3)
        if l5_nodes:
            layers_hit.append(5)

    # ── Update access counts ───────────────────────────────────────────────────
    all_accessed = l2_nodes[:5] + l3_nodes[:5] + l4_nodes[:5]
    await _bump_access(all_accessed)

    confidence = _compute_confidence(l1_nodes, l2_nodes, l3_nodes, l4_nodes, l5_nodes)

    return SearchResult(
        l1_nodes   = l1_nodes,
        l2_nodes   = l2_nodes[:5],
        l3_nodes   = l3_nodes[:8],
        l4_nodes   = l4_nodes[:8],
        l5_nodes   = l5_nodes[:3],
        layers_hit = sorted(set(layers_hit)),
        confidence = confidence,
    )


async def _multi_search(
    store, embeds: list[list[float]], layer: int, limit: int
) -> list[MemoryNode]:
    """Search a layer with multiple query embeddings, deduplicated by id."""
    seen: dict[str, MemoryNode] = {}
    per_embed = max(limit // len(embeds), 3)
    for embed in embeds:
        hits = await store.asearch_layer(embed, layer=layer, limit=per_embed)
        for n in hits:
            if n.id not in seen:
                seen[n.id] = n
    return list(seen.values())



async def _bump_access(nodes: list[MemoryNode]):
    store = get_store()
    from datetime import datetime, timezone
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for node in nodes:
        await store.aupdate_props(
            node.id,
            accessCount  = node.access_count + 1,
            lastAccessed = now_str,
        )


def _rank_by_topic(nodes: list[MemoryNode], topics: list[str]) -> list[MemoryNode]:
    topic_set = set(topics)
    def _score(n: MemoryNode) -> float:
        tag_overlap = len(set(n.topic_tags) & topic_set)
        return n.retention_score + tag_overlap * 0.1
    return sorted(nodes, key=_score, reverse=True)


def _compute_confidence(l1, l2, l3, l4, l5) -> str:
    if l5 and not (l2 or l3 or l4):
        return "low"   # raw transcript fallback
    total = len(l1) + len(l2) + len(l3) + len(l4)
    if total == 0:
        return "low"
    if (l1 or l2) and (l3 or l4):
        return "high"
    if l2 or l3:
        return "medium"
    return "low"
