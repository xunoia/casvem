"""
Read pipeline — Steps 2–4
Hierarchical search:
  1. L1 auto-inject (always)
  2. L2 vector search using query intent
  3. Sufficiency check (LLM)
  4. If not sufficient → follow source_pointers into L3
  5. If still not sufficient → follow into L4 (rare)
"""

from __future__ import annotations
from dataclasses import dataclass

from models import MemoryNode
from database import get_store
from providers.router import fast, embedder
from engines.read.analyser import QueryIntent
from config import L2_SEARCH_LIMIT, L3_POINTER_LIMIT

_SUFFICIENCY_PROMPT = """\
Are these memory results sufficient to answer the query completely?
Reply with ONLY one word: SUFFICIENT or DEEPER

QUERY: {query}

RETRIEVED MEMORIES:
{memories}

Reply:"""


@dataclass
class SearchResult:
    l1_nodes: list[MemoryNode]
    l2_nodes: list[MemoryNode]
    l3_nodes: list[MemoryNode]
    l4_nodes: list[MemoryNode]
    layers_hit: list[int]
    confidence: str


async def run(query: str, intent: QueryIntent) -> SearchResult:
    store = get_store()

    # ── Step 1: L1 auto-inject ─────────────────────────────────────────────────
    l1_nodes = await store.aget_layer(layer=1, status="active", limit=20)
    layers_hit = [1] if l1_nodes else []

    # ── Step 2: L2 vector search ───────────────────────────────────────────────
    l2_nodes: list[MemoryNode] = []
    for term in intent.search_terms[:3]:  # use top 3 search terms
        term_embed = await embedder().embed(term)
        hits = await store.asearch_layer(term_embed, layer=2, limit=L2_SEARCH_LIMIT)
        for h in hits:
            if h.id not in {n.id for n in l2_nodes}:
                l2_nodes.append(h)

    # Boost nodes matching topic tags
    l2_nodes = _rank_by_topic(l2_nodes, intent.topics)

    if l2_nodes:
        layers_hit.append(2)

    # ── Step 3: Sufficiency check ─────────────────────────────────────────────
    if l2_nodes:
        sufficient = await _check_sufficiency(query, l2_nodes[:5])
    else:
        sufficient = False

    l3_nodes: list[MemoryNode] = []
    l4_nodes: list[MemoryNode] = []

    if not sufficient:
        # ── Step 4a: Follow source_pointers L2 → L3 ──────────────────────────
        l3_nodes = await _follow_to_l3(l2_nodes[:3])

        # ── Step 4b: L2 empty or no graph hits → search L3 directly ──────────
        if not l3_nodes:
            for term in intent.search_terms[:3]:
                term_embed = await embedder().embed(term)
                hits = await store.asearch_layer(term_embed, layer=3, limit=L2_SEARCH_LIMIT)
                for h in hits:
                    if h.id not in {n.id for n in l3_nodes}:
                        l3_nodes.append(h)
            l3_nodes = _rank_by_topic(l3_nodes, intent.topics)

        if l3_nodes:
            layers_hit.append(3)

        # ── Step 4c: L4 fallback when L3 is also sparse ───────────────────────
        if len(l3_nodes) < 2:
            l4_nodes = await _follow_to_l4(l3_nodes[:2])
            if not l4_nodes:
                for term in intent.search_terms[:2]:
                    term_embed = await embedder().embed(term)
                    hits = await store.asearch_layer(term_embed, layer=4, limit=5)
                    for h in hits:
                        if h.id not in {n.id for n in l4_nodes}:
                            l4_nodes.append(h)
            if l4_nodes:
                layers_hit.append(4)

    # ── Update access counts ───────────────────────────────────────────────────
    all_accessed = l2_nodes[:5] + l3_nodes[:5] + l4_nodes[:3]
    await _bump_access(all_accessed)

    confidence = _compute_confidence(l1_nodes, l2_nodes, l3_nodes)

    return SearchResult(
        l1_nodes   = l1_nodes,
        l2_nodes   = l2_nodes[:5],
        l3_nodes   = l3_nodes,
        l4_nodes   = l4_nodes,
        layers_hit = layers_hit,
        confidence = confidence,
    )


async def _check_sufficiency(query: str, nodes: list[MemoryNode]) -> bool:
    memories_text = "\n".join(f"- {n.content}" for n in nodes)
    prompt = _SUFFICIENCY_PROMPT.format(query=query, memories=memories_text)
    raw    = await fast().generate(prompt, temperature=0.0)
    return "SUFFICIENT" in raw.upper()


async def _follow_to_l3(l2_nodes: list[MemoryNode]) -> list[MemoryNode]:
    store   = get_store()
    l3_hits: list[MemoryNode] = []
    seen = set()

    for l2 in l2_nodes:
        # Follow graph edge: L2 ──sourcedFrom──► L3
        neighbours = await store.aget_neighbours(l2.id, "sourcedFrom")
        for n in neighbours:
            if n.layer == 3 and n.status == "active" and n.id not in seen:
                l3_hits.append(n)
                seen.add(n.id)
                if len(l3_hits) >= L3_POINTER_LIMIT * len(l2_nodes):
                    break

    return l3_hits[:L3_POINTER_LIMIT * 3]


async def _follow_to_l4(l3_nodes: list[MemoryNode]) -> list[MemoryNode]:
    store   = get_store()
    l4_hits: list[MemoryNode] = []
    seen = set()

    for l3 in l3_nodes:
        neighbours = await store.aget_neighbours(l3.id, "sourcedFrom")
        for n in neighbours:
            if n.layer == 4 and n.status == "active" and n.id not in seen:
                l4_hits.append(n)
                seen.add(n.id)

    return l4_hits[:10]


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


def _compute_confidence(l1, l2, l3) -> str:
    total = len(l1) + len(l2) + len(l3)
    if total == 0:
        return "low"
    if l1 and l2 and len(l2) >= 3:
        return "high"
    if l2:
        return "medium"
    return "low"
