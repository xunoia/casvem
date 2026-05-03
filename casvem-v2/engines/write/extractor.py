"""
Write pipeline — Step 1 & 2
  L5: save raw session transcript (append-only)
  L4: extract atomic facts from the transcript via LLM
"""

from __future__ import annotations
import json
import math
from datetime import datetime, timezone

from models import MemoryNode
from database import get_store
from providers.router import strong, embedder
from config import DEDUP_SIMILARITY_THRESHOLD

_EXTRACT_PROMPT = """\
Extract every factual statement from the conversation below.
Return ONLY a valid JSON array. No preamble, no explanation.

Each item in the array must have exactly these keys:
  "fact"        : the atomic fact as a self-contained sentence (no pronouns without antecedent)
  "importance"  : float 0.0–1.0 (how memorable / significant is this?)
  "topic_tags"  : list of strings (e.g. ["work", "preferences", "people", "projects", "finance"])

Rules:
  - Each fact must stand alone — include the subject explicitly
  - Extract only facts, not questions, greetings, or filler
  - Deduplicate: if the same fact appears twice, include it once
  - Assign higher importance to decisions, preferences, and named entities
  - Assign lower importance to casual remarks

TRANSCRIPT:
{transcript}

JSON array:"""


async def run(session_id: str, user_id: str, transcript: str) -> list[MemoryNode]:
    """
    1. Save raw session to L5.
    2. Extract facts → L4 nodes.
    Returns the list of new L4 nodes inserted.
    """
    store = get_store()

    # ── L5: save raw session ───────────────────────────────────────────────────
    l5_embed = await embedder().embed(transcript[:2000])  # embed first 2k chars as summary
    l5_node  = MemoryNode(
        layer       = 5,
        content     = transcript,
        embedding   = l5_embed,
        importance  = 0.5,
        topic_tags  = ["session"],
        confidence  = "high",
        status      = "active",
    )
    l5_id = await store.ainsert(l5_node)

    # ── L4: extract facts via LLM ─────────────────────────────────────────────
    prompt   = _EXTRACT_PROMPT.format(transcript=transcript)
    raw_json = await strong().generate_json(prompt)

    try:
        facts = json.loads(raw_json)
        if not isinstance(facts, list):
            facts = []
    except (json.JSONDecodeError, ValueError):
        facts = []

    new_nodes: list[MemoryNode] = []

    for item in facts:
        if not isinstance(item, dict):
            continue
        fact_text  = str(item.get("fact", "")).strip()
        importance = float(item.get("importance", 0.5))
        tags       = list(item.get("topic_tags", []))

        if not fact_text:
            continue

        # embed the fact
        fact_embed = await embedder().embed(fact_text)

        # dedup: check cosine similarity against existing active L4 nodes with same tags
        if await _is_duplicate(fact_embed, tags):
            continue

        node = MemoryNode(
            layer          = 4,
            content        = fact_text,
            embedding      = fact_embed,
            importance     = max(0.0, min(1.0, importance)),
            topic_tags     = tags,
            confidence     = "high",
            source_pointers= [l5_node.node_id],
        )
        node.compute_retention()
        wid = await store.ainsert(node)

        # graph edge: L4 node ──sourcedFrom──► L5 node
        await store.aadd_edge(wid, "sourcedFrom", l5_id)

        new_nodes.append(node)

    return new_nodes


async def _is_duplicate(embedding: list[float], tags: list[str]) -> bool:
    """Return True if a very similar fact already exists in L4."""
    store    = get_store()
    existing = await store.asearch_layer(embedding, layer=4, limit=5)

    for node in existing:
        if node.embedding:
            sim = _cosine(embedding, node.embedding)
            if sim >= DEDUP_SIMILARITY_THRESHOLD:
                return True
    return False


def _cosine(a: list[float], b: list[float]) -> float:
    dot   = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    return dot / (mag_a * mag_b) if mag_a * mag_b > 0 else 0.0
