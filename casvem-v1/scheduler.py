"""
Promotion / Demotion / Merge engine.
Runs every SCHEDULER_INTERVAL_MINUTES. Never blocks reads or writes.

Promotion:  retention_score > layer.promote  → move line up one layer
Demotion:   retention_score < layer.demote   → move line down one layer
Merge:      cosine_similarity > 0.88 between two lines in the same layer
            → LLM decides if they are the same fact → merge if yes
"""

from __future__ import annotations
import math
import asyncio
import logging

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from models import MemoryNode
from database.weaviate_store import get_store
from providers.router import fast, strong, embedder
from config import LAYER_CONFIG, SCHEDULER_INTERVAL_MINUTES, MERGE_SIMILARITY_THRESHOLD

log = logging.getLogger("casvem.scheduler")

_scheduler: AsyncIOScheduler | None = None
_running = False


def start():
    global _scheduler
    _scheduler = AsyncIOScheduler()
    _scheduler.add_job(
        _run_cycle,
        trigger  = "interval",
        minutes  = SCHEDULER_INTERVAL_MINUTES,
        id       = "promotion_cycle",
        max_instances = 1,
    )
    _scheduler.start()
    log.info("Scheduler started — interval: %d min", SCHEDULER_INTERVAL_MINUTES)


def stop():
    if _scheduler:
        _scheduler.shutdown(wait=False)


def is_running() -> bool:
    return _scheduler is not None and _scheduler.running


async def trigger_now():
    """Manually trigger a full cycle (used by admin endpoint)."""
    await _run_cycle()


# ── Main cycle ─────────────────────────────────────────────────────────────────

async def _run_cycle():
    global _running
    if _running:
        return
    _running = True
    try:
        log.info("Promotion/demotion cycle starting...")
        for layer in range(1, 5):   # L1–L4; L5 is append-only
            await _process_layer(layer)
        # Merge duplicates in L2 and L3 (most impactful layers)
        for layer in (2, 3):
            await _merge_layer(layer)
        # Run compression for any dirty nodes created during this cycle
        from engines.write import compressor
        await compressor.run_l3_to_l2()
        await compressor.run_l2_to_l1()
        log.info("Promotion/demotion cycle complete.")
    finally:
        _running = False


async def _process_layer(layer: int):
    store  = get_store()
    cfg    = LAYER_CONFIG[layer]
    nodes  = await store.aget_layer(layer)

    for node in nodes:
        node.compute_retention()

    promoted = demoted = 0

    for node in nodes:
        score = node.retention_score

        if layer < 4 and score > cfg["promote"]:
            await _promote(node, layer)
            promoted += 1

        elif layer > 1 and score < cfg["demote"]:
            await _demote(node, layer)
            demoted += 1

    if promoted or demoted:
        log.info("L%d: promoted=%d, demoted=%d", layer, promoted, demoted)


async def _promote(node: MemoryNode, from_layer: int):
    """Move a high-scoring node one layer up (compress + insert)."""
    store      = get_store()
    target_cfg = LAYER_CONFIG[from_layer - 1]

    # Check target layer capacity
    if await _layer_full(from_layer - 1, target_cfg["max_lines"]):
        # Swap with the lowest-scoring node in the target layer
        await _swap_for_promotion(node, from_layer - 1)
        return

    # Create a compressed version at the target layer
    compressed = await _compress_for_layer(node.content, from_layer - 1)
    embed      = await embedder().embed(compressed)

    new_node = MemoryNode(
        layer           = from_layer - 1,
        content         = compressed,
        embedding       = embed,
        importance      = node.importance,
        topic_tags      = node.topic_tags,
        confidence      = node.confidence,
        source_pointers = [node.node_id],
        dirty           = False,
    )
    new_node.compute_retention()
    new_id = await store.ainsert(new_node)
    await store.aadd_edge(new_id, "sourcedFrom", node.id)

    await store.aupdate_props(node.id, status="archived")


async def _demote(node: MemoryNode, from_layer: int):
    """Move a cold node one layer down."""
    store      = get_store()
    target_cfg = LAYER_CONFIG[from_layer + 1]

    # Verify source still exists in the layer below
    sources = await store.aget_neighbours(node.id, "sourcedFrom")
    if not sources:
        # No source — just archive rather than dangling demotion
        await store.aupdate_props(node.id, status="archived")
        return

    await store.aupdate_props(node.id, status="archived")


async def _swap_for_promotion(new_node: MemoryNode, target_layer: int):
    store  = get_store()
    nodes  = await store.aget_layer(target_layer)
    if not nodes:
        return

    for n in nodes:
        n.compute_retention()
    lowest = min(nodes, key=lambda n: n.retention_score)

    cfg    = LAYER_CONFIG[target_layer]
    if new_node.retention_score > lowest.retention_score:
        await store.aupdate_props(lowest.id, status="archived")
        await _promote(new_node, target_layer + 1)


async def _layer_full(layer: int, max_lines: int) -> bool:
    if max_lines < 0:
        return False
    store = get_store()
    count = store.count_layer(layer)
    return count >= max_lines


# ── Merge engine ───────────────────────────────────────────────────────────────

_MERGE_PROMPT = """\
Are these two memory entries describing the same fact?
Reply with ONLY one word: YES or NO

Entry A: {a}
Entry B: {b}

Reply:"""

_COMBINE_PROMPT = """\
Merge these two memory entries into one, combining all information.
Return ONLY the merged entry (1–3 sentences). No preamble.

Entry A: {a}
Entry B: {b}

Merged:"""


async def _merge_layer(layer: int):
    store = get_store()
    nodes = await store.aget_layer(layer)
    if len(nodes) < 2:
        return

    merged_ids: set[str] = set()

    for i, node_a in enumerate(nodes):
        if node_a.id in merged_ids or not node_a.embedding:
            continue
        for node_b in nodes[i + 1:]:
            if node_b.id in merged_ids or not node_b.embedding:
                continue

            sim = _cosine(node_a.embedding, node_b.embedding)
            if sim < MERGE_SIMILARITY_THRESHOLD:
                continue

            # Ask LLM if they're truly the same fact
            prompt  = _MERGE_PROMPT.format(a=node_a.content, b=node_b.content)
            verdict = await fast().generate(prompt, temperature=0.0)
            if "YES" not in verdict.upper():
                continue

            # Merge
            merge_prompt = _COMBINE_PROMPT.format(a=node_a.content, b=node_b.content)
            merged_text  = await strong().generate(merge_prompt)
            merged_embed = await embedder().embed(merged_text)

            await store.aupdate_props(
                node_a.id,
                content     = merged_text,
                accessCount = node_a.access_count + node_b.access_count,
                importance  = max(node_a.importance, node_b.importance),
                dirty       = True,
            )
            await _update_vector(node_a.id, merged_embed)
            await store.aupdate_props(node_b.id, status="archived")
            merged_ids.add(node_b.id)
            log.debug("Merged L%d nodes: %s ← %s", layer, node_a.node_id, node_b.node_id)
            break


async def _compress_for_layer(content: str, target_layer: int) -> str:
    if target_layer == 1:
        from engines.write.compressor import _L2_TO_L1_PROMPT
        prompt = _L2_TO_L1_PROMPT.format(summary=content)
    else:
        from engines.write.compressor import _L3_TO_L2_PROMPT
        prompt = _L3_TO_L2_PROMPT.format(topic="promotion", facts=content)
    return await strong().generate(prompt)


def _cosine(a: list[float], b: list[float]) -> float:
    dot   = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    return dot / (mag_a * mag_b) if mag_a * mag_b > 0 else 0.0


async def _update_vector(weaviate_id: str, vector: list[float]):
    store = get_store()
    def _do():
        store._col().data.update(uuid=weaviate_id, vector=vector)
    await asyncio.to_thread(_do)
