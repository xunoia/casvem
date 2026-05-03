"""
CaSVeM FastAPI application.

Endpoints:
  POST /session           → submit conversation → triggers write pipeline
  POST /query             → submit query → runs read pipeline → returns answer
  GET  /memory/{layer}    → list memory nodes at a layer
  GET  /memory/node/{id}  → get a specific node
  GET  /status            → system health + counts
  POST /admin/consolidate → manually trigger full write pipeline flush
  POST /admin/promote     → manually trigger promotion/demotion cycle
"""

from __future__ import annotations
import asyncio
import logging

from fastapi import FastAPI, HTTPException, BackgroundTasks

from models import (
    SessionRequest, QueryRequest, QueryResponse,
    MemoryListResponse, StatusResponse,
)
from database.weaviate_store import get_store
from providers.router import fast, strong, embedder
import scheduler as sched

log = logging.getLogger("casvem.api")

app = FastAPI(
    title       = "CaSVeM — Cached Smart Vector Memory",
    description = "Hierarchical graph + vector memory for local LLMs",
    version     = "1.0.0",
)


# ── Session (write pipeline) ───────────────────────────────────────────────────

@app.post("/session", summary="Submit a conversation session")
async def submit_session(req: SessionRequest, bg: BackgroundTasks):
    """
    Save the session and trigger the full write pipeline asynchronously.
    Returns immediately — pipeline runs in background.
    """
    bg.add_task(_run_write_pipeline, req.session_id, req.user_id, req.transcript)
    return {"session_id": req.session_id, "status": "queued"}


async def _run_write_pipeline(session_id: str, user_id: str, transcript: str):
    from engines.write import extractor, consolidator, compressor
    try:
        log.info("Write pipeline start: session=%s", session_id)
        new_l4 = await extractor.run(session_id, user_id, transcript)
        log.info("Extracted %d L4 facts", len(new_l4))

        new_l3 = await consolidator.run(new_l4)
        log.info("Consolidated into %d L3 nodes", len(new_l3))

        await compressor.run_l3_to_l2()
        await compressor.run_l2_to_l1()
        log.info("Write pipeline complete: session=%s", session_id)
    except Exception as e:
        log.error("Write pipeline error: %s", e, exc_info=True)


async def _run_lazy_promotion(l5_weaviate_id: str, user_id: str, transcript: str):
    """Re-extract facts from an existing L5 node and promote them up the hierarchy."""
    from engines.write import extractor, consolidator, compressor
    try:
        log.info("Lazy promotion start: l5_id=%s", l5_weaviate_id)
        new_l4 = await extractor.run(
            l5_weaviate_id, user_id, transcript,
            existing_l5_id=l5_weaviate_id,
        )
        log.info("Lazy promotion: extracted %d new L4 facts", len(new_l4))
        if new_l4:
            new_l3 = await consolidator.run(new_l4)
            log.info("Lazy promotion: consolidated into %d L3 nodes", len(new_l3))
            await compressor.run_l3_to_l2()
            await compressor.run_l2_to_l1()
        log.info("Lazy promotion complete: l5_id=%s", l5_weaviate_id)
    except Exception as e:
        log.error("Lazy promotion error: %s", e, exc_info=True)


# ── Query (read pipeline) ──────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse, summary="Query memory")
async def query_memory(req: QueryRequest, bg: BackgroundTasks):
    from engines.read import analyser, searcher, synthesiser

    intent       = await analyser.run(req.query)
    search_result= await searcher.run(req.query, intent)
    memory_block = await synthesiser.synthesise(req.query, search_result)
    routing      = await synthesiser.route(
        req.query, memory_block,
        search_result.confidence,
        len(search_result.l2_nodes) + len(search_result.l3_nodes),
    )
    final_answer = await synthesiser.answer(req.query, memory_block)

    # Lazy promotion: if we fell through to L5, re-extract and promote facts
    # from those raw transcripts up to L4→L3→L2→L1 in the background.
    if search_result.l5_nodes:
        for node in search_result.l5_nodes:
            bg.add_task(_run_lazy_promotion, node.id, req.user_id, node.content)
            log.info("Lazy promotion queued for L5 node %s", node.node_id)

    return QueryResponse(
        query        = req.query,
        memory_block = memory_block,
        answer       = final_answer,
        confidence   = search_result.confidence,
        routed_to    = routing.lower(),
        layers_hit   = search_result.layers_hit,
    )


# ── Memory inspection ──────────────────────────────────────────────────────────

@app.get("/memory/{layer}", response_model=MemoryListResponse)
async def list_memory(layer: int):
    if layer not in range(1, 6):
        raise HTTPException(400, "Layer must be 1–5")
    store = get_store()
    nodes = await store.aget_layer(layer)
    return MemoryListResponse(layer=layer, count=len(nodes), nodes=nodes)


@app.get("/memory/node/{node_id}", summary="Get a specific memory node")
async def get_node(node_id: str):
    store = get_store()
    node  = await store.aget_by_node_id(node_id)
    if node is None:
        node = await store.aget(node_id)  # fallback: try Weaviate UUID
    if node is None:
        raise HTTPException(404, f"Node {node_id!r} not found")
    return node


# ── Status ─────────────────────────────────────────────────────────────────────

@app.get("/status", response_model=StatusResponse)
async def status():
    store = get_store()

    weaviate_ok = False
    try:
        store._client.is_ready()
        weaviate_ok = True
    except Exception:
        pass

    ollama_ok = False
    try:
        await embedder().embed("health check")
        ollama_ok = True
    except Exception:
        pass

    counts = await store.aget_all_counts()

    return StatusResponse(
        weaviate_ok      = weaviate_ok,
        ollama_ok        = ollama_ok,
        layer_counts     = counts,
        scheduler_running= sched.is_running(),
    )


# ── Admin ──────────────────────────────────────────────────────────────────────

@app.post("/admin/consolidate", summary="Manually trigger write pipeline compression")
async def admin_consolidate(bg: BackgroundTasks):
    async def _do():
        from engines.write import compressor
        await compressor.run_l3_to_l2()
        await compressor.run_l2_to_l1()
    bg.add_task(_do)
    return {"status": "consolidation queued"}


@app.post("/admin/promote", summary="Manually trigger promotion/demotion cycle")
async def admin_promote(bg: BackgroundTasks):
    bg.add_task(sched.trigger_now)
    return {"status": "promotion cycle queued"}


@app.post("/admin/reset", summary="Wipe ALL memory — benchmark use only")
async def admin_reset():
    """Delete every MemoryNode and recreate the schema. Used by benchmark harness."""
    store = get_store()
    await store.adelete_all()
    log.warning("Memory WIPED by /admin/reset")
    return {"status": "reset", "message": "all memory nodes deleted"}
