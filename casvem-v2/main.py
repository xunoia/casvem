from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, BackgroundTasks

from config import OLLAMA_BASE_URL, LLM_BACKEND, VECTOR_STORE_BACKEND
from models import (
    SessionRequest, QueryRequest, QueryResponse,
    MemoryListResponse, StatusResponse,
)
from providers.router import init_providers, close_providers
from database import init_store, close_store, get_store
from engines.write import extractor, consolidator, compressor
from engines.read  import analyser, searcher, synthesiser


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_store()
    init_providers()
    yield
    await close_providers()
    close_store()

app = FastAPI(title="CaSVeM Memory API", version="2.0", lifespan=lifespan)


@app.get("/status", response_model=StatusResponse)
async def get_status() -> StatusResponse:
    store = get_store()

    try:
        counts = store.get_all_counts()
        vs_ok  = True
    except Exception:
        counts = {f"L{i}": 0 for i in range(1, 6)}
        vs_ok  = False

    try:
        async with httpx.AsyncClient(timeout=3) as client:
            r = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
        ollama_ok = r.status_code == 200
    except Exception:
        ollama_ok = False

    return StatusResponse(
        vector_store_ok   = vs_ok,
        ollama_ok         = ollama_ok,
        layer_counts      = counts,
        scheduler_running = False,
    )


@app.post("/session")
async def process_session(req: SessionRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(
        _write_pipeline, req.session_id, req.user_id, req.transcript
    )
    return {"session_id": req.session_id, "status": "queued"}


async def _write_pipeline(session_id: str, user_id: str, transcript: str):
    new_l4 = await extractor.run(session_id, user_id, transcript)
    await consolidator.run(new_l4)


@app.get("/memory/{layer}", response_model=MemoryListResponse)
async def get_memory(layer: int) -> MemoryListResponse:
    nodes = await get_store().aget_layer(layer)
    return MemoryListResponse(layer=layer, count=len(nodes), nodes=nodes)


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest) -> QueryResponse:
    intent  = await analyser.run(req.query)
    result  = await searcher.run(req.query, intent)
    mem_blk = await synthesiser.synthesise(req.query, result)
    routing = await synthesiser.route(
        req.query, mem_blk, result.confidence,
        len(result.l1_nodes) + len(result.l2_nodes),
    )
    ans = await synthesiser.answer(req.query, mem_blk)
    return QueryResponse(
        query        = req.query,
        memory_block = mem_blk,
        answer       = ans,
        confidence   = result.confidence,
        routed_to    = routing,
        layers_hit   = result.layers_hit,
    )


@app.post("/admin/consolidate")
async def admin_consolidate():
    await compressor.run_l3_to_l2()
    return {"status": "ok"}


@app.post("/admin/promote")
async def admin_promote():
    await compressor.run_l2_to_l1()
    return {"status": "ok"}


@app.post("/admin/reset")
async def admin_reset():
    """Delete all vectors from Pinecone — use before each clean test run."""
    import asyncio
    store = get_store()
    await asyncio.to_thread(_do_reset, store)
    return {"status": "reset complete"}


def _do_reset(store):
    from config import VECTOR_STORE_BACKEND
    import time
    if VECTOR_STORE_BACKEND == "pinecone":
        store._index.delete(delete_all=True)
        # Pinecone delete_all propagates asynchronously — poll until index is empty
        for _ in range(60):
            stats = store._index.describe_index_stats()
            if stats.total_vector_count == 0:
                return
            time.sleep(1)
    else:
        raise RuntimeError("reset only implemented for pinecone backend")
