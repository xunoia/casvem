from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from core.memory.updater import invalidate_memory
from core.storage import get_storage
from pipeline.ingest import ingest
from pipeline.query import query

router = APIRouter()


# ── Request / Response models ─────────────────────────────────────────────────

class IngestRequest(BaseModel):
    text: str
    memory_type: str = "fact"
    project_id: Optional[str] = None
    author_id: Optional[str] = None


class IngestResponse(BaseModel):
    id: str


class QueryRequest(BaseModel):
    text: str
    memory_type: Optional[str] = None
    project_id: Optional[str] = None
    author_id: Optional[str] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    hit_type: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    memory_count: int


class StatsResponse(BaseModel):
    total_queries: int
    cache_hits: int
    cache_misses: int
    hit_rate: float
    cold_input_tokens: int
    cold_output_tokens: int
    memory_count: int
    cache_entries: int


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/memory", response_model=IngestResponse)
async def add_memory(req: IngestRequest):
    memory_id = ingest(
        text=req.text,
        memory_type=req.memory_type,
        project_id=req.project_id,
        author_id=req.author_id,
    )
    return IngestResponse(id=memory_id)


@router.post("/query", response_model=QueryResponse)
async def query_memories(req: QueryRequest):
    result = await query(
        text=req.text,
        memory_type=req.memory_type,
        project_id=req.project_id,
        author_id=req.author_id,
        date_from=req.date_from,
        date_to=req.date_to,
    )
    return QueryResponse(
        answer=result.answer,
        hit_type=result.hit_type,
        latency_ms=round(result.latency_ms, 2),
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        memory_count=len(result.memories),
    )


@router.get("/stats", response_model=StatsResponse)
async def get_stats():
    return StatsResponse(**get_storage().get_stats())


@router.delete("/memory/{memory_id}")
async def delete_memory(memory_id: str):
    mem = get_storage().get_memory(memory_id)
    if not mem:
        raise HTTPException(status_code=404, detail="Memory not found")
    invalidate_memory(memory_id)
    return {"ok": True, "id": memory_id}
