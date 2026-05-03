"""
Full query pipeline.

Flow:
  encode → cache_gate.check
    → HIT:  log + record_hit + return cached memories as context
    → MISS: bitmap_filter → hnsw_search → rerank → context_build
            → llm.complete → cache_writeback → log → return
"""

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from config import cfg
from core.cache import cache_gate
from core.cache.access_log import get_access_log
from core.context.builder import build_context, build_prompt
from core.encoder import get_encoder
from core.llm import CompletionResult, get_llm_provider
from core.memory.updater import cache_writeback, queue_mlp_retrain
from core.memory.writer import get_bitmap
from core.retrieval.reranker import get_reranker
from core.storage import get_storage


@dataclass
class QueryResult:
    answer: str
    context: str
    memories: list[dict]
    hit_type: str          # 'L1', 'L2', or 'cold'
    latency_ms: float
    input_tokens: int
    output_tokens: int
    query_hash: str


async def query(
    text: str,
    memory_type: Optional[str] = None,
    project_id: Optional[str] = None,
    author_id: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> QueryResult:
    t0 = time.perf_counter()
    encoder = get_encoder()
    storage = get_storage()

    query_vec = encoder.encode(text)

    # ── Cache check ───────────────────────────────────────────────────────────
    cache_result = cache_gate.check(query_vec, text)

    if cache_result.hit:
        memories = storage.get_memories_by_ids(cache_result.memory_ids)
        context = build_context(text, memories)

        latency_ms = (time.perf_counter() - t0) * 1000
        get_access_log().log(
            query_hash=cache_result.query_hash,
            memory_ids=cache_result.memory_ids,
            hit_type=cache_result.tier,
            latency_ms=latency_ms,
            input_tokens=0,
            output_tokens=0,
        )
        cache_gate.record_hit(cache_result.query_hash, query_vec)

        return QueryResult(
            answer=context,       # cached path: context IS the answer (no LLM call)
            context=context,
            memories=memories,
            hit_type=cache_result.tier,
            latency_ms=latency_ms,
            input_tokens=0,
            output_tokens=0,
            query_hash=cache_result.query_hash,
        )

    # ── Cold path: bitmap → hnsw → rerank → context → llm ────────────────────
    bitmap = get_bitmap()
    candidate_labels = bitmap.filter(
        memory_type=memory_type,
        project_id=project_id,
        author_id=author_id,
        date_from=date_from,
        date_to=date_to,
    )

    # Convert hnsw_labels to memory_ids for filtering
    candidate_ids: Optional[set] = None
    if candidate_labels is not None:
        label_to_id = storage._label_to_id
        candidate_ids = {label_to_id[lbl] for lbl in candidate_labels if lbl in label_to_id}

    hnsw_results = storage.search_hnsw(query_vec, k=cfg.top_k, candidate_ids=candidate_ids)

    if not hnsw_results:
        latency_ms = (time.perf_counter() - t0) * 1000
        return QueryResult(
            answer="I don't have any relevant memories for this query.",
            context="",
            memories=[],
            hit_type="cold",
            latency_ms=latency_ms,
            input_tokens=0,
            output_tokens=0,
            query_hash=cache_result.query_hash,
        )

    memory_ids = [mid for mid, _ in hnsw_results]
    memories_raw = storage.get_memories_by_ids(memory_ids)

    # Attach hnsw scores before reranking
    score_map = dict(hnsw_results)
    for m in memories_raw:
        m["hnsw_score"] = score_map.get(m["id"], 0.0)

    reranked = get_reranker().rerank(text, memories_raw, top_n=cfg.top_n)
    context = build_context(text, reranked)
    prompt = build_prompt(text, context)

    completion: CompletionResult = await get_llm_provider().complete(prompt)

    latency_ms = (time.perf_counter() - t0) * 1000
    used_ids = [m["id"] for m in reranked]

    # ── Writeback + log ───────────────────────────────────────────────────────
    cache_writeback(
        query_hash=cache_result.query_hash,
        query_text=text,
        query_vec=query_vec,
        memory_ids=used_ids,
    )
    get_access_log().log(
        query_hash=cache_result.query_hash,
        memory_ids=used_ids,
        hit_type="cold",
        latency_ms=latency_ms,
        input_tokens=completion.input_tokens,
        output_tokens=completion.output_tokens,
    )
    queue_mlp_retrain()

    return QueryResult(
        answer=completion.text,
        context=context,
        memories=reranked,
        hit_type="cold",
        latency_ms=latency_ms,
        input_tokens=completion.input_tokens,
        output_tokens=completion.output_tokens,
        query_hash=cache_result.query_hash,
    )
