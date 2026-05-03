"""
Unified Cache Gate.

Phase 1 Week 1-2: LRU eviction (USE_MLP=false).
Phase 1 Week 3-4: MLP predictor replaces LRU (USE_MLP=true).

The rest of the pipeline calls only this module — never lru_cache or mlp_predictor directly.
Switching from LRU to MLP is a one-line .env change.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from config import cfg
from core.cache.lru_cache import LRUCacheGate, make_exact_key
from core.storage import get_storage


@dataclass
class CacheCheckResult:
    hit: bool
    tier: Optional[str]       # 'L1', 'L2', or None
    query_hash: str
    memory_ids: Optional[list[str]]


_lru = LRUCacheGate()


def check(query_vec: np.ndarray, query_text: str) -> CacheCheckResult:
    """
    Check if query_vec is in cache.
    Opt 1: verify cosine similarity before accepting hit.
    """
    tier, query_hash = _lru.check(query_vec)

    if tier is not None:
        memory_ids = _lru.get_memory_ids(query_hash)
        if memory_ids is not None:
            # Opt 1: verify similarity via stored query_vec in SQLite
            entry = get_storage().get_cache_entry(query_hash)
            if entry and _lru.verify_hit(query_vec, entry["query_vec"]):
                return CacheCheckResult(hit=True, tier=tier, query_hash=query_hash,
                                        memory_ids=memory_ids)
            # Similarity check failed — treat as miss, fall through
            _lru.invalidate(query_hash)

    return CacheCheckResult(hit=False, tier=None, query_hash=query_hash, memory_ids=None)


def write(
    query_hash: str,
    query_text: str,
    query_vec: np.ndarray,
    memory_ids: list[str],
    hit_count: int = 0,
):
    """Write a result to cache after a cold-path query."""
    tier = _tier_for_count(hit_count)
    _lru.write(query_hash, memory_ids, tier)
    get_storage().set_cache_entry(
        query_hash=query_hash,
        query_text=query_text,
        query_vec=query_vec,
        memory_ids=memory_ids,
        tier=tier,
    )

    if cfg.use_mlp:
        from core.cache.mlp_predictor import get_mlp_predictor
        predictor = get_mlp_predictor()
        if predictor.should_pre_populate():
            # Pre-populate: write at higher tier
            _lru.write(query_hash, memory_ids, "L2")


def record_hit(query_hash: str, query_vec: np.ndarray):
    """Increment hit count and promote to L1 if threshold crossed."""
    storage = get_storage()
    entry = storage.get_cache_entry(query_hash)
    if entry:
        new_count = entry["hit_count"] + 1
        if new_count >= cfg.cache_l1_threshold:
            _lru.promote(query_hash)
            storage.update_cache_entry_tier(query_hash, "L1")


def invalidate(query_hash: str):
    _lru.invalidate(query_hash)
    # SQLite entry remains for training data; only in-memory LRU is cleared


def reset_for_benchmark():
    """Wipe in-memory LRU state between benchmark items."""
    _lru.clear()


def _tier_for_count(hit_count: int) -> str:
    if hit_count >= cfg.cache_l1_threshold:
        return "L1"
    if hit_count >= cfg.cache_l2_threshold:
        return "L2"
    return "L2"  # all new entries start at L2
