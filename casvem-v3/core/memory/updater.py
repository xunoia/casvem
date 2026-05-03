"""
Memory Updater — the flywheel.

cache_writeback():    called after every cold-path query → writes result to cache
invalidate():         called on memory edit/delete → evicts stale cache entries
decay_old_entries():  called on schedule → applies confidence decay
queue_mlp_retrain():  called when log grows enough → retrains MLP predictor
"""

from typing import Optional
import numpy as np

from config import cfg
from core.cache import cache_gate
from core.storage import get_storage


def cache_writeback(
    query_hash: str,
    query_text: str,
    query_vec: np.ndarray,
    memory_ids: list[str],
):
    """Write cold-path result to cache. Increment access counts on matched memories."""
    storage = get_storage()

    existing = storage.get_cache_entry(query_hash)
    hit_count = existing["hit_count"] if existing else 0

    cache_gate.write(
        query_hash=query_hash,
        query_text=query_text,
        query_vec=query_vec,
        memory_ids=memory_ids,
        hit_count=hit_count,
    )

    for mid in memory_ids:
        storage.increment_access_count(mid)


def invalidate_memory(memory_id: str):
    """
    A memory was edited or deleted. Remove it from all cache entries that reference it.
    Called by DELETE /memory/{id} and any future edit endpoint.
    """
    storage = get_storage()
    storage.invalidate_cache_entries_for_memory(memory_id)

    # Also remove from bitmap
    from core.memory.writer import get_bitmap
    mem = storage.get_memory(memory_id)
    if mem:
        get_bitmap().remove(mem["hnsw_label"])
        storage.delete_memory(memory_id)


def decay_old_entries():
    """
    Apply confidence decay to memories older than 30 days.
    Should be called once daily via a scheduler or background task.
    """
    get_storage().decay_confidence(
        decay_rate=cfg.cache_confidence_decay,
        older_than_days=30,
    )


def queue_mlp_retrain():
    """
    Trigger async MLP retrain if access_log has grown enough.
    Called from the query pipeline after logging.
    Non-blocking: runs in a thread so it doesn't slow down responses.
    """
    if not cfg.use_mlp:
        return

    from core.cache.access_log import get_access_log
    log = get_access_log()

    if log.should_retrain_mlp():
        import threading
        from core.cache.mlp_predictor import get_mlp_predictor

        def retrain():
            get_mlp_predictor().train()
            log.mark_retrained()

        threading.Thread(target=retrain, daemon=True).start()
