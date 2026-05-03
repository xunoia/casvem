import hashlib
import json
from typing import Optional

import numpy as np
from cachetools import LRUCache

from config import cfg


def _quantize(vec: np.ndarray, decimals: int) -> bytes:
    """Round vector to `decimals` decimal places and convert to bytes for hashing."""
    return np.round(vec, decimals).astype(np.float32).tobytes()


def make_exact_key(vec: np.ndarray) -> str:
    return hashlib.sha256(_quantize(vec, 2)).hexdigest()


def make_semantic_key(vec: np.ndarray) -> str:
    return hashlib.sha256(_quantize(vec, 1)).hexdigest()


class LRUCacheGate:
    """
    Two-level LRU cache.
    L1 (hot):  access_count > cfg.cache_l1_threshold → served in ~0.1ms
    L2 (warm): access_count > cfg.cache_l2_threshold → served in ~0.5ms

    Both exact (2 decimal) and semantic (1 decimal) keys are checked.
    Opt 1: verify_hit() validates cosine similarity before returning a result.
    """

    def __init__(self):
        self._l1: LRUCache = LRUCache(maxsize=cfg.cache_l1_maxsize)
        self._l2: LRUCache = LRUCache(maxsize=cfg.cache_l2_maxsize)

    def check(self, query_vec: np.ndarray) -> tuple[Optional[str], Optional[str]]:
        """
        Returns (tier, query_hash) if found, else (None, exact_key).
        Tier is 'L1' or 'L2'.
        Always returns the exact_key so callers can write back.
        """
        exact_key = make_exact_key(query_vec)
        semantic_key = make_semantic_key(query_vec)

        for key in (exact_key, semantic_key):
            if key in self._l1:
                return "L1", key
            if key in self._l2:
                return "L2", key

        return None, exact_key

    def get_memory_ids(self, query_hash: str) -> Optional[list[str]]:
        if query_hash in self._l1:
            return self._l1[query_hash]
        if query_hash in self._l2:
            return self._l2[query_hash]
        return None

    def write(self, query_hash: str, memory_ids: list[str], tier: str = "L2"):
        if tier == "L1":
            self._l1[query_hash] = memory_ids
        else:
            self._l2[query_hash] = memory_ids

    def promote(self, query_hash: str):
        """Move an entry from L2 to L1."""
        if query_hash in self._l2:
            ids = self._l2.pop(query_hash)
            self._l1[query_hash] = ids

    def invalidate(self, query_hash: str):
        self._l1.pop(query_hash, None)
        self._l2.pop(query_hash, None)

    def clear(self):
        self._l1.clear()
        self._l2.clear()

    def verify_hit(self, query_vec: np.ndarray, stored_vec_bytes: bytes) -> bool:
        """
        Opt 1: cosine similarity check between current query and cached query vector.
        Both vectors are L2-normalized → dot product = cosine similarity.
        Returns False (treat as miss) if similarity < threshold.
        """
        stored_vec = np.frombuffer(stored_vec_bytes, dtype=np.float32)
        similarity = float(np.dot(query_vec, stored_vec))
        return similarity >= cfg.cache_hit_similarity_threshold
