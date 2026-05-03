import json
import os
import sqlite3
import time
import uuid
from functools import lru_cache
from typing import Optional

import numpy as np
from usearch.index import Index as USearchIndex

from config import cfg


class Storage:

    def __init__(self, sqlite_path: str, hnsw_path: str, dim: int):
        os.makedirs(os.path.dirname(sqlite_path) or ".", exist_ok=True)
        self._db_path = sqlite_path
        self._hnsw_path = hnsw_path
        self._dim = dim

        self._conn = sqlite3.connect(sqlite_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

        self._hnsw = self._load_or_create_hnsw()
        # In-memory label → memory_id map for O(1) lookup
        self._label_to_id: dict[int, str] = self._build_label_map()
        self._next_label: int = max(self._label_to_id.keys(), default=-1) + 1

    # ── Schema ────────────────────────────────────────────────────────────────

    def _init_schema(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id            TEXT PRIMARY KEY,
                text          TEXT NOT NULL,
                vector        BLOB NOT NULL,
                memory_type   TEXT DEFAULT 'fact',
                project_id    TEXT,
                author_id     TEXT,
                created_at    INTEGER NOT NULL,
                confidence    REAL DEFAULT 1.0,
                access_count  INTEGER DEFAULT 0,
                hnsw_label    INTEGER NOT NULL UNIQUE
            );
            CREATE TABLE IF NOT EXISTS access_log (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                query_hash    TEXT NOT NULL,
                memory_ids    TEXT NOT NULL,
                hit_type      TEXT NOT NULL,
                latency_ms    REAL NOT NULL,
                input_tokens  INTEGER NOT NULL DEFAULT 0,
                output_tokens INTEGER NOT NULL DEFAULT 0,
                created_at    INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS cache_entries (
                query_hash    TEXT PRIMARY KEY,
                query_text    TEXT NOT NULL,
                query_vec     BLOB NOT NULL,
                memory_ids    TEXT NOT NULL,
                tier          TEXT NOT NULL,
                hit_count     INTEGER DEFAULT 0,
                created_at    INTEGER NOT NULL,
                last_hit      INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_memories_type    ON memories(memory_type);
            CREATE INDEX IF NOT EXISTS idx_memories_project ON memories(project_id);
            CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);
            CREATE INDEX IF NOT EXISTS idx_access_log_hash  ON access_log(query_hash);
            CREATE INDEX IF NOT EXISTS idx_access_log_time  ON access_log(created_at);
        """)
        self._conn.commit()

    # ── HNSW ──────────────────────────────────────────────────────────────────

    def _load_or_create_hnsw(self) -> USearchIndex:
        idx = USearchIndex(
            ndim=self._dim,
            metric="cos",
            connectivity=cfg.hnsw_m,
            expansion_add=cfg.hnsw_ef_construction,
            expansion_search=cfg.hnsw_ef_search,
        )
        if os.path.exists(self._hnsw_path):
            idx.load(self._hnsw_path)
        return idx

    def _build_label_map(self) -> dict[int, str]:
        rows = self._conn.execute("SELECT hnsw_label, id FROM memories").fetchall()
        return {row["hnsw_label"]: row["id"] for row in rows}

    def _save_hnsw(self):
        os.makedirs(os.path.dirname(self._hnsw_path) or ".", exist_ok=True)
        self._hnsw.save(self._hnsw_path)

    # ── Memories ──────────────────────────────────────────────────────────────

    def add_memory(
        self,
        text: str,
        vector: np.ndarray,
        memory_type: str = "fact",
        project_id: Optional[str] = None,
        author_id: Optional[str] = None,
    ) -> str:
        memory_id = str(uuid.uuid4())
        # Re-read max label from DB each time — safe when multiple processes share the file
        row = self._conn.execute("SELECT COALESCE(MAX(hnsw_label), -1) FROM memories").fetchone()
        label = int(row[0]) + 1
        self._next_label = label + 1

        self._conn.execute(
            """INSERT INTO memories
               (id, text, vector, memory_type, project_id, author_id, created_at, hnsw_label)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                memory_id, text, vector.tobytes(), memory_type,
                project_id, author_id, int(time.time()), label,
            ),
        )
        self._conn.commit()

        self._hnsw.add(np.array([label], dtype=np.uint64), vector.reshape(1, -1))
        self._label_to_id[label] = memory_id

        if self._next_label % 100 == 0:
            self._save_hnsw()

        return memory_id

    def get_memory(self, memory_id: str) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        d["vector"] = np.frombuffer(d["vector"], dtype=np.float32)
        return d

    def get_memories_by_ids(self, ids: list[str]) -> list[dict]:
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        rows = self._conn.execute(
            f"SELECT * FROM memories WHERE id IN ({placeholders})", ids
        ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["vector"] = np.frombuffer(d["vector"], dtype=np.float32)
            result.append(d)
        return result

    def delete_memory(self, memory_id: str):
        self._conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self._conn.commit()

    def increment_access_count(self, memory_id: str):
        self._conn.execute(
            "UPDATE memories SET access_count = access_count + 1 WHERE id = ?",
            (memory_id,),
        )
        self._conn.commit()

    def decay_confidence(self, decay_rate: float, older_than_days: int = 30):
        cutoff = int(time.time()) - older_than_days * 86400
        self._conn.execute(
            "UPDATE memories SET confidence = confidence * ? WHERE created_at < ?",
            (1.0 - decay_rate, cutoff),
        )
        self._conn.commit()

    # ── HNSW Search ───────────────────────────────────────────────────────────

    def search_hnsw(
        self,
        query_vec: np.ndarray,
        k: int = 50,
        candidate_ids: Optional[set] = None,
    ) -> list[tuple[str, float]]:
        """Returns [(memory_id, score)] sorted by score descending."""
        if len(self._hnsw) == 0:
            return []

        # Over-fetch when post-filtering so we have enough after intersection
        fetch_k = min(k * 5 if candidate_ids else k, len(self._hnsw))

        matches = self._hnsw.search(query_vec.reshape(1, -1), fetch_k)

        results = []
        for label, distance in zip(matches.keys.flatten(), matches.distances.flatten()):
            mem_id = self._label_to_id.get(int(label))
            if mem_id is None:
                continue
            if candidate_ids is not None and mem_id not in candidate_ids:
                continue
            score = 1.0 - float(distance)  # cosine distance → similarity
            results.append((mem_id, score))
            if len(results) >= k:
                break

        return results

    # ── Access Log ────────────────────────────────────────────────────────────

    def log_access(
        self,
        query_hash: str,
        memory_ids: list[str],
        hit_type: str,
        latency_ms: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ):
        self._conn.execute(
            """INSERT INTO access_log
               (query_hash, memory_ids, hit_type, latency_ms, input_tokens, output_tokens, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                query_hash, json.dumps(memory_ids), hit_type,
                latency_ms, input_tokens, output_tokens, int(time.time()),
            ),
        )
        self._conn.commit()

    def count_access_log(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM access_log").fetchone()[0]

    def get_access_log_for_training(self, limit: int = 10_000) -> list[dict]:
        rows = self._conn.execute(
            "SELECT query_hash, hit_type, created_at FROM access_log ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Cache Entries ─────────────────────────────────────────────────────────

    def get_cache_entry(self, query_hash: str) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT * FROM cache_entries WHERE query_hash = ?", (query_hash,)
        ).fetchone()
        return dict(row) if row else None

    def set_cache_entry(
        self,
        query_hash: str,
        query_text: str,
        query_vec: np.ndarray,
        memory_ids: list[str],
        tier: str,
    ):
        now = int(time.time())
        self._conn.execute(
            """INSERT INTO cache_entries
               (query_hash, query_text, query_vec, memory_ids, tier, created_at, last_hit)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(query_hash) DO UPDATE SET
                   hit_count = hit_count + 1,
                   last_hit  = excluded.last_hit,
                   tier      = excluded.tier""",
            (
                query_hash, query_text, query_vec.tobytes(),
                json.dumps(memory_ids), tier, now, now,
            ),
        )
        self._conn.commit()

    def update_cache_entry_tier(self, query_hash: str, tier: str):
        self._conn.execute(
            "UPDATE cache_entries SET tier = ? WHERE query_hash = ?", (tier, query_hash)
        )
        self._conn.commit()

    def invalidate_cache_entries_for_memory(self, memory_id: str):
        """Remove all cache entries that reference this memory_id."""
        rows = self._conn.execute(
            "SELECT query_hash, memory_ids FROM cache_entries"
        ).fetchall()
        to_delete = []
        for row in rows:
            ids = json.loads(row["memory_ids"])
            if memory_id in ids:
                to_delete.append(row["query_hash"])
        if to_delete:
            placeholders = ",".join("?" * len(to_delete))
            self._conn.execute(
                f"DELETE FROM cache_entries WHERE query_hash IN ({placeholders})", to_delete
            )
            self._conn.commit()

    # ── Stats ─────────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        total = self._conn.execute("SELECT COUNT(*) FROM access_log").fetchone()[0]
        hits = self._conn.execute(
            "SELECT COUNT(*) FROM access_log WHERE hit_type != 'cold'"
        ).fetchone()[0]
        row = self._conn.execute(
            "SELECT SUM(input_tokens), SUM(output_tokens) FROM access_log WHERE hit_type = 'cold'"
        ).fetchone()
        cold_input_tokens = row[0] or 0
        cold_output_tokens = row[1] or 0

        return {
            "total_queries": total,
            "cache_hits": hits,
            "cache_misses": total - hits,
            "hit_rate": hits / total if total > 0 else 0.0,
            "cold_input_tokens": cold_input_tokens,
            "cold_output_tokens": cold_output_tokens,
            "memory_count": self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0],
            "cache_entries": self._conn.execute("SELECT COUNT(*) FROM cache_entries").fetchone()[0],
        }

    def reset_for_benchmark(self):
        """
        Wipe all memories and cache entries between benchmark items.
        Rebuilds a fresh usearch index in-memory without touching disk.
        Called by benchmark runners between dataset items.
        """
        self._conn.execute("DELETE FROM memories")
        self._conn.execute("DELETE FROM cache_entries")
        self._conn.commit()
        self._label_to_id.clear()
        self._next_label = 0
        self._hnsw = USearchIndex(
            ndim=self._dim,
            metric="cos",
            connectivity=cfg.hnsw_m,
            expansion_add=cfg.hnsw_ef_construction,
            expansion_search=cfg.hnsw_ef_search,
        )

    def close(self):
        self._save_hnsw()
        self._conn.close()


@lru_cache(maxsize=1)
def get_storage() -> Storage:
    from core.encoder import get_encoder
    return Storage(
        sqlite_path=cfg.sqlite_path,
        hnsw_path=cfg.hnsw_index_path,
        dim=get_encoder().dim,
    )
