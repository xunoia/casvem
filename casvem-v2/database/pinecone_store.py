"""
Pinecone vector store for CaSVeM.

Every MemoryNode is a Pinecone vector with:
  - values: the embedding from Ollama
  - metadata: all node properties + serialised graph edges

Graph edges are simulated via metadata lists:
  sourcedFrom_ids  → lower-layer nodes this was derived from
  summarizedBy_ids → higher-layer nodes that summarise this
  chainPrev_ids    → previous node in a chain
  chainNext_ids    → next node in a chain
  contradicts_ids  → archived node this one contradicts
"""

from __future__ import annotations
import json
import asyncio
import math
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

from models import MemoryNode, ChainMembership
from config import (
    PINECONE_API_KEY, PINECONE_CLOUD, PINECONE_REGION,
    PINECONE_INDEX, EMBEDDING_DIM,
)

# Dummy unit vector used for "scan" queries (get_layer / count_layer).
_DUMMY = [1.0 / math.sqrt(EMBEDDING_DIM)] * EMBEDDING_DIM

# Pinecone metadata field names for each graph edge type.
_EDGE_FIELD = {
    "sourcedFrom":  "sourcedFrom_ids",
    "summarizedBy": "summarizedBy_ids",
    "chainPrev":    "chainPrev_ids",
    "chainNext":    "chainNext_ids",
    "contradicts":  "contradicts_ids",
}

# Pinecone metadata is limited to ~40 KB per vector; truncate long content.
_MAX_CONTENT = 8_000


class PineconeStore:
    def __init__(self):
        self._pc    = None
        self._index = None

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def connect(self):
        from pinecone import Pinecone, ServerlessSpec
        self._pc = Pinecone(api_key=PINECONE_API_KEY)

        if PINECONE_INDEX not in self._pc.list_indexes().names():
            self._pc.create_index(
                name=PINECONE_INDEX,
                dimension=EMBEDDING_DIM,
                metric="cosine",
                spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
            )
            for _ in range(90):
                if self._pc.describe_index(PINECONE_INDEX).status["ready"]:
                    break
                time.sleep(1)

        self._index = self._pc.Index(PINECONE_INDEX)

    def close(self):
        self._pc    = None
        self._index = None

    # ── CRUD ───────────────────────────────────────────────────────────────────

    def insert(self, node: MemoryNode) -> str:
        wid = str(uuid.uuid4())
        self._index.upsert(vectors=[{
            "id":       wid,
            "values":   node.embedding or ([0.0] * EMBEDDING_DIM),
            "metadata": _to_meta(node),
        }])
        node.id = wid
        return wid

    def get(self, wid: str) -> Optional[MemoryNode]:
        resp = self._index.fetch(ids=[wid])
        vecs = resp.vectors
        if not vecs or wid not in vecs:
            return None
        v = vecs[wid]
        return _from_meta(wid, v.metadata, getattr(v, "values", None))

    def get_by_node_id(self, node_id: str) -> Optional[MemoryNode]:
        resp = self._index.query(
            vector=_DUMMY,
            top_k=1,
            filter={"nodeId": {"$eq": node_id}},
            include_metadata=True,
            include_values=False,
        )
        if not resp.matches:
            return None
        m = resp.matches[0]
        return _from_meta(m.id, m.metadata, None)

    def update_props(self, wid: str, **props):
        self._index.update(id=wid, set_metadata=props)

    def update_vector(self, wid: str, vector: list[float]):
        self._index.update(id=wid, values=vector)

    def delete(self, wid: str):
        self._index.delete(ids=[wid])

    # ── Graph edges ────────────────────────────────────────────────────────────

    def add_edge(self, from_id: str, edge_name: str, to_id: str):
        field = _EDGE_FIELD.get(edge_name)
        if not field:
            return
        resp = self._index.fetch(ids=[from_id])
        if not resp.vectors or from_id not in resp.vectors:
            return
        current = list(resp.vectors[from_id].metadata.get(field) or [])
        if to_id not in current:
            current.append(to_id)
            self._index.update(id=from_id, set_metadata={field: current})

    def remove_edge(self, from_id: str, edge_name: str, to_id: str):
        field = _EDGE_FIELD.get(edge_name)
        if not field:
            return
        resp = self._index.fetch(ids=[from_id])
        if not resp.vectors or from_id not in resp.vectors:
            return
        current = [x for x in (resp.vectors[from_id].metadata.get(field) or []) if x != to_id]
        self._index.update(id=from_id, set_metadata={field: current})

    def get_neighbours(self, from_id: str, edge_name: str) -> list[MemoryNode]:
        field = _EDGE_FIELD.get(edge_name)
        if not field:
            return []
        resp = self._index.fetch(ids=[from_id])
        if not resp.vectors or from_id not in resp.vectors:
            return []
        ids = list(resp.vectors[from_id].metadata.get(field) or [])
        if not ids:
            return []
        fetched = self._index.fetch(ids=ids)
        nodes = []
        for nid in ids:
            if nid in fetched.vectors:
                v = fetched.vectors[nid]
                nodes.append(_from_meta(nid, v.metadata, getattr(v, "values", None)))
        return nodes

    # ── Vector search ──────────────────────────────────────────────────────────

    def search_layer(
        self,
        vector: list[float],
        layer: int,
        limit: int = 10,
        status: str = "active",
    ) -> list[MemoryNode]:
        resp = self._index.query(
            vector=vector,
            top_k=limit,
            filter={"layer": {"$eq": layer}, "status": {"$eq": status}},
            include_metadata=True,
            include_values=True,
        )
        return [_from_meta(m.id, m.metadata, m.values) for m in resp.matches]

    def search_by_tags(self, tags: list[str], layer: int, limit: int = 20) -> list[MemoryNode]:
        filt: dict = {"layer": {"$eq": layer}}
        if tags:
            filt["topicTags"] = {"$in": tags}
        resp = self._index.query(
            vector=_DUMMY,
            top_k=limit,
            filter=filt,
            include_metadata=True,
            include_values=False,
        )
        return [_from_meta(m.id, m.metadata, None) for m in resp.matches]

    # ── Layer management ────────────────────────────────────────────────────────

    def count_layer(self, layer: int) -> int:
        resp = self._index.query(
            vector=_DUMMY,
            top_k=10_000,
            filter={"layer": {"$eq": layer}},
            include_metadata=False,
            include_values=False,
        )
        return len(resp.matches)

    def get_layer(self, layer: int, status: str = "active", limit: int = 500) -> list[MemoryNode]:
        resp = self._index.query(
            vector=_DUMMY,
            top_k=limit,
            filter={"layer": {"$eq": layer}, "status": {"$eq": status}},
            include_metadata=True,
            include_values=False,
        )
        return [_from_meta(m.id, m.metadata, None) for m in resp.matches]

    def get_dirty(self, layer: int) -> list[MemoryNode]:
        resp = self._index.query(
            vector=_DUMMY,
            top_k=500,
            filter={
                "layer":  {"$eq": layer},
                "dirty":  {"$eq": True},
                "status": {"$eq": "active"},
            },
            include_metadata=True,
            include_values=False,
        )
        return [_from_meta(m.id, m.metadata, None) for m in resp.matches]

    def get_all_counts(self) -> dict[str, int]:
        return {f"L{layer}": self.count_layer(layer) for layer in range(1, 6)}

    # ── Async wrappers ─────────────────────────────────────────────────────────

    async def ainsert(self, node: MemoryNode) -> str:
        return await asyncio.to_thread(self.insert, node)

    async def aget(self, wid: str) -> Optional[MemoryNode]:
        return await asyncio.to_thread(self.get, wid)

    async def aget_by_node_id(self, node_id: str) -> Optional[MemoryNode]:
        return await asyncio.to_thread(self.get_by_node_id, node_id)

    async def aupdate_props(self, wid: str, **props):
        await asyncio.to_thread(self.update_props, wid, **props)

    async def aupdate_vector(self, wid: str, vector: list[float]):
        await asyncio.to_thread(self.update_vector, wid, vector)

    async def aadd_edge(self, from_id: str, edge: str, to_id: str):
        await asyncio.to_thread(self.add_edge, from_id, edge, to_id)

    async def aremove_edge(self, from_id: str, edge: str, to_id: str):
        await asyncio.to_thread(self.remove_edge, from_id, edge, to_id)

    async def aget_neighbours(self, wid: str, edge: str) -> list[MemoryNode]:
        return await asyncio.to_thread(self.get_neighbours, wid, edge)

    async def asearch_layer(self, vector, layer, limit=10, status="active") -> list[MemoryNode]:
        return await asyncio.to_thread(self.search_layer, vector, layer, limit, status)

    async def aget_layer(self, layer, status="active", limit=500) -> list[MemoryNode]:
        return await asyncio.to_thread(self.get_layer, layer, status, limit)

    async def aget_dirty(self, layer: int) -> list[MemoryNode]:
        return await asyncio.to_thread(self.get_dirty, layer)

    async def aget_all_counts(self) -> dict[str, int]:
        return await asyncio.to_thread(self.get_all_counts)


# ── Singleton ──────────────────────────────────────────────────────────────────

_store: PineconeStore | None = None


def get_store() -> PineconeStore:
    if _store is None:
        raise RuntimeError("Store not initialised — call init_store() first.")
    return _store


def init_store() -> PineconeStore:
    global _store
    _store = PineconeStore()
    _store.connect()
    return _store


def close_store():
    if _store:
        _store.close()


# ── Serialisation helpers ──────────────────────────────────────────────────────

def _to_meta(node: MemoryNode) -> dict:
    def _fmt(dt: datetime) -> str:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    return {
        "nodeId":               node.node_id,
        "layer":                node.layer,
        "content":              node.content[:_MAX_CONTENT],
        "retentionScore":       node.retention_score,
        "importance":           node.importance,
        "recency":              node.recency,
        "accessCount":          node.access_count,
        "lastAccessed":         _fmt(node.last_accessed),
        "createdAt":            _fmt(node.created_at),
        "dirty":                node.dirty,
        "status":               node.status,
        "topicTags":            node.topic_tags,
        "confidence":           node.confidence,
        "chainMembershipsJson": json.dumps([m.model_dump() for m in node.chain_memberships]),
        # Graph edge lists (empty on insert; populated via add_edge)
        "sourcedFrom_ids":  [],
        "summarizedBy_ids": [],
        "chainPrev_ids":    [],
        "chainNext_ids":    [],
        "contradicts_ids":  [],
    }


def _from_meta(wid: str, p: dict, values: Optional[list[float]]) -> MemoryNode:
    chain_memberships = []
    try:
        chain_memberships = [
            ChainMembership(**m)
            for m in json.loads(p.get("chainMembershipsJson", "[]") or "[]")
        ]
    except Exception:
        pass

    def _parse_dt(v) -> datetime:
        if isinstance(v, datetime):
            return v
        if isinstance(v, str):
            for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S+00:00", "%Y-%m-%dT%H:%M:%S"):
                try:
                    return datetime.strptime(v, fmt).replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
        return datetime.now(timezone.utc)

    return MemoryNode(
        id=wid,
        node_id=p.get("nodeId", ""),
        layer=int(p.get("layer", 4)),
        content=p.get("content", ""),
        embedding=values if values else None,
        retention_score=float(p.get("retentionScore", 0.5)),
        importance=float(p.get("importance", 0.5)),
        recency=float(p.get("recency", 1.0)),
        access_count=int(p.get("accessCount", 0)),
        last_accessed=_parse_dt(p.get("lastAccessed")),
        created_at=_parse_dt(p.get("createdAt")),
        dirty=bool(p.get("dirty", False)),
        status=p.get("status", "active"),
        topic_tags=list(p.get("topicTags") or []),
        confidence=p.get("confidence", "medium"),
        chain_memberships=chain_memberships,
    )
