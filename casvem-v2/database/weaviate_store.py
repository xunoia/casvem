"""
Weaviate graph + vector store for CaSVeM.

Every MemoryNode is a Weaviate object with:
  - A custom vector (from qwen3-embedding:0.6b)
  - Properties matching the MemoryNode schema
  - Cross-references for graph edges:
      sourcedFrom   → lower-layer nodes this was derived from
      summarizedBy  → higher-layer nodes that summarise this
      chainPrev     → previous node in a memory chain
      chainNext     → next node in a memory chain
      contradicts   → archived node that this one contradicts
"""

from __future__ import annotations
import json
import asyncio
from datetime import datetime, timezone
from typing import Optional
import uuid

import weaviate
import weaviate.classes as wvc
from weaviate.classes.config import Configure, Property, DataType, ReferenceProperty
from weaviate.classes.query import Filter, QueryReference, MetadataQuery

from models import MemoryNode, ChainMembership
from config import WEAVIATE_HOST, WEAVIATE_PORT, WEAVIATE_GRPC_PORT

COLLECTION = "MemoryNode"


class WeaviateStore:
    def __init__(self):
        self._client: weaviate.WeaviateClient | None = None

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def connect(self):
        self._client = weaviate.connect_to_local(
            host=WEAVIATE_HOST,
            port=WEAVIATE_PORT,
            grpc_port=WEAVIATE_GRPC_PORT,
        )
        self._ensure_schema()

    def close(self):
        if self._client:
            self._client.close()

    def _col(self):
        return self._client.collections.get(COLLECTION)

    # ── Schema ─────────────────────────────────────────────────────────────────

    def _ensure_schema(self):
        if self._client.collections.exists(COLLECTION):
            return
        self._client.collections.create(
            name=COLLECTION,
            vectorizer_config=Configure.Vectorizer.none(),
            properties=[
                Property(name="nodeId",        data_type=DataType.TEXT,   skip_vectorization=True),
                Property(name="layer",         data_type=DataType.INT,    skip_vectorization=True),
                Property(name="content",       data_type=DataType.TEXT),
                Property(name="retentionScore",data_type=DataType.NUMBER, skip_vectorization=True),
                Property(name="importance",    data_type=DataType.NUMBER, skip_vectorization=True),
                Property(name="recency",       data_type=DataType.NUMBER, skip_vectorization=True),
                Property(name="accessCount",   data_type=DataType.INT,    skip_vectorization=True),
                Property(name="lastAccessed",  data_type=DataType.DATE,   skip_vectorization=True),
                Property(name="createdAt",     data_type=DataType.DATE,   skip_vectorization=True),
                Property(name="dirty",         data_type=DataType.BOOL,   skip_vectorization=True),
                Property(name="status",        data_type=DataType.TEXT,   skip_vectorization=True),
                Property(name="topicTags",     data_type=DataType.TEXT_ARRAY, skip_vectorization=True),
                Property(name="confidence",    data_type=DataType.TEXT,   skip_vectorization=True),
                Property(name="chainMembershipsJson", data_type=DataType.TEXT, skip_vectorization=True),
            ],
            references=[
                ReferenceProperty(name="sourcedFrom",  target_collection=COLLECTION),
                ReferenceProperty(name="summarizedBy", target_collection=COLLECTION),
                ReferenceProperty(name="chainPrev",    target_collection=COLLECTION),
                ReferenceProperty(name="chainNext",    target_collection=COLLECTION),
                ReferenceProperty(name="contradicts",  target_collection=COLLECTION),
            ],
        )

    # ── CRUD ───────────────────────────────────────────────────────────────────

    def insert(self, node: MemoryNode) -> str:
        """Insert a new node. Returns its Weaviate UUID."""
        wid = str(uuid.uuid4())
        self._col().data.insert(
            uuid=wid,
            properties=_to_props(node),
            vector=node.embedding or [],
        )
        node.id = wid
        return wid

    def get(self, weaviate_id: str) -> Optional[MemoryNode]:
        obj = self._col().query.fetch_object_by_id(weaviate_id)
        if obj is None:
            return None
        return _from_obj(obj)

    def get_by_node_id(self, node_id: str) -> Optional[MemoryNode]:
        results = self._col().query.fetch_objects(
            filters=Filter.by_property("nodeId").equal(node_id),
            limit=1,
        )
        if not results.objects:
            return None
        return _from_obj(results.objects[0])

    def update_props(self, weaviate_id: str, **props):
        """Partial update of scalar properties."""
        self._col().data.update(uuid=weaviate_id, properties=props)

    def delete(self, weaviate_id: str):
        self._col().data.delete_by_id(weaviate_id)

    # ── Graph edges ────────────────────────────────────────────────────────────

    def add_edge(self, from_id: str, edge_name: str, to_id: str):
        self._col().data.reference_add(
            from_uuid=from_id,
            from_property=edge_name,
            to=to_id,
        )

    def remove_edge(self, from_id: str, edge_name: str, to_id: str):
        self._col().data.reference_delete(
            from_uuid=from_id,
            from_property=edge_name,
            to=to_id,
        )

    def get_neighbours(self, weaviate_id: str, edge_name: str) -> list[MemoryNode]:
        """Return all nodes reachable via a named edge from the given node."""
        obj = self._col().query.fetch_object_by_id(
            weaviate_id,
            return_references=[
                QueryReference(link_on=edge_name, return_properties=list(_PROP_NAMES))
            ],
        )
        if obj is None or not obj.references:
            return []
        refs = obj.references.get(edge_name)
        if not refs:
            return []
        return [_from_obj(r) for r in refs.objects]

    # ── Vector search ──────────────────────────────────────────────────────────

    def search_layer(
        self,
        vector: list[float],
        layer: int,
        limit: int = 10,
        status: str = "active",
    ) -> list[MemoryNode]:
        results = self._col().query.near_vector(
            near_vector=vector,
            limit=limit,
            filters=(
                Filter.by_property("layer").equal(layer)
                & Filter.by_property("status").equal(status)
            ),
            return_metadata=MetadataQuery(distance=True),
        )
        return [_from_obj(o) for o in results.objects]

    def search_by_tags(
        self,
        tags: list[str],
        layer: int,
        limit: int = 20,
    ) -> list[MemoryNode]:
        filters = Filter.by_property("layer").equal(layer)
        if tags:
            tag_filter = Filter.by_property("topicTags").contains_any(tags)
            filters = filters & tag_filter
        results = self._col().query.fetch_objects(
            filters=filters,
            limit=limit,
        )
        return [_from_obj(o) for o in results.objects]

    # ── Layer management ────────────────────────────────────────────────────────

    def count_layer(self, layer: int) -> int:
        agg = self._col().aggregate.over_all(
            filters=Filter.by_property("layer").equal(layer)
        )
        return agg.total_count or 0

    def get_layer(self, layer: int, status: str = "active", limit: int = 500) -> list[MemoryNode]:
        results = self._col().query.fetch_objects(
            filters=(
                Filter.by_property("layer").equal(layer)
                & Filter.by_property("status").equal(status)
            ),
            limit=limit,
        )
        return [_from_obj(o) for o in results.objects]

    def get_dirty(self, layer: int) -> list[MemoryNode]:
        results = self._col().query.fetch_objects(
            filters=(
                Filter.by_property("layer").equal(layer)
                & Filter.by_property("dirty").equal(True)
                & Filter.by_property("status").equal("active")
            ),
            limit=500,
        )
        return [_from_obj(o) for o in results.objects]

    def get_all_counts(self) -> dict[str, int]:
        counts = {}
        for layer in range(1, 6):
            counts[f"L{layer}"] = self.count_layer(layer)
        return counts

    # ── Async wrappers ─────────────────────────────────────────────────────────
    # Weaviate client is sync; run in thread pool for async contexts.

    async def ainsert(self, node: MemoryNode) -> str:
        return await asyncio.to_thread(self.insert, node)

    async def aget(self, wid: str) -> Optional[MemoryNode]:
        return await asyncio.to_thread(self.get, wid)

    async def aget_by_node_id(self, node_id: str) -> Optional[MemoryNode]:
        return await asyncio.to_thread(self.get_by_node_id, node_id)

    async def aupdate_props(self, wid: str, **props):
        await asyncio.to_thread(self.update_props, wid, **props)

    async def aadd_edge(self, from_id: str, edge: str, to_id: str):
        await asyncio.to_thread(self.add_edge, from_id, edge, to_id)

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

    def update_vector(self, wid: str, vector: list[float]):
        self._col().data.update(uuid=wid, vector=vector)

    async def aupdate_vector(self, wid: str, vector: list[float]):
        await asyncio.to_thread(self.update_vector, wid, vector)


# ── Singleton ──────────────────────────────────────────────────────────────────

_store: WeaviateStore | None = None


def get_store() -> WeaviateStore:
    if _store is None:
        raise RuntimeError("Store not initialised — call init_store() first.")
    return _store


def init_store() -> WeaviateStore:
    global _store
    _store = WeaviateStore()
    _store.connect()
    return _store


def close_store():
    if _store:
        _store.close()


# ── Serialisation helpers ──────────────────────────────────────────────────────

_PROP_NAMES = {
    "nodeId", "layer", "content", "retentionScore", "importance", "recency",
    "accessCount", "lastAccessed", "createdAt", "dirty", "status",
    "topicTags", "confidence", "chainMembershipsJson",
}


def _to_props(node: MemoryNode) -> dict:
    def _fmt_date(dt: datetime) -> str:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    chain_json = json.dumps(
        [m.model_dump() for m in node.chain_memberships]
    )
    return {
        "nodeId":         node.node_id,
        "layer":          node.layer,
        "content":        node.content,
        "retentionScore": node.retention_score,
        "importance":     node.importance,
        "recency":        node.recency,
        "accessCount":    node.access_count,
        "lastAccessed":   _fmt_date(node.last_accessed),
        "createdAt":      _fmt_date(node.created_at),
        "dirty":          node.dirty,
        "status":         node.status,
        "topicTags":      node.topic_tags,
        "confidence":     node.confidence,
        "chainMembershipsJson": chain_json,
    }


def _from_obj(obj) -> MemoryNode:
    p = obj.properties
    chain_memberships = []
    raw_chain = p.get("chainMembershipsJson", "[]") or "[]"
    try:
        chain_memberships = [ChainMembership(**m) for m in json.loads(raw_chain)]
    except Exception:
        pass

    def _parse_date(v) -> datetime:
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
        id=str(obj.uuid),
        node_id=p.get("nodeId", ""),
        layer=int(p.get("layer", 4)),
        content=p.get("content", ""),
        retention_score=float(p.get("retentionScore", 0.5)),
        importance=float(p.get("importance", 0.5)),
        recency=float(p.get("recency", 1.0)),
        access_count=int(p.get("accessCount", 0)),
        last_accessed=_parse_date(p.get("lastAccessed")),
        created_at=_parse_date(p.get("createdAt")),
        dirty=bool(p.get("dirty", False)),
        status=p.get("status", "active"),
        topic_tags=list(p.get("topicTags") or []),
        confidence=p.get("confidence", "medium"),
        chain_memberships=chain_memberships,
    )
