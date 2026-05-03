from __future__ import annotations
import math
import uuid
from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel, Field
from config import LAYER_CONFIG


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _new_id(layer: int) -> str:
    return f"mem_L{layer}_{uuid.uuid4().hex[:8]}"


# ── Core memory node ──────────────────────────────────────────────────────────

class ChainMembership(BaseModel):
    chain_id:    str
    chain_type:  str            # temporal | causal | departmental | entity | contradictory
    position:    int
    prev_id:     Optional[str] = None
    next_id:     Optional[str] = None
    causal_label: Optional[str] = None


class MemoryNode(BaseModel):
    id:               str   = Field(default_factory=lambda: str(uuid.uuid4()))
    node_id:          str   = ""              # human-readable e.g. mem_L2_3f8a1b
    layer:            int
    content:          str
    embedding:        Optional[list[float]] = None

    source_pointers:  list[str] = []          # points DOWN → where this was derived from
    derived_by:       list[str] = []          # points UP   → what summarised this

    retention_score:  float = 0.5
    importance:       float = 0.5
    recency:          float = 1.0
    access_count:     int   = 0
    last_accessed:    datetime = Field(default_factory=_now)
    created_at:       datetime = Field(default_factory=_now)

    dirty:            bool  = False
    status:           str   = "active"        # active | archived
    topic_tags:       list[str] = []
    confidence:       str   = "medium"        # high | medium | low

    chain_memberships: list[ChainMembership] = []

    def model_post_init(self, __context):
        if not self.node_id:
            self.node_id = _new_id(self.layer)

    def compute_retention(
        self,
        neighbor_embeddings: Optional[list[list[float]]] = None
    ) -> float:
        cfg = LAYER_CONFIG[self.layer]
        lam = cfg["lambda"]

        now = _now()
        la  = self.last_accessed
        if la.tzinfo is None:
            la = la.replace(tzinfo=timezone.utc)

        days_since   = max(0, (now - la).days)
        recency      = math.exp(-lam * days_since)

        ca = self.created_at
        if ca.tzinfo is None:
            ca = ca.replace(tzinfo=timezone.utc)
        age_days     = max(1, (now - ca).days)
        access_freq  = min(1.0, (self.access_count / age_days) * 10)

        uniqueness = 1.0
        if self.embedding and neighbor_embeddings:
            def _cos(a: list[float], b: list[float]) -> float:
                dot   = sum(x * y for x, y in zip(a, b))
                mag_a = math.sqrt(sum(x * x for x in a))
                mag_b = math.sqrt(sum(x * x for x in b))
                return dot / (mag_a * mag_b) if mag_a * mag_b > 0 else 0.0
            sims = [_cos(self.embedding, n) for n in neighbor_embeddings if n]
            if sims:
                uniqueness = 1.0 - max(sims)

        score = (
            self.importance  * 0.40 +
            recency          * 0.30 +
            access_freq      * 0.20 +
            uniqueness       * 0.10
        )
        self.recency         = recency
        self.retention_score = max(0.0, min(1.0, score))
        return self.retention_score


# ── API request / response models ─────────────────────────────────────────────

class SessionRequest(BaseModel):
    user_id:    str = "default"
    session_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    transcript: str                   # raw conversation text


class QueryRequest(BaseModel):
    user_id: str = "default"
    query:   str


class QueryResponse(BaseModel):
    query:         str
    memory_block:  str
    answer:        str
    confidence:    str
    routed_to:     str                # "local" | "cloud"
    layers_hit:    list[int]


class MemoryListResponse(BaseModel):
    layer:    int
    count:    int
    nodes:    list[MemoryNode]


class StatusResponse(BaseModel):
    weaviate_ok:    bool
    ollama_ok:      bool
    layer_counts:   dict[str, int]
    scheduler_running: bool


class ConsolidationLabel(str):
    CONTRADICTS = "CONTRADICTS"
    ADDS        = "ADDS"
    REINFORCES  = "REINFORCES"
    IRRELEVANT  = "IRRELEVANT"


class RoutingDecision(str):
    LOCAL = "LOCAL"
    CLOUD = "CLOUD"
