import time
from datetime import datetime
from typing import Optional

from pyroaring import BitMap


class BitmapFilter:
    """
    Single-field Roaring Bitmap indexes over memory metadata.
    Each field-value pair has its own BitMap of HNSW labels.

    Indexed fields: memory_type, project_id, author_id, date_bucket (YYYY-MM).

    filter() returns the intersection of all provided constraints.
    Returns None if no constraints given (caller should search full index).
    """

    def __init__(self):
        # { field_name: { field_value: BitMap } }
        self._indexes: dict[str, dict[str, BitMap]] = {
            "memory_type": {},
            "project_id": {},
            "author_id": {},
            "date_bucket": {},
        }

    def add(
        self,
        hnsw_label: int,
        memory_type: Optional[str] = None,
        project_id: Optional[str] = None,
        author_id: Optional[str] = None,
        created_at: Optional[int] = None,
    ):
        """Index a memory by its HNSW label."""
        if memory_type:
            self._get_or_create("memory_type", memory_type).add(hnsw_label)
        if project_id:
            self._get_or_create("project_id", project_id).add(hnsw_label)
        if author_id:
            self._get_or_create("author_id", author_id).add(hnsw_label)
        if created_at:
            bucket = datetime.fromtimestamp(created_at).strftime("%Y-%m")
            self._get_or_create("date_bucket", bucket).add(hnsw_label)

    def remove(self, hnsw_label: int):
        """Remove a label from all indexes (called on memory delete)."""
        for field_map in self._indexes.values():
            for bm in field_map.values():
                bm.discard(hnsw_label)

    def filter(
        self,
        memory_type: Optional[str] = None,
        project_id: Optional[str] = None,
        author_id: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> Optional[set[str]]:
        """
        Returns set of memory_ids matching all provided constraints.
        Returns None if no constraints — caller treats None as 'search all'.
        Requires label_to_id map for the final conversion.
        """
        bitmaps: list[BitMap] = []

        if memory_type:
            bm = self._indexes["memory_type"].get(memory_type)
            if bm is None:
                return set()  # no memories of this type
            bitmaps.append(bm)

        if project_id:
            bm = self._indexes["project_id"].get(project_id)
            if bm is None:
                return set()
            bitmaps.append(bm)

        if author_id:
            bm = self._indexes["author_id"].get(author_id)
            if bm is None:
                return set()
            bitmaps.append(bm)

        if date_from or date_to:
            date_bm = self._filter_date_range(date_from, date_to)
            if not date_bm:
                return set()
            bitmaps.append(date_bm)

        if not bitmaps:
            return None  # no constraints → search all

        result = bitmaps[0]
        for bm in bitmaps[1:]:
            result = result & bm  # intersection

        return set(result)  # returns set of hnsw_labels

    def _filter_date_range(
        self, date_from: Optional[str], date_to: Optional[str]
    ) -> Optional[BitMap]:
        """Merge all date_bucket bitmaps within [date_from, date_to] range."""
        matching = BitMap()
        for bucket, bm in self._indexes["date_bucket"].items():
            if date_from and bucket < date_from:
                continue
            if date_to and bucket > date_to:
                continue
            matching |= bm
        return matching if matching else None

    def _get_or_create(self, field: str, value: str) -> BitMap:
        if value not in self._indexes[field]:
            self._indexes[field][value] = BitMap()
        return self._indexes[field][value]

    def rebuild_from_storage(self, storage):
        """Rebuild all indexes from SQLite on startup."""
        rows = storage._conn.execute(
            "SELECT hnsw_label, memory_type, project_id, author_id, created_at FROM memories"
        ).fetchall()
        for row in rows:
            self.add(
                hnsw_label=row["hnsw_label"],
                memory_type=row["memory_type"],
                project_id=row["project_id"],
                author_id=row["author_id"],
                created_at=row["created_at"],
            )
