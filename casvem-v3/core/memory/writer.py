from typing import Optional
from core.encoder import get_encoder
from core.storage import get_storage
from core.retrieval.bitmap_filter import BitmapFilter


# Module-level bitmap filter instance (rebuilt from storage on startup)
_bitmap: Optional[BitmapFilter] = None


def get_bitmap() -> BitmapFilter:
    global _bitmap
    if _bitmap is None:
        _bitmap = BitmapFilter()
        _bitmap.rebuild_from_storage(get_storage())
    return _bitmap


def reset_bitmap():
    global _bitmap
    _bitmap = None


def ingest(
    text: str,
    memory_type: str = "fact",
    project_id: Optional[str] = None,
    author_id: Optional[str] = None,
) -> str:
    """
    Encode text → store in SQLite → add to HNSW → update bitmap indexes.
    Returns memory_id.
    """
    encoder = get_encoder()
    storage = get_storage()
    bitmap = get_bitmap()

    vec = encoder.encode(text)
    memory_id = storage.add_memory(
        text=text,
        vector=vec,
        memory_type=memory_type,
        project_id=project_id,
        author_id=author_id,
    )

    # Get the hnsw_label that was just assigned
    mem = storage.get_memory(memory_id)
    bitmap.add(
        hnsw_label=mem["hnsw_label"],
        memory_type=memory_type,
        project_id=project_id,
        author_id=author_id,
        created_at=mem["created_at"],
    )

    return memory_id
