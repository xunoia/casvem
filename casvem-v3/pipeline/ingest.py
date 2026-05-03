from typing import Optional
from core.memory.writer import ingest as _ingest


def ingest(
    text: str,
    memory_type: str = "fact",
    project_id: Optional[str] = None,
    author_id: Optional[str] = None,
) -> str:
    """Thin wrapper. Encode → store → index → return memory_id."""
    return _ingest(
        text=text,
        memory_type=memory_type,
        project_id=project_id,
        author_id=author_id,
    )
