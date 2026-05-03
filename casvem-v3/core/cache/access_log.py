from config import cfg
from core.storage import get_storage


class AccessLog:
    """
    Thin wrapper around Storage.log_access().
    Tracks access counts per query_hash and triggers MLP retrain when threshold is hit.
    """

    def __init__(self):
        self._log_count_at_last_train: int = 0

    def log(
        self,
        query_hash: str,
        memory_ids: list[str],
        hit_type: str,
        latency_ms: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ):
        storage = get_storage()
        storage.log_access(
            query_hash=query_hash,
            memory_ids=memory_ids,
            hit_type=hit_type,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def should_retrain_mlp(self) -> bool:
        """True when enough new entries have accumulated since last training."""
        current = get_storage().count_access_log()
        new_entries = current - self._log_count_at_last_train
        return new_entries >= cfg.mlp_retrain_after

    def mark_retrained(self):
        self._log_count_at_last_train = get_storage().count_access_log()


_access_log = AccessLog()


def get_access_log() -> AccessLog:
    return _access_log
