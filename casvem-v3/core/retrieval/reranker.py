from functools import lru_cache
from sentence_transformers import CrossEncoder
from config import cfg


class Reranker:

    def __init__(self, model_name: str, early_exit_threshold: float):
        self._model = CrossEncoder(model_name)
        self._threshold = early_exit_threshold

    def rerank(
        self, query: str, candidates: list[dict], top_n: int
    ) -> list[dict]:
        """
        candidates: list of memory dicts (must have 'text' key).
        Returns top_n memories sorted by cross-encoder score, highest first.
        Early exit: if top score > threshold, skip remaining candidates.
        """
        if not candidates:
            return []

        pairs = [[query, m["text"]] for m in candidates]
        scores = self._model.predict(pairs, show_progress_bar=False)

        scored = sorted(
            zip(scores, candidates), key=lambda x: x[0], reverse=True
        )

        result = []
        for score, memory in scored:
            memory = dict(memory)
            memory["rerank_score"] = float(score)
            result.append(memory)
            if len(result) >= top_n:
                break
            # Early exit: top result is confident enough, stop here
            if score > self._threshold and len(result) == 1:
                break

        return result


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    return Reranker(
        model_name=cfg.reranker_model,
        early_exit_threshold=cfg.reranker_early_exit_threshold,
    )
