from functools import lru_cache
import numpy as np
from sentence_transformers import SentenceTransformer
from config import cfg


class EuclideanEncoder:

    def __init__(self, model_name: str):
        self._model = SentenceTransformer(model_name)
        self._dim = self._model.get_embedding_dimension()

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, text: str) -> np.ndarray:
        vec = self._model.encode(text, normalize_embeddings=True, show_progress_bar=False)
        return vec.astype(np.float32)

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(
            texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False
        )
        return vecs.astype(np.float32)


@lru_cache(maxsize=1)
def get_encoder() -> EuclideanEncoder:
    return EuclideanEncoder(cfg.encoder_model)
