"""
MLP Cache Predictor — Phase 1, Week 3-4.

Disabled by default (cfg.use_mlp = False). Enable by setting USE_MLP=true in .env
after the access_log has 1000+ entries.

Predicts P(cache_hit) from query vector + temporal features.
Used by CacheGate when USE_MLP=true to decide L2 pre-population.
"""

import os
import pickle
import time
from typing import Optional

import numpy as np

from config import cfg

_MODEL_PATH = "data/mlp_model.pkl"


class MLPCachePredictor:

    def __init__(self):
        self._clf = None
        self._trained = False
        self._load_if_exists()

    def train(self):
        """Train on all access_log entries in SQLite."""
        from sklearn.neural_network import MLPClassifier
        from core.storage import get_storage

        rows = get_storage().get_access_log_for_training()
        if len(rows) < 100:
            return  # not enough data

        X, y = [], []
        for row in rows:
            # We don't store query_vec in access_log (too large).
            # We use the query_hash to reconstruct feature proxy:
            # temporal features only for now; full vec added when cache writes include vec.
            ts = row["created_at"]
            t = time.localtime(ts)
            hour_vec = [0] * 24
            hour_vec[t.tm_hour] = 1
            day_vec = [0] * 7
            day_vec[t.tm_wday] = 1
            features = hour_vec + day_vec  # 31 dims (no query_vec yet — needs Phase 2 log schema)
            X.append(features)
            y.append(1 if row["hit_type"] != "cold" else 0)

        self._clf = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=200,
            random_state=42,
        )
        self._clf.fit(X, y)
        self._trained = True
        self._save()

    def predict(self, created_at: Optional[int] = None) -> float:
        """Returns P(cache_hit) for temporal features only."""
        if not self._trained or self._clf is None:
            return 0.5

        ts = created_at or int(time.time())
        t = time.localtime(ts)
        hour_vec = [0] * 24
        hour_vec[t.tm_hour] = 1
        day_vec = [0] * 7
        day_vec[t.tm_wday] = 1
        features = np.array(hour_vec + day_vec).reshape(1, -1)

        proba = self._clf.predict_proba(features)
        return float(proba[0][1])

    def should_pre_populate(self, threshold: float = 0.6) -> bool:
        return self.predict() > threshold

    def _save(self):
        os.makedirs("data", exist_ok=True)
        with open(_MODEL_PATH, "wb") as f:
            pickle.dump(self._clf, f)

    def _load_if_exists(self):
        if os.path.exists(_MODEL_PATH):
            with open(_MODEL_PATH, "rb") as f:
                self._clf = pickle.load(f)
            self._trained = True


_predictor = MLPCachePredictor()


def get_mlp_predictor() -> MLPCachePredictor:
    return _predictor
