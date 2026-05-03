import os
from dotenv import load_dotenv

load_dotenv()

# ── Models ────────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL    = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_FAST_MODEL     = os.getenv("LLM_FAST",    "qwen3:1.7b")      # binary tasks — 1.4GB, fast on CPU
LLM_STRONG_MODEL   = os.getenv("LLM_STRONG",  "qwen3:1.7b")      # JSON extraction — share model with fast, 8-15 tok/s on CPU
EMBEDDING_MODEL    = os.getenv("EMBEDDER",    "nomic-embed-text") # 274MB, fast embeddings
EMBEDDING_DIM      = int(os.getenv("EMBEDDING_DIM", "1024"))

# ── Weaviate ──────────────────────────────────────────────────────────────────
WEAVIATE_HOST      = os.getenv("WEAVIATE_HOST", "localhost")
WEAVIATE_PORT      = int(os.getenv("WEAVIATE_PORT", "8080"))
WEAVIATE_GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))

# ── Layer configuration ────────────────────────────────────────────────────────
#   lambda = recency decay rate  |  promote/demote = retention_score thresholds
LAYER_CONFIG = {
    1: {"max_tokens": 500,     "max_lines": 20,   "lambda": 0.10, "promote": 0.85, "demote": 0.30},
    2: {"max_tokens": 5_000,   "max_lines": 100,  "lambda": 0.05, "promote": 0.75, "demote": 0.20},
    3: {"max_tokens": 50_000,  "max_lines": 500,  "lambda": 0.02, "promote": 0.65, "demote": 0.15},
    4: {"max_tokens": 500_000, "max_lines": 5000, "lambda": 0.01, "promote": 0.55, "demote": 0.10},
    5: {"max_tokens": -1,      "max_lines": -1,   "lambda": 0.005,"promote": 0.0,  "demote": 0.0},
}

# ── Write pipeline ────────────────────────────────────────────────────────────
DEDUP_SIMILARITY_THRESHOLD   = 0.92   # L4 dedup: skip if too similar to existing
MERGE_SIMILARITY_THRESHOLD   = 0.88   # promotion engine: merge if this similar
CONTRADICTION_CONFIRM_THRESHOLD = 0.75 # min importance to confirm a contradiction

# ── Read pipeline ─────────────────────────────────────────────────────────────
L2_SEARCH_LIMIT       = 10    # max L2 results from vector search
L3_POINTER_LIMIT      = 5     # max L3 lines to follow per L2 hit
SYNTHESIS_MAX_TOKENS  = 1000  # max tokens in synthesised memory block
L1_ALWAYS_INJECT      = True  # always include L1 in every prompt

# ── Scheduler ─────────────────────────────────────────────────────────────────
SCHEDULER_INTERVAL_MINUTES = 60
