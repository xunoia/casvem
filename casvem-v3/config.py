import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    # LLM
    llm_backend: str = os.getenv("LLM_BACKEND", "gemini")
    llm_model: str = os.getenv("LLM_MODEL", "gemini-2.5-flash")
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "2048"))
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # Benchmark judge (can differ from answer LLM)
    judge_backend: str = os.getenv("JUDGE_BACKEND", "gemini")
    judge_model: str = os.getenv("JUDGE_MODEL", "gemini-2.5-flash")

    # Encoder + reranker (always local)
    encoder_model: str = os.getenv("ENCODER_MODEL", "all-MiniLM-L6-v2")
    reranker_model: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    # HNSW
    hnsw_m: int = int(os.getenv("HNSW_M", "16"))
    hnsw_ef_construction: int = int(os.getenv("HNSW_EF_CONSTRUCTION", "200"))
    hnsw_ef_search: int = int(os.getenv("HNSW_EF_SEARCH", "50"))
    top_k: int = int(os.getenv("TOP_K", "50"))
    top_n: int = int(os.getenv("TOP_N", "5"))

    # Cache
    cache_l1_threshold: int = int(os.getenv("CACHE_L1_THRESHOLD", "10"))
    cache_l2_threshold: int = int(os.getenv("CACHE_L2_THRESHOLD", "3"))
    cache_l1_maxsize: int = int(os.getenv("CACHE_L1_MAXSIZE", "500"))
    cache_l2_maxsize: int = int(os.getenv("CACHE_L2_MAXSIZE", "2000"))
    cache_confidence_decay: float = float(os.getenv("CACHE_CONFIDENCE_DECAY", "0.01"))
    cache_hit_similarity_threshold: float = float(
        os.getenv("CACHE_HIT_SIMILARITY_THRESHOLD", "0.92")
    )
    mlp_retrain_after: int = int(os.getenv("MLP_RETRAIN_AFTER", "1000"))
    use_mlp: bool = os.getenv("USE_MLP", "false").lower() == "true"

    # Context builder
    context_token_budget: int = int(os.getenv("CONTEXT_TOKEN_BUDGET", "2048"))
    reranker_early_exit_threshold: float = float(
        os.getenv("RERANKER_EARLY_EXIT_THRESHOLD", "0.95")
    )

    # Storage
    sqlite_path: str = os.getenv("SQLITE_PATH", "data/casvem.db")
    hnsw_index_path: str = os.getenv("HNSW_INDEX_PATH", "data/hnsw_index.bin")

    # Dashboard + cost tracking (Opt 3)
    cost_dashboard: bool = os.getenv("COST_DASHBOARD", "true").lower() == "true"
    mem0_cost_per_query: float = float(os.getenv("MEM0_COST_PER_QUERY", "0.02"))
    gemini_cost_per_1k_input: float = float(os.getenv("GEMINI_COST_PER_1K_INPUT", "0.0001"))
    gemini_cost_per_1k_output: float = float(os.getenv("GEMINI_COST_PER_1K_OUTPUT", "0.0004"))


cfg = Config()
