import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM Backend ───────────────────────────────────────────────────────────────
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama")   # "groq" | "ollama"

# Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# Ollama
OLLAMA_BASE_URL  = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL        = os.getenv("LLM_MODEL", "qwen3:1.7b")
LLM_FAST_MODEL   = LLM_MODEL
LLM_STRONG_MODEL = LLM_MODEL
EMBEDDING_MODEL  = os.getenv("EMBEDDING_MODEL", "nomic-embed-text:latest")
EMBEDDING_DIM    = int(os.getenv("EMBEDDING_DIM", "768"))

# ── Vector Store ──────────────────────────────────────────────────────────────
VECTOR_STORE_BACKEND = os.getenv("VECTOR_STORE_BACKEND", "pinecone")  # "pinecone" | "qdrant"

# Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_CLOUD   = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION  = os.getenv("PINECONE_REGION", "us-east-1")
PINECONE_INDEX   = os.getenv("PINECONE_INDEX", "casvem-memory")

# Weaviate (legacy — used only when VECTOR_STORE_BACKEND=weaviate)
WEAVIATE_HOST      = os.getenv("WEAVIATE_HOST", "localhost")
WEAVIATE_PORT      = int(os.getenv("WEAVIATE_PORT", "8080"))
WEAVIATE_GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))

# ── Retrieval tuning ──────────────────────────────────────────────────────────
L2_SEARCH_LIMIT            = 8
L3_POINTER_LIMIT           = 3
DEDUP_SIMILARITY_THRESHOLD = 0.92

# ── Layer config ──────────────────────────────────────────────────────────────
LAYER_CONFIG = {
    1: {"lambda": 0.001, "max_lines": 20},
    2: {"lambda": 0.005, "max_lines": 100},
    3: {"lambda": 0.01,  "max_lines": 500},
    4: {"lambda": 0.01,  "max_lines": 2000},
    5: {"lambda": 0.001, "max_lines": 10000},
}
