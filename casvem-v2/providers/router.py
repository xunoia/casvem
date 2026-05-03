from providers.base import LLMProvider, EmbeddingProvider
from config import LLM_BACKEND, GROQ_MODEL, LLM_FAST_MODEL, LLM_STRONG_MODEL, EMBEDDING_MODEL

# Singleton providers — initialised once at startup
_fast:     LLMProvider      | None = None
_strong:   LLMProvider      | None = None
_embedder: EmbeddingProvider | None = None


def init_providers():
    global _fast, _strong, _embedder

    if LLM_BACKEND == "groq":
        from providers.groq_provider import GroqLLMProvider
        _fast   = GroqLLMProvider(GROQ_MODEL)
        _strong = GroqLLMProvider(GROQ_MODEL)
    else:
        from providers.ollama_provider import OllamaLLMProvider
        _fast   = OllamaLLMProvider(LLM_FAST_MODEL)
        _strong = OllamaLLMProvider(LLM_STRONG_MODEL)

    # Embeddings always use Ollama
    from providers.ollama_provider import OllamaEmbeddingProvider
    _embedder = OllamaEmbeddingProvider(EMBEDDING_MODEL)


async def close_providers():
    if _fast:     await _fast.close()
    if _strong and _strong is not _fast:
        await _strong.close()
    if _embedder: await _embedder.close()


def fast() -> LLMProvider:
    """Binary / short output tasks: classification, routing, sufficiency check."""
    if _fast is None:
        raise RuntimeError("Providers not initialised — call init_providers() first.")
    return _fast


def strong() -> LLMProvider:
    """Structured JSON output tasks: extraction, analysis, synthesis."""
    if _strong is None:
        raise RuntimeError("Providers not initialised — call init_providers() first.")
    return _strong


def embedder() -> EmbeddingProvider:
    if _embedder is None:
        raise RuntimeError("Providers not initialised — call init_providers() first.")
    return _embedder
