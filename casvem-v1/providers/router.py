from providers.ollama_provider import OllamaLLMProvider, OllamaEmbeddingProvider
from config import LLM_FAST_MODEL, LLM_STRONG_MODEL, EMBEDDING_MODEL

# Singleton providers — initialised once at startup
_fast:    OllamaLLMProvider     | None = None
_strong:  OllamaLLMProvider     | None = None
_embedder: OllamaEmbeddingProvider | None = None


def init_providers():
    global _fast, _strong, _embedder
    _fast     = OllamaLLMProvider(LLM_FAST_MODEL)
    _strong   = OllamaLLMProvider(LLM_STRONG_MODEL)
    _embedder = OllamaEmbeddingProvider(EMBEDDING_MODEL)


async def close_providers():
    if _fast:    await _fast.close()
    if _strong:  await _strong.close()
    if _embedder: await _embedder.close()


def fast() -> OllamaLLMProvider:
    """Binary / short output tasks: classification, routing, sufficiency check."""
    if _fast is None:
        raise RuntimeError("Providers not initialised — call init_providers() first.")
    return _fast


def strong() -> OllamaLLMProvider:
    """Structured JSON output tasks: extraction, analysis, synthesis."""
    if _strong is None:
        raise RuntimeError("Providers not initialised — call init_providers() first.")
    return _strong


def embedder() -> OllamaEmbeddingProvider:
    if _embedder is None:
        raise RuntimeError("Providers not initialised — call init_providers() first.")
    return _embedder
