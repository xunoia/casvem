from functools import lru_cache
from config import cfg
from .base import BaseLLMProvider


@lru_cache(maxsize=2)
def get_llm_provider(backend: str = None, model: str = None) -> BaseLLMProvider:
    """
    Returns a cached LLM provider instance.
    Called with no args → uses cfg.llm_backend and cfg.llm_model.
    Called with backend/model → used by benchmark to get the judge provider.
    """
    backend = backend or cfg.llm_backend
    model = model or cfg.llm_model

    if backend == "gemini":
        from .gemini import GeminiProvider
        return GeminiProvider(
            model=model,
            api_key=cfg.gemini_api_key,
            temperature=cfg.llm_temperature,
        )

    if backend == "ollama":
        from .ollama import OllamaProvider
        return OllamaProvider(
            model=model,
            base_url=cfg.ollama_base_url,
            temperature=cfg.llm_temperature,
        )

    raise ValueError(
        f"Unknown LLM_BACKEND: '{backend}'. "
        f"Add a new provider file in core/llm/ and register it here."
    )


def get_judge_provider() -> BaseLLMProvider:
    return get_llm_provider(backend=cfg.judge_backend, model=cfg.judge_model)
