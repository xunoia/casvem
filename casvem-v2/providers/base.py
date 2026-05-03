from abc import ABC, abstractmethod


class LLMProvider(ABC):
    @abstractmethod
    async def generate(self, prompt: str, *, temperature: float = 0.1) -> str: ...

    @abstractmethod
    async def generate_json(self, prompt: str) -> str:
        """Like generate() but signals to the provider that JSON output is expected."""
        ...


class EmbeddingProvider(ABC):
    @abstractmethod
    async def embed(self, text: str) -> list[float]: ...

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]: ...
