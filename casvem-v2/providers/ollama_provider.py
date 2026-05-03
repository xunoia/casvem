import json
import re
import httpx
from providers.base import LLMProvider, EmbeddingProvider
from config import OLLAMA_BASE_URL


class OllamaLLMProvider(LLMProvider):
    def __init__(self, model: str):
        self.model = model
        self._client = httpx.AsyncClient(
            base_url=OLLAMA_BASE_URL,
            timeout=httpx.Timeout(300.0)
        )

    async def generate(self, prompt: str, *, temperature: float = 0.1) -> str:
        # Use /api/chat with think:false — disables Qwen3 thinking mode
        # which otherwise consumes all tokens before outputting the response.
        payload = {
            "model":    self.model,
            "think":    False,
            "stream":   False,
            "messages": [{"role": "user", "content": prompt}],
            "options":  {"temperature": temperature, "num_predict": 512},
        }
        r = await self._client.post("/api/chat", json=payload)
        r.raise_for_status()
        return r.json()["message"]["content"].strip()

    async def generate_json(self, prompt: str) -> str:
        payload = {
            "model":    self.model,
            "think":    False,
            "stream":   False,
            "messages": [{"role": "user", "content": prompt}],
            "options":  {"temperature": 0.0, "num_predict": 512},
        }
        r = await self._client.post("/api/chat", json=payload)
        r.raise_for_status()
        raw = r.json()["message"]["content"].strip()
        return _extract_json(raw)

    async def close(self):
        await self._client.aclose()


class OllamaEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model: str):
        self.model = model
        self._client = httpx.AsyncClient(
            base_url=OLLAMA_BASE_URL,
            timeout=httpx.Timeout(120.0)
        )

    async def embed(self, text: str) -> list[float]:
        payload = {"model": self.model, "input": text}
        r = await self._client.post("/api/embed", json=payload)
        if r.status_code == 404:
            r = await self._client.post(
                "/api/embeddings", json={"model": self.model, "prompt": text}
            )
            r.raise_for_status()
            return r.json()["embedding"]
        r.raise_for_status()
        data = r.json()
        embs = data.get("embeddings") or data.get("embedding")
        if isinstance(embs[0], list):
            return embs[0]
        return embs

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        results = []
        for text in texts:
            results.append(await self.embed(text))
        return results

    async def close(self):
        await self._client.aclose()


def _extract_json(text: str) -> str:
    """Pull the first valid JSON object or array out of raw LLM output."""
    text = text.strip()
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass
    fence = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
    if fence:
        candidate = fence.group(1).strip()
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass
    # Array first (extractor returns arrays), then object
    for pat in (r"(\[[\s\S]+\])", r"(\{[\s\S]+\})"):
        m = re.search(pat, text)
        if m:
            candidate = m.group(1)
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                continue
    return text
