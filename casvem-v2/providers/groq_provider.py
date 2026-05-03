import asyncio
import json
import re
import httpx
from providers.base import LLMProvider
from config import GROQ_API_KEY, GROQ_MODEL

_GROQ_URL    = "https://api.groq.com/openai/v1/chat/completions"
_MAX_RETRIES = 12
_MIN_WAIT    = 15   # minimum seconds to wait on 429

# Global lock: serialise all Groq requests so concurrent background tasks
# don't flood the 30 RPM free-tier limit and trigger cascading 429s.
_groq_lock: asyncio.Lock | None = None


def _get_lock() -> asyncio.Lock:
    global _groq_lock
    if _groq_lock is None:
        _groq_lock = asyncio.Lock()
    return _groq_lock


class GroqLLMProvider(LLMProvider):
    def __init__(self, model: str = GROQ_MODEL):
        self.model = model
        self._client = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type":  "application/json",
            },
            timeout=httpx.Timeout(180.0),
        )

    async def _post(self, payload: dict) -> dict:
        async with _get_lock():
            for attempt in range(_MAX_RETRIES):
                r = await self._client.post(_GROQ_URL, json=payload)
                if r.status_code == 429:
                    retry_after = r.headers.get("retry-after")
                    wait = max(_MIN_WAIT, int(retry_after)) if retry_after else _MIN_WAIT * (2 ** min(attempt, 2))
                    await asyncio.sleep(wait)
                    continue
                r.raise_for_status()
                # small inter-request gap to stay well within RPM budget
                await asyncio.sleep(2)
                return r.json()
            raise RuntimeError(f"Groq still rate-limiting after {_MAX_RETRIES} retries")

    async def generate(self, prompt: str, *, temperature: float = 0.1) -> str:
        payload = {
            "model":       self.model,
            "messages":    [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens":  512,
        }
        data = await self._post(payload)
        return data["choices"][0]["message"]["content"].strip()

    async def generate_json(self, prompt: str) -> str:
        payload = {
            "model":       self.model,
            "messages":    [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens":  1024,
        }
        data = await self._post(payload)
        raw = data["choices"][0]["message"]["content"].strip()
        return _extract_json(raw)

    async def close(self):
        await self._client.aclose()


def _extract_json(text: str) -> str:
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
