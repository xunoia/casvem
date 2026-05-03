import ollama
from .base import BaseLLMProvider, CompletionResult


class OllamaProvider(BaseLLMProvider):

    def __init__(self, model: str, base_url: str = "http://localhost:11434",
                 temperature: float = 0.1):
        self._model = model
        self._client = ollama.AsyncClient(host=base_url)
        self._temperature = temperature

    async def complete(self, prompt: str, max_tokens: int = 1024) -> CompletionResult:
        response = await self._client.generate(
            model=self._model,
            prompt=prompt,
            options={
                "num_predict": max_tokens,
                "temperature": self._temperature,
            },
        )
        # Ollama returns eval_count (output tokens) and prompt_eval_count (input tokens)
        input_tokens = response.get("prompt_eval_count", 0)
        output_tokens = response.get("eval_count", 0)

        return CompletionResult(
            text=response["response"],
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
