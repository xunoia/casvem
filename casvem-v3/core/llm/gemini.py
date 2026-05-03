from google import genai
from google.genai import types
from .base import BaseLLMProvider, CompletionResult


class GeminiProvider(BaseLLMProvider):

    def __init__(self, model: str, api_key: str, temperature: float = 0.1):
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._temperature = temperature

    async def complete(self, prompt: str, max_tokens: int = 1024) -> CompletionResult:
        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=self._temperature,
                max_output_tokens=max_tokens,
            ),
        )

        # Opt 3: real token counts from response metadata
        input_tokens = 0
        output_tokens = 0
        if response.usage_metadata:
            input_tokens = response.usage_metadata.prompt_token_count or 0
            output_tokens = response.usage_metadata.candidates_token_count or 0

        return CompletionResult(
            text=response.text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
