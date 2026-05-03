from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class CompletionResult:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


JUDGE_PROMPT = """You are evaluating whether an AI answer is correct.

Question: {question}
Ground truth answer: {ground_truth}
AI answer: {answer}

Is the AI answer correct or equivalent to the ground truth?
Reply with only one word: YES or NO"""


class BaseLLMProvider(ABC):

    @abstractmethod
    async def complete(self, prompt: str, max_tokens: int = 1024) -> CompletionResult:
        """Generate a completion. Returns text + actual token counts."""
        ...

    async def judge(self, question: str, ground_truth: str, answer: str) -> bool:
        """Return True if the answer is correct vs ground_truth. Used by benchmark scorer."""
        prompt = JUDGE_PROMPT.format(
            question=question, ground_truth=ground_truth, answer=answer
        )
        result = await self.complete(prompt, max_tokens=5)
        return "yes" in result.text.strip().lower()
