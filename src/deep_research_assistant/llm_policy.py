"""Shared token budgets and retry policy for Hy3 workflow stages."""

from dataclasses import dataclass
from math import ceil


@dataclass(frozen=True)
class LLMBudget:
    """Conservative limits and reasoning policy for one model call."""

    input_tokens: int
    output_tokens: int
    recovery_output_tokens: int
    reasoning_effort: str


STAGE_BUDGETS: dict[str, LLMBudget] = {
    # Deterministic extraction and routing stages do not benefit enough from
    # hidden reasoning to justify spending their output budget on it.
    "intent": LLMBudget(6_000, 1_500, 3_000, "no_think"),
    "outline": LLMBudget(10_000, 3_500, 6_000, "no_think"),
    "search": LLMBudget(14_000, 3_000, 5_000, "no_think"),
    # Synthesis stages retain low reasoning, with a bounded no-think fallback.
    "evidence": LLMBudget(28_000, 10_000, 12_000, "low"),
    "report": LLMBudget(28_000, 24_000, 24_000, "low"),
    "summary": LLMBudget(12_000, 1_800, 3_000, "no_think"),
    "followup": LLMBudget(24_000, 4_000, 6_000, "low"),
}


class InputBudgetExceededError(ValueError):
    """Raised before an API request whose prompt exceeds its stage budget."""


def estimate_text_tokens(text: str) -> int:
    """Estimate mixed Chinese/English tokens conservatively without a model tokenizer."""

    if not text:
        return 0
    cjk_count = sum("\u3400" <= char <= "\u9fff" for char in text)
    other_count = len(text) - cjk_count
    return cjk_count + ceil(other_count / 4)


def estimate_messages_tokens(messages: list[dict[str, str]]) -> int:
    """Include a small per-message allowance for chat serialization metadata."""

    return sum(estimate_text_tokens(item.get("content", "")) + 8 for item in messages) + 4
