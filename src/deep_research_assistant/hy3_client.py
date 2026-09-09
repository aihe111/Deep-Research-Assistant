"""Small OpenAI-compatible client wrapper for Hy3."""

import logging
import time
from collections.abc import Sequence
from typing import Any, TypeVar

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError
from pydantic import BaseModel

from deep_research_assistant.config import Settings, get_settings
from deep_research_assistant.json_utils import extract_json_object
from deep_research_assistant.llm_policy import (
    STAGE_BUDGETS,
    InputBudgetExceededError,
    estimate_messages_tokens,
)

StructuredModel = TypeVar("StructuredModel", bound=BaseModel)
logger = logging.getLogger(__name__)


class Hy3EmptyResponseError(RuntimeError):
    """Raised when Hy3 produces no user-visible output after one recovery attempt."""


def _strict_json_schema(value: Any) -> Any:
    """Return a JSON Schema with closed object definitions for Hy3 strict output."""

    if isinstance(value, dict):
        schema = {key: _strict_json_schema(item) for key, item in value.items()}
        if schema.get("type") == "object" or "properties" in schema:
            schema["additionalProperties"] = False
        return schema
    if isinstance(value, list):
        return [_strict_json_schema(item) for item in value]
    return value


class Hy3Client:
    """Provide one stable chat interface for all workflow nodes."""

    def __init__(self, settings: Settings | None = None, client: Any | None = None) -> None:
        self.settings = settings or get_settings()
        self._client = client or OpenAI(
            base_url=self.settings.hy3_base_url,
            api_key=self.settings.hy3_api_key,
            timeout=self.settings.hy3_timeout_seconds,
            max_retries=0,
        )

    @staticmethod
    def _is_retriable(exc: Exception) -> bool:
        if isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError)):
            return True
        return isinstance(exc, APIStatusError) and exc.status_code >= 500

    def _create_with_retry(self, **kwargs: Any) -> Any:
        """Retry only transient transport, rate-limit, and server failures."""

        for attempt in range(self.settings.hy3_max_retries + 1):
            try:
                return self._client.chat.completions.create(**kwargs)
            except Exception as exc:
                if attempt >= self.settings.hy3_max_retries or not self._is_retriable(exc):
                    raise
                delay = self.settings.hy3_retry_base_seconds * (2**attempt)
                if delay:
                    time.sleep(delay)
        raise RuntimeError("Hy3 请求重试状态异常")

    @staticmethod
    def _resolve_budget(
        messages: Sequence[dict[str, str]],
        stage: str,
        max_output_tokens: int | None,
    ) -> tuple[list[dict[str, str]], int]:
        prepared = list(messages)
        budget = STAGE_BUDGETS.get(stage, STAGE_BUDGETS["followup"])
        estimated = estimate_messages_tokens(prepared)
        if estimated > budget.input_tokens:
            raise InputBudgetExceededError(
                f"{stage} 阶段输入约 {estimated} tokens，超过预算 {budget.input_tokens}；"
                "请缩小调研范围或减少输入材料。"
            )
        resolved_output = max_output_tokens or budget.output_tokens
        return prepared, min(resolved_output, budget.output_tokens)

    def _resolve_reasoning_effort(
        self,
        stage: str,
        reasoning_effort: str | None,
    ) -> str:
        if reasoning_effort:
            return reasoning_effort
        budget = STAGE_BUDGETS.get(stage)
        if budget is not None:
            return budget.reasoning_effort
        return self.settings.hy3_reasoning_effort

    @staticmethod
    def _apply_reasoning_policy(kwargs: dict[str, Any], reasoning_effort: str) -> None:
        """Use TokenHub's explicit thinking switch for deterministic stages."""

        kwargs.pop("reasoning_effort", None)
        kwargs.pop("extra_body", None)
        if reasoning_effort == "no_think":
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        else:
            kwargs["reasoning_effort"] = reasoning_effort

    @staticmethod
    def _response_diagnostics(response: Any) -> dict[str, Any]:
        choice = response.choices[0] if getattr(response, "choices", None) else None
        message = getattr(choice, "message", None)
        usage = getattr(response, "usage", None)
        details = getattr(usage, "completion_tokens_details", None)
        return {
            "request_id": getattr(response, "_request_id", None),
            "finish_reason": getattr(choice, "finish_reason", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "reasoning_tokens": getattr(details, "reasoning_tokens", None),
            "refusal": bool(getattr(message, "refusal", None)),
        }

    @staticmethod
    def _content(response: Any) -> str:
        if not getattr(response, "choices", None):
            return ""
        content = getattr(response.choices[0].message, "content", None)
        return content.strip() if content and content.strip() else ""

    def _create_with_empty_recovery(
        self,
        *,
        stage: str,
        output_limit: int,
        kwargs: dict[str, Any],
    ) -> str:
        """Retry one empty completion with thinking disabled and bounded headroom."""

        response = self._create_with_retry(**kwargs)
        content = self._content(response)
        if content:
            return content

        diagnostics = self._response_diagnostics(response)
        logger.warning(
            "Hy3 returned an empty completion: stage=%s diagnostics=%s",
            stage,
            diagnostics,
        )
        if diagnostics["refusal"]:
            raise Hy3EmptyResponseError(f"Hy3 拒绝生成响应（stage={stage}）")

        budget = STAGE_BUDGETS.get(stage, STAGE_BUDGETS["followup"])
        recovery_limit = min(
            budget.recovery_output_tokens,
            max(output_limit * 2, output_limit + 512),
        )
        recovery_kwargs = dict(kwargs)
        recovery_kwargs["max_tokens"] = max(output_limit, recovery_limit)
        self._apply_reasoning_policy(recovery_kwargs, "no_think")
        logger.warning(
            "Retrying empty Hy3 completion once: stage=%s max_tokens=%s thinking=disabled",
            stage,
            recovery_kwargs["max_tokens"],
        )
        recovered = self._create_with_retry(**recovery_kwargs)
        content = self._content(recovered)
        if content:
            return content

        recovered_diagnostics = self._response_diagnostics(recovered)
        logger.error(
            "Hy3 empty-completion recovery failed: stage=%s diagnostics=%s",
            stage,
            recovered_diagnostics,
        )
        raise Hy3EmptyResponseError(
            "Hy3 连续返回空响应"
            f"（stage={stage}, finish_reason={recovered_diagnostics['finish_reason']}, "
            f"completion_tokens={recovered_diagnostics['completion_tokens']}, "
            f"reasoning_tokens={recovered_diagnostics['reasoning_tokens']}）"
        )

    def chat(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float = 0.2,
        reasoning_effort: str | None = None,
        stage: str = "followup",
        max_output_tokens: int | None = None,
    ) -> str:
        """Send a chat-completions request and return non-empty text content."""

        prepared, output_limit = self._resolve_budget(messages, stage, max_output_tokens)
        kwargs: dict[str, Any] = {
            "model": self.settings.hy3_model,
            "messages": prepared,
            "temperature": temperature,
            "top_p": 1.0,
            "max_tokens": output_limit,
        }
        self._apply_reasoning_policy(
            kwargs,
            self._resolve_reasoning_effort(stage, reasoning_effort),
        )
        return self._create_with_empty_recovery(
            stage=stage,
            output_limit=output_limit,
            kwargs=kwargs,
        )

    def chat_structured(
        self,
        messages: Sequence[dict[str, str]],
        response_model: type[StructuredModel],
        *,
        schema_name: str,
        temperature: float = 0.1,
        reasoning_effort: str | None = None,
        stage: str = "intent",
        max_output_tokens: int | None = None,
    ) -> StructuredModel:
        """Request JSON Schema output and validate it with a Pydantic model."""

        prepared, output_limit = self._resolve_budget(messages, stage, max_output_tokens)
        kwargs: dict[str, Any] = {
            "model": self.settings.hy3_model,
            "messages": prepared,
            "temperature": temperature,
            "top_p": 1.0,
            "max_tokens": output_limit,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": _strict_json_schema(response_model.model_json_schema()),
                },
            },
        }
        self._apply_reasoning_policy(
            kwargs,
            self._resolve_reasoning_effort(stage, reasoning_effort),
        )
        content = self._create_with_empty_recovery(
            stage=stage,
            output_limit=output_limit,
            kwargs=kwargs,
        )
        return response_model.model_validate(extract_json_object(content))
