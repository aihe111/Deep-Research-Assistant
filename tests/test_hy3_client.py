from types import SimpleNamespace

import httpx
import pytest
from openai import RateLimitError
from pydantic import BaseModel

from deep_research_assistant.config import Settings
from deep_research_assistant.hy3_client import Hy3Client, Hy3EmptyResponseError
from deep_research_assistant.llm_policy import InputBudgetExceededError


class FakeCompletions:
    def __init__(
        self,
        failures: list[Exception] | None = None,
        responses: list[object] | None = None,
    ) -> None:
        self.failures = failures or []
        self.responses = responses or []
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.failures:
            raise self.failures.pop(0)
        if self.responses:
            return self.responses.pop(0)
        return _response("正常响应")


def _fake_sdk(completions: FakeCompletions):
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


def _response(
    content: str | None,
    *,
    finish_reason: str = "stop",
    completion_tokens: int = 10,
    reasoning_tokens: int = 0,
    refusal: str | None = None,
):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(content=content, refusal=refusal),
            )
        ],
        usage=SimpleNamespace(
            completion_tokens=completion_tokens,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=reasoning_tokens),
        ),
        _request_id="request-test",
    )


def test_stage_budget_sets_output_limit_and_rejects_oversized_input() -> None:
    completions = FakeCompletions()
    client = Hy3Client(Settings(_env_file=None), client=_fake_sdk(completions))

    assert client.chat([{"role": "user", "content": "你好"}], stage="intent") == "正常响应"
    assert completions.calls[0]["max_tokens"] == 1500
    assert completions.calls[0]["extra_body"] == {"thinking": {"type": "disabled"}}
    assert "reasoning_effort" not in completions.calls[0]

    with pytest.raises(InputBudgetExceededError, match="超过预算"):
        client.chat([{"role": "user", "content": "中" * 6001}], stage="intent")
    assert len(completions.calls) == 1


def test_transient_hy3_error_retries_twice_at_most() -> None:
    request = httpx.Request("POST", "https://example.test/chat/completions")
    response = httpx.Response(429, request=request)
    completions = FakeCompletions([RateLimitError("rate limited", response=response, body=None)])
    settings = Settings(_env_file=None, hy3_max_retries=2, hy3_retry_base_seconds=0)
    client = Hy3Client(settings, client=_fake_sdk(completions))

    assert client.chat([{"role": "user", "content": "retry"}]) == "正常响应"
    assert len(completions.calls) == 2


def test_non_transient_hy3_error_is_not_retried() -> None:
    completions = FakeCompletions([ValueError("bad request")])
    settings = Settings(_env_file=None, hy3_max_retries=2, hy3_retry_base_seconds=0)
    client = Hy3Client(settings, client=_fake_sdk(completions))

    with pytest.raises(ValueError, match="bad request"):
        client.chat([{"role": "user", "content": "fail"}])
    assert len(completions.calls) == 1


class NestedPayload(BaseModel):
    value: str


class StructuredPayload(BaseModel):
    nested: NestedPayload


def test_structured_output_uses_closed_schema_and_stage_reasoning_policy() -> None:
    completions = FakeCompletions(responses=[_response('{"nested":{"value":"ok"}}')])
    client = Hy3Client(Settings(_env_file=None), client=_fake_sdk(completions))

    result = client.chat_structured(
        [{"role": "user", "content": "extract"}],
        StructuredPayload,
        schema_name="structured_payload",
        stage="intent",
    )

    assert result.nested.value == "ok"
    request = completions.calls[0]
    json_schema = request["response_format"]["json_schema"]
    assert json_schema["strict"] is True
    assert json_schema["schema"]["additionalProperties"] is False
    assert json_schema["schema"]["$defs"]["NestedPayload"]["additionalProperties"] is False
    assert request["extra_body"] == {"thinking": {"type": "disabled"}}


def test_empty_structured_response_retries_once_without_thinking() -> None:
    completions = FakeCompletions(
        responses=[
            _response(
                None,
                finish_reason="length",
                completion_tokens=1500,
                reasoning_tokens=1500,
            ),
            _response('{"nested":{"value":"recovered"}}'),
        ]
    )
    client = Hy3Client(Settings(_env_file=None), client=_fake_sdk(completions))

    result = client.chat_structured(
        [{"role": "user", "content": "extract"}],
        StructuredPayload,
        schema_name="structured_payload",
        stage="intent",
        reasoning_effort="low",
    )

    assert result.nested.value == "recovered"
    assert len(completions.calls) == 2
    assert completions.calls[0]["reasoning_effort"] == "low"
    assert completions.calls[1]["max_tokens"] == 3000
    assert completions.calls[1]["extra_body"] == {"thinking": {"type": "disabled"}}
    assert "reasoning_effort" not in completions.calls[1]


def test_repeated_empty_response_raises_diagnostic_error() -> None:
    completions = FakeCompletions(
        responses=[
            _response(None, finish_reason="length", completion_tokens=1500),
            _response(None, finish_reason="length", completion_tokens=3000),
        ]
    )
    client = Hy3Client(Settings(_env_file=None), client=_fake_sdk(completions))

    with pytest.raises(
        Hy3EmptyResponseError,
        match=r"stage=intent.*finish_reason=length.*completion_tokens=3000",
    ):
        client.chat_structured(
            [{"role": "user", "content": "extract"}],
            StructuredPayload,
            schema_name="structured_payload",
            stage="intent",
        )

    assert len(completions.calls) == 2
