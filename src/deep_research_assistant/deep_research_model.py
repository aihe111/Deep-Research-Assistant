"""LangChain chat-model factory for the Hy3 OpenAI-compatible endpoint."""

from langchain_openai import ChatOpenAI

from deep_research_assistant.config import Settings


def build_chat_model(
    settings: Settings,
    *,
    max_tokens: int,
    disable_thinking: bool = False,
) -> ChatOpenAI:
    """Create a stage-scoped Hy3 model with LangSmith-compatible tracing."""

    extra_body = None
    reasoning_effort = settings.hy3_reasoning_effort
    if disable_thinking or reasoning_effort == "no_think":
        extra_body = {"thinking": {"type": "disabled"}}
        reasoning_effort = None
    return ChatOpenAI(
        model=settings.hy3_model,
        base_url=settings.hy3_base_url,
        api_key=settings.hy3_api_key,
        timeout=settings.hy3_timeout_seconds,
        max_retries=settings.hy3_max_retries,
        max_tokens=max_tokens,
        temperature=0.1,
        reasoning_effort=reasoning_effort,
        extra_body=extra_body,
    )
