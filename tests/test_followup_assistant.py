import json
from typing import Any

from deep_research_assistant.conversation_store import ConversationMessage
from deep_research_assistant.followup_assistant import (
    FollowUpAssistant,
    PreparedContext,
    _bounded_history,
    _retrieve_relevant_memories,
    _select_report_context,
)


class FakeChatClient:
    def __init__(self) -> None:
        self.calls: list[list[dict[str, str]]] = []

    def chat(self, messages: list[dict[str, str]], **_: Any) -> str:
        self.calls.append(messages)
        if "记忆压缩节点" in messages[0]["content"]:
            return "- 用户要求报告约 3000 字。\n- 尚未解决评测局限问题。"
        return "该结论由报告、摘要和相关旧记忆共同支持。[REF001]"


def _message(
    message_id: int,
    role: str,
    content: str,
    *,
    kind: str = "text",
) -> ConversationMessage:
    return ConversationMessage(
        id=message_id,
        conversation_id="web-1",
        role=role,
        content=content,
        kind=kind,
        payload={},
        created_at=f"2026-08-22 00:00:{message_id:02d}",
    )


def test_follow_up_sends_all_four_context_channels() -> None:
    client = FakeChatClient()
    assistant = FollowUpAssistant(client)  # type: ignore[arg-type]
    context = PreparedContext(
        conversation_summary="用户关注检索质量。",
        summarized_through_message_id=4,
        recent_messages=[{"role": "user", "content": "解释结论"}],
        relevant_old_memories=[{"role": "user", "content": "报告控制在3000字"}],
        summary_updated=False,
    )

    answer = assistant.respond("# 报告\n\n核心结论。[REF001]", context)
    payload = json.loads(client.calls[-1][1]["content"])

    assert answer.endswith("[REF001]")
    assert set(payload) == {
        "current_report",
        "conversation_summary",
        "recent_messages",
        "relevant_old_memories",
    }
    assert payload["conversation_summary"] == "用户关注检索质量。"


def test_bounded_history_keeps_newest_messages() -> None:
    history = [_message(index, "user", f"消息{index}") for index in range(1, 20)]

    bounded = _bounded_history(history, max_messages=3)

    assert [item["content"] for item in bounded] == ["消息17", "消息18", "消息19"]


def test_prepare_context_rolls_old_messages_into_summary_once() -> None:
    client = FakeChatClient()
    assistant = FollowUpAssistant(client)  # type: ignore[arg-type]
    messages = [
        _message(1, "user", "报告字数要求是3000字"),
        *[_message(index, "assistant", f"中间讨论消息{index}") for index in range(2, 20)],
        _message(20, "user", "之前对报告字数的要求是什么？"),
    ]

    context = assistant.prepare_context(messages)
    second_context = assistant.prepare_context(
        messages,
        previous_summary=context.conversation_summary,
        summarized_through_message_id=context.summarized_through_message_id,
    )
    extended_messages = [
        *messages,
        _message(21, "assistant", "继续讨论报告结构"),
        _message(22, "user", "再次确认报告字数要求"),
    ]
    third_context = assistant.prepare_context(
        extended_messages,
        previous_summary=context.conversation_summary,
        summarized_through_message_id=context.summarized_through_message_id,
    )

    assert context.summary_updated is True
    assert context.summarized_through_message_id == 4
    assert "3000" in context.conversation_summary
    assert len(context.recent_messages) == 16
    assert context.relevant_old_memories[0]["message_id"] == 1
    assert second_context.summary_updated is False
    assert third_context.summary_updated is True
    assert third_context.summarized_through_message_id == 6
    assert len(client.calls) == 2


def test_relevant_memory_retrieval_ignores_unrelated_old_messages() -> None:
    candidates = [
        _message(1, "user", "报告目标字数为3000字"),
        _message(2, "assistant", "检索使用 OpenAlex 数据源"),
        _message(3, "user", "页面配色采用黑白风格"),
    ]

    memories = _retrieve_relevant_memories("我之前要求报告写多少字？", candidates)

    assert [memory["message_id"] for memory in memories] == [1]


def test_long_report_context_keeps_query_relevant_section() -> None:
    report = (
        "# 报告\n\n## 无关章节\n" + "页面视觉设计。" * 1800
        + "\n\n## 检索质量\n" + "OpenAlex 检索召回率和关键词改写。" * 800
    )

    selected = _select_report_context(report, "OpenAlex 的检索召回率如何？", max_tokens=1200)

    assert "OpenAlex" in selected
    assert len(selected) < len(report)
