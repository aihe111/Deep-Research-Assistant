"""Layered conversation memory for grounded report follow-up chat."""

import json
import math
import re
from dataclasses import dataclass
from typing import Any

from deep_research_assistant.conversation_store import ConversationMessage
from deep_research_assistant.hy3_client import Hy3Client
from deep_research_assistant.llm_policy import estimate_text_tokens

RECENT_MAX_MESSAGES = 16
RECENT_MAX_CHARACTERS = 12000
RELEVANT_MEMORY_LIMIT = 6
RELEVANT_MEMORY_MAX_CHARACTERS = 5000
SUMMARY_MAX_CHARACTERS = 4000
SUMMARY_INPUT_MAX_TOKENS = 6_000
REPORT_CONTEXT_MAX_TOKENS = 7_000
FOLLOWUP_SUMMARY_MAX_TOKENS = 3_000
FOLLOWUP_RECENT_MAX_TOKENS = 8_000
FOLLOWUP_RELEVANT_MAX_TOKENS = 3_000

SYSTEM_PROMPT = """你是调研报告的后续问答助手。用户已经完成一轮文献调研，现在会围绕报告继续提问。

你会收到四类上下文：current_report、conversation_summary、recent_messages 和
relevant_old_memories。遵守以下规则：
1. 当前报告是研究事实和引用的主要依据；最近消息反映当前对话；摘要用于维持长期脉络；
   相关旧记忆用于恢复早期的具体约定。
2. 不得虚构论文、引用编号、实验数字或报告中没有的事实。
3. 引用报告观点时保留原有 [REF001] 格式；禁止创造报告中不存在的引用编号。
4. 如果不同记忆互相冲突，以用户时间更近的明确要求为准，并简要指出冲突。
5. 如果证据不足，明确指出不足，并说明需要补充哪类文献或信息。
6. 用户要求解释、比较、总结或改写时直接完成；必要时先提出一个简短澄清问题。
7. 如果用户明显切换到完全无关的新调研主题，建议新建会话，避免混淆两份调研上下文。
8. 回答使用中文 Markdown，简洁但完整，不生成参考文献列表。
"""

SUMMARY_PROMPT = """你是会话记忆压缩节点。请把旧摘要和新增的较早对话合并成一份滚动摘要。

要求：
1. 只保留未来对话可能需要的信息：用户目标、明确约束、术语定义、关键结论、已确认决定、
   报告修改要求、尚未解决的问题。
2. 明确区分用户要求与助手判断，不得增加原文没有的事实。
3. 新信息与旧信息冲突时保留最新要求，并记录发生了更新。
4. 删除寒暄、运行状态、重复内容和已经失效的临时信息。
5. 使用紧凑的中文分点，控制在 2500 字以内；直接输出摘要，不解释压缩过程。
"""

TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]+")
ENGLISH_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "for",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}
IGNORED_MEMORY_KINDS = {"error", "report_ready", "outline_confirmation", "clarification"}
MEMORY_SYNONYMS = {
    "多少字": "字数",
    "篇幅": "字数",
    "控制在": "要求",
    "目标为": "要求",
    "希望": "要求",
    "几篇论文": "文献数量",
    "多少篇论文": "文献数量",
}


@dataclass(frozen=True)
class PreparedContext:
    """The four bounded context channels sent to the answer model."""

    conversation_summary: str
    summarized_through_message_id: int
    recent_messages: list[dict[str, Any]]
    relevant_old_memories: list[dict[str, Any]]
    summary_updated: bool


def _message_payload(message: ConversationMessage) -> dict[str, Any]:
    return {
        "message_id": message.id,
        "role": message.role,
        "kind": message.kind,
        "content": message.content.strip(),
        "created_at": message.created_at,
    }


def _truncate_to_tokens(text: str, max_tokens: int, *, keep_tail: bool = False) -> str:
    """Trim text with the same conservative estimator used by the API guard."""

    if estimate_text_tokens(text) <= max_tokens:
        return text
    marker = "[前文已裁剪]\n" if keep_tail else "\n[后文已裁剪]"
    content_budget = max(max_tokens - estimate_text_tokens(marker), 0)
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        candidate = text[-middle:] if keep_tail else text[:middle]
        if estimate_text_tokens(candidate) <= content_budget:
            low = middle
        else:
            high = middle - 1
    clipped = text[-low:] if keep_tail else text[:low]
    return marker + clipped if keep_tail else clipped + marker


def _bounded_payload_items(
    items: list[dict[str, Any]],
    max_tokens: int,
    *,
    newest_first: bool,
) -> list[dict[str, Any]]:
    ordered = list(reversed(items)) if newest_first else list(items)
    selected: list[dict[str, Any]] = []
    used = 0
    for item in ordered:
        content = str(item.get("content", ""))
        remaining = max_tokens - used
        if remaining <= 0:
            break
        bounded = _truncate_to_tokens(content, remaining, keep_tail=newest_first)
        copied = dict(item)
        copied["content"] = bounded
        selected.append(copied)
        used += estimate_text_tokens(bounded)
        if bounded != content:
            break
    if newest_first:
        selected.reverse()
    return selected


def _select_report_context(
    report: str,
    query: str,
    max_tokens: int = REPORT_CONTEXT_MAX_TOKENS,
) -> str:
    """Keep a short report whole; otherwise retrieve query-relevant Markdown sections."""

    if estimate_text_tokens(report) <= max_tokens:
        return report

    heading_sections = [
        part.strip()
        for part in re.split(r"(?=^#{1,3}\s)", report, flags=re.M)
        if part.strip()
    ]
    chunks: list[str] = []
    for section in heading_sections or [report]:
        while estimate_text_tokens(section) > 2_000:
            clipped = _truncate_to_tokens(section, 2_000)
            chunks.append(clipped.removesuffix("\n[后文已裁剪]"))
            section = section[len(chunks[-1]) :].lstrip()
        if section:
            chunks.append(section)

    query_tokens = _memory_tokens(query)
    scored: list[tuple[float, int, str]] = []
    for index, chunk in enumerate(chunks):
        chunk_tokens = _memory_tokens(chunk)
        overlap = len(query_tokens & chunk_tokens)
        score = overlap / math.sqrt(max(len(query_tokens) * len(chunk_tokens), 1))
        if index == 0:
            score += 0.15
        scored.append((score, index, chunk))
    scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)

    selected: list[tuple[int, str]] = []
    used = 0
    for _, index, chunk in scored:
        remaining = max_tokens - used
        if remaining <= 0:
            break
        bounded = _truncate_to_tokens(chunk, remaining)
        selected.append((index, bounded))
        used += estimate_text_tokens(bounded)
    selected.sort(key=lambda item: item[0])
    return "\n\n[中间省略了与当前问题相关性较低的报告章节]\n\n".join(
        chunk for _, chunk in selected
    )


def _select_recent_messages(
    messages: list[ConversationMessage],
    *,
    max_messages: int = RECENT_MAX_MESSAGES,
    max_characters: int = RECENT_MAX_CHARACTERS,
) -> list[dict[str, Any]]:
    """Keep newest complete messages when possible, within a character budget."""

    selected: list[dict[str, Any]] = []
    used = 0
    for message in reversed(messages):
        if len(selected) >= max_messages:
            break
        content = message.content.strip()
        if not content:
            continue
        remaining = max_characters - used
        if remaining <= 0:
            break
        if len(content) > remaining:
            if selected:
                break
            content = content[-remaining:]
        payload = _message_payload(message)
        payload["content"] = content
        selected.append(payload)
        used += len(content)
    selected.reverse()
    return selected


def _bounded_history(
    messages: list[ConversationMessage],
    *,
    max_messages: int = RECENT_MAX_MESSAGES,
    max_characters: int = RECENT_MAX_CHARACTERS,
) -> list[dict[str, str]]:
    """Backward-compatible role/content view of the recent-message window."""

    return [
        {"role": item["role"], "content": item["content"]}
        for item in _select_recent_messages(
            messages,
            max_messages=max_messages,
            max_characters=max_characters,
        )
    ]


def _memory_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    normalized = text.casefold()
    for source, target in MEMORY_SYNONYMS.items():
        normalized = normalized.replace(source, target)
    for match in TOKEN_PATTERN.findall(normalized):
        if match.isascii():
            if match not in ENGLISH_STOP_WORDS and len(match) > 1:
                tokens.add(match)
            continue
        characters = list(match)
        if len(characters) == 1:
            tokens.add(match)
        else:
            tokens.update(
                "".join(characters[index : index + 2])
                for index in range(len(characters) - 1)
            )
    return tokens


def _retrieve_relevant_memories(
    query: str,
    candidates: list[ConversationMessage],
    *,
    limit: int = RELEVANT_MEMORY_LIMIT,
    max_characters: int = RELEVANT_MEMORY_MAX_CHARACTERS,
) -> list[dict[str, Any]]:
    """Retrieve precise older turns with lexical English/Chinese n-gram matching."""

    query_tokens = _memory_tokens(query)
    if not query_tokens:
        return []
    newest_id = max((message.id for message in candidates), default=0)
    scored: list[tuple[float, ConversationMessage]] = []
    for message in candidates:
        if message.kind in IGNORED_MEMORY_KINDS or not message.content.strip():
            continue
        candidate_tokens = _memory_tokens(message.content)
        overlap = query_tokens & candidate_tokens
        if not overlap:
            continue
        lexical_score = len(overlap) / math.sqrt(len(query_tokens) * len(candidate_tokens))
        recency_bonus = 0.05 / (1 + max(newest_id - message.id, 0))
        role_bonus = 1.08 if message.role == "user" else 1.0
        scored.append(((lexical_score + recency_bonus) * role_bonus, message))

    scored.sort(key=lambda item: (item[0], item[1].id), reverse=True)
    selected: list[dict[str, Any]] = []
    used = 0
    for score, message in scored[:limit]:
        remaining = max_characters - used
        if remaining <= 0:
            break
        content = message.content.strip()
        if len(content) > remaining:
            content = content[:remaining]
        payload = _message_payload(message)
        payload["content"] = content
        payload["relevance_score"] = round(score, 4)
        selected.append(payload)
        used += len(content)
    return selected


class FollowUpAssistant:
    """Prepare layered memory and answer grounded follow-up questions."""

    def __init__(self, client: Hy3Client) -> None:
        self.client = client

    def prepare_context(
        self,
        messages: list[ConversationMessage],
        *,
        previous_summary: str = "",
        summarized_through_message_id: int = 0,
    ) -> PreparedContext:
        recent = _select_recent_messages(messages)
        recent_ids = {int(item["message_id"]) for item in recent}
        older_messages = [message for message in messages if message.id not in recent_ids]
        unsummarized = [
            message
            for message in older_messages
            if message.id > summarized_through_message_id
            and message.kind not in IGNORED_MEMORY_KINDS
            and message.content.strip()
        ]

        summary = previous_summary.strip()
        through_id = summarized_through_message_id
        summary_updated = False
        if unsummarized:
            summary_batch: list[dict[str, Any]] = []
            summary_input_used = 0
            for message in unsummarized:
                remaining = SUMMARY_INPUT_MAX_TOKENS - summary_input_used
                if remaining <= 0:
                    break
                payload_item = _message_payload(message)
                bounded_content = _truncate_to_tokens(payload_item["content"], remaining)
                payload_item["content"] = bounded_content
                summary_batch.append(payload_item)
                summary_input_used += estimate_text_tokens(bounded_content)
                if bounded_content != message.content.strip():
                    break
            summary_payload = json.dumps(
                {
                    "previous_summary": summary,
                    "new_messages_to_compress": [
                        message for message in summary_batch
                    ],
                },
                ensure_ascii=False,
            )
            summary = self.client.chat(
                [
                    {"role": "system", "content": SUMMARY_PROMPT},
                    {"role": "user", "content": summary_payload},
                ],
                temperature=0.1,
                stage="summary",
            ).strip()
            summary = summary[:SUMMARY_MAX_CHARACTERS]
            through_id = max(int(message["message_id"]) for message in summary_batch)
            summary_updated = True

        latest_user_query = next(
            (
                message.content
                for message in reversed(messages)
                if message.role == "user" and message.content.strip()
            ),
            "",
        )
        relevant = _retrieve_relevant_memories(latest_user_query, older_messages)
        return PreparedContext(
            conversation_summary=summary,
            summarized_through_message_id=through_id,
            recent_messages=recent,
            relevant_old_memories=relevant,
            summary_updated=summary_updated,
        )

    def respond(self, report_markdown: str, context: PreparedContext) -> str:
        latest_query = next(
            (
                str(item.get("content", ""))
                for item in reversed(context.recent_messages)
                if item.get("role") == "user" and item.get("content")
            ),
            "",
        )
        payload = json.dumps(
            {
                "current_report": _select_report_context(report_markdown, latest_query),
                "conversation_summary": _truncate_to_tokens(
                    context.conversation_summary,
                    FOLLOWUP_SUMMARY_MAX_TOKENS,
                ),
                "recent_messages": _bounded_payload_items(
                    context.recent_messages,
                    FOLLOWUP_RECENT_MAX_TOKENS,
                    newest_first=True,
                ),
                "relevant_old_memories": _bounded_payload_items(
                    context.relevant_old_memories,
                    FOLLOWUP_RELEVANT_MAX_TOKENS,
                    newest_first=False,
                ),
            },
            ensure_ascii=False,
        )
        return self.client.chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": payload},
            ],
            temperature=0.2,
            reasoning_effort="low",
            stage="followup",
        )
