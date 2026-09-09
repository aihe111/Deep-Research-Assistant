"""General-purpose manager/researcher LangGraph for deep research."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from deep_research_assistant.artifact_store import ArtifactStore
from deep_research_assistant.config import Settings
from deep_research_assistant.conversation_store import ConversationStore
from deep_research_assistant.deep_research_model import build_chat_model
from deep_research_assistant.deep_research_prompts import (
    CITATION_REPAIR_PROMPT,
    CLARIFICATION_PROMPT,
    COMPRESSION_PROMPT,
    FINAL_REPORT_PROMPT,
    MANAGER_FINAL_REVIEW_PROMPT,
    REPORT_COMPLETENESS_CHECK_PROMPT,
    RESEARCH_BRIEF_PROMPT,
    RESEARCHER_PROMPT,
    SUPERVISOR_PROMPT,
)
from deep_research_assistant.deep_research_state import (
    ClarificationDecision,
    ConductResearch,
    DeepResearchInputState,
    DeepResearchState,
    ManagerOutputState,
    ManagerState,
    ReportCompletenessCheck,
    ResearchComplete,
    ResearcherOutputState,
    ResearcherState,
    ResearchQuestion,
)
from deep_research_assistant.deep_research_tools import ResearchToolProvider, think_tool
from deep_research_assistant.hy3_budget import Hy3CallBudgetManager

ModelFactory = Callable[[int], BaseChatModel]
ToolLoader = Callable[[], Awaitable[list[BaseTool]]]
FollowupHandler = Callable[[str], dict[str, Any]]


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("text"):
                parts.append(str(item["text"]))
        return "\n".join(parts)
    return str(content or "")


def _parse_outline_resume(value: Any, *, _depth: int = 0) -> dict[str, Any]:
    """Normalize Studio/API resume values into the outline decision object."""

    if isinstance(value, dict):
        return value
    if value is True:
        return {"action": "approve", "feedback": ""}
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.casefold() in {"approve", "approved"} or stripped in {"确认", "同意"}:
            return {"action": "approve", "feedback": ""}
        if not stripped or _depth >= 2:
            raise ValueError("大纲确认内容为空或嵌套格式无效")
        try:
            decoded = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "大纲确认格式无效，请提交 JSON 对象或字符串 approve"
            ) from exc
        return _parse_outline_resume(decoded, _depth=_depth + 1)
    raise ValueError(
        f"大纲确认格式无效：收到 {type(value).__name__}，需要 JSON 对象或字符串 approve"
    )


def _tool_calls(messages: Sequence[AnyMessage]) -> list[dict[str, Any]]:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return list(message.tool_calls or [])
    return []


def _is_empty_ai_response(message: AnyMessage) -> bool:
    """Detect a silent model response that must not be treated as completion."""

    return (
        isinstance(message, AIMessage)
        and not _text(message.content).strip()
        and not message.tool_calls
    )


def _is_call(call: dict[str, Any], schema: type[Any]) -> bool:
    return str(call.get("name") or "").lower() == schema.__name__.lower()


def _serialize_tool_output(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except TypeError:
        return str(value)


def _truncate_text(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    marker = "\n…（工具结果已压缩，省略中间内容）…\n"
    available = max(limit - len(marker), 0)
    head = int(available * 0.8)
    return value[:head] + marker + value[-(available - head) :]


def _compact_tool_output(value: Any, limit: int) -> str:
    """Bound tool evidence while retaining every result's source identity."""

    serialized = _serialize_tool_output(value)
    if len(serialized) <= limit:
        return serialized
    try:
        payload = json.loads(serialized)
    except (TypeError, json.JSONDecodeError):
        return _truncate_text(serialized, limit)
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        return _truncate_text(serialized, limit)

    results = payload["results"]
    compacted: list[Any] = []
    contents: list[str] = []
    for item in results:
        if not isinstance(item, dict):
            compacted.append(item)
            contents.append("")
            continue
        reduced = dict(item)
        contents.append(str(reduced.get("content") or ""))
        if "content" in reduced:
            reduced["content"] = ""
        compacted.append(reduced)
    payload["results"] = compacted

    base = json.dumps(payload, ensure_ascii=False, default=str)
    if len(base) > limit:
        # Large Tavily batches may contain rich metadata for dozens of pages.
        # Prefer keeping every title and URL over keeping metadata for only the
        # first and last pages.
        payload = {
            key: value
            for key, value in payload.items()
            if key
            in {
                "provider",
                "citation_rule",
                "queries",
                "errors",
                "candidate_count",
                "duplicates_removed",
                "retained_count",
            }
        }
        compacted = [
            {
                key: item.get(key)
                for key in ("title", "url", "evidence_level", "content")
                if isinstance(item, dict) and key in item
            }
            if isinstance(item, dict)
            else item
            for item in compacted
        ]
        payload["results"] = compacted
        base = json.dumps(payload, ensure_ascii=False, default=str)

    if len(base) > limit or not compacted:
        return _truncate_text(base, limit)

    per_result = max((limit - len(base) - 32) // len(compacted), 0)
    while True:
        for index, item in enumerate(compacted):
            if isinstance(item, dict) and "content" in item:
                item["content"] = _truncate_text(contents[index], per_result)
        serialized = json.dumps(payload, ensure_ascii=False, default=str)
        if len(serialized) <= limit or per_result <= 0:
            return serialized if len(serialized) <= limit else base
        per_result = int(per_result * 0.75)


def _bounded_notes(notes: Sequence[str], limit: int) -> str:
    """Keep the newest compact evidence notes inside a deterministic context cap."""

    selected: list[str] = []
    remaining = limit
    for note in reversed(notes):
        bounded = _truncate_text(str(note), limit)
        if selected and len(bounded) + 2 > remaining:
            continue
        selected.append(bounded[:remaining])
        remaining -= len(selected[-1]) + 2
        if remaining <= 0:
            break
    selected.reverse()
    return "\n\n".join(selected)


_SOURCE_HEADING_RE = re.compile(
    r"(?im)^[ \t]*(?:#{1,6}[ \t]*)?(主要来源|来源|Sources)[ \t]*:?[ \t]*$"
)
_NUMBERED_SOURCE_RE = re.compile(r"(?<!\w)\[(\d{1,3})\][ \t]*")
_BODY_CITATION_RE = re.compile(r"\[(\d{1,3})\]")
_LABELED_DOI_RE = re.compile(
    r"(?i)\bdoi\s*:\s*(10\.\d{4,9}/[^\s<>\[\]]+)"
)
_BARE_DOI_RE = re.compile(r"(?i)(?<![\w/])(10\.\d{4,9}/[^\s<>\[\]]+)")


def _split_source_section(report: str) -> tuple[str, str, str] | None:
    """Split a report at its final bibliography heading."""

    matches = list(_SOURCE_HEADING_RE.finditer(report))
    if not matches:
        return None
    heading = matches[-1]
    return report[: heading.start()].rstrip(), heading.group(1), report[heading.end() :]


def _parse_numbered_sources(source_text: str) -> list[tuple[int, str]]:
    """Parse numbered bibliography entries even when the model joins them."""

    matches = list(_NUMBERED_SOURCE_RE.finditer(source_text))
    entries: list[tuple[int, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(source_text)
        value = source_text[match.end() : end].strip()
        value = re.sub(r"^[\-*+]\s*", "", value).strip()
        entries.append((int(match.group(1)), " ".join(value.split())))
    return entries


def _split_doi_trailing_punctuation(candidate: str) -> tuple[str, str]:
    """Separate punctuation that belongs to prose rather than to a DOI."""

    doi = candidate
    trailing = ""
    while doi and doi[-1] in ".,;:!?，。；：！？":
        trailing = doi[-1] + trailing
        doi = doi[:-1]
    while doi.endswith(")") and doi.count(")") > doi.count("("):
        trailing = ")" + trailing
        doi = doi[:-1]
    return doi, trailing


def _canonicalize_doi_links(value: str) -> str:
    """Turn DOI labels or bare DOI identifiers into canonical HTTPS links."""

    def replace_labeled(match: re.Match[str]) -> str:
        doi, trailing = _split_doi_trailing_punctuation(match.group(1))
        return f"DOI: https://doi.org/{doi}{trailing}"

    normalized = _LABELED_DOI_RE.sub(replace_labeled, value)

    def replace_bare(match: re.Match[str]) -> str:
        prefix = normalized[max(0, match.start() - 20) : match.start()].casefold()
        if prefix.endswith("doi.org/"):
            return match.group(0)
        doi, trailing = _split_doi_trailing_punctuation(match.group(1))
        return f"https://doi.org/{doi}{trailing}"

    return _BARE_DOI_RE.sub(replace_bare, normalized)


def _normalize_numbered_sources(report: str) -> str:
    """Sort the bibliography and force exactly one numbered source per line."""

    split = _split_source_section(report)
    if split is None:
        return report.strip()
    body, heading, source_text = split
    entries = _parse_numbered_sources(source_text)
    if not entries:
        return report.strip()

    unique: dict[int, str] = {}
    for number, entry in entries:
        unique.setdefault(number, _canonicalize_doi_links(entry))
    normalized_heading = "## Sources" if heading.casefold() == "sources" else "## 主要来源"
    source_lines = [f"- [{number}] {unique[number]}" for number in sorted(unique)]
    return f"{body}\n\n{normalized_heading}\n\n" + "\n".join(source_lines)


def _repair_numbered_citations(report: str) -> str:
    """Apply safe citation-only repairs without asking a model to rewrite prose."""

    normalized = _normalize_numbered_sources(report)
    split = _split_source_section(normalized)
    if split is None:
        return normalized
    body, heading, source_text = split
    entries = _parse_numbered_sources(source_text)
    entry_by_number = {number: entry for number, entry in entries}
    citation_order: list[int] = []
    for match in _BODY_CITATION_RE.finditer(body):
        number = int(match.group(1))
        if number not in citation_order:
            citation_order.append(number)
    if not citation_order or any(number not in entry_by_number for number in citation_order):
        return normalized

    renumber = {old: new for new, old in enumerate(citation_order, start=1)}
    repaired_body = _BODY_CITATION_RE.sub(
        lambda match: f"[{renumber.get(int(match.group(1)), int(match.group(1)))}]",
        body,
    )
    normalized_heading = "## Sources" if heading.casefold() == "sources" else "## 主要来源"
    repaired_sources = [
        f"- [{renumber[number]}] {entry_by_number[number]}"
        for number in citation_order
    ]
    return f"{repaired_body}\n\n{normalized_heading}\n\n" + "\n".join(
        repaired_sources
    )


def _response_finish_reason(message: Any) -> str | None:
    """Read a provider finish reason from a LangChain message or stream chunk."""

    metadata = getattr(message, "response_metadata", None)
    if not isinstance(metadata, dict):
        return None
    reason = metadata.get("finish_reason") or metadata.get("stop_reason")
    return str(reason) if reason is not None else None


def _validate_numbered_report(
    report: str,
    *,
    finish_reason: str | None = None,
) -> list[str]:
    """Return human-readable defects for truncated or inconsistent citations."""

    errors: list[str] = []
    if not report.strip():
        return ["报告为空"]
    if finish_reason and finish_reason.casefold() in {
        "length",
        "max_tokens",
        "max_output_tokens",
    }:
        errors.append(f"模型因输出长度停止（{finish_reason}）")
    if not report.lstrip().startswith("# "):
        errors.append("缺少报告一级标题")

    split = _split_source_section(report)
    if split is None:
        errors.append("缺少主要来源章节")
        return errors
    body, _, source_text = split
    entries = _parse_numbered_sources(source_text)
    if not entries:
        errors.append("主要来源章节没有可识别的编号条目")
        return errors

    source_numbers = [number for number, _ in entries]
    unique_source_numbers = set(source_numbers)
    expected_numbers = list(range(1, len(unique_source_numbers) + 1))
    if len(source_numbers) != len(unique_source_numbers):
        errors.append("来源编号存在重复")
    if sorted(unique_source_numbers) != expected_numbers:
        errors.append("来源编号不连续或缺号")
    body_numbers = {int(value) for value in _BODY_CITATION_RE.findall(body)}
    if not body_numbers:
        errors.append("正文没有编号引用")
    elif body_numbers != unique_source_numbers:
        missing = sorted(body_numbers - unique_source_numbers)
        unused = sorted(unique_source_numbers - body_numbers)
        if missing:
            errors.append("正文引用未列入来源：" + ", ".join(f"[{n}]" for n in missing))
        if unused:
            errors.append("来源未在正文引用：" + ", ".join(f"[{n}]" for n in unused))

    for number, entry in entries:
        has_public_url = bool(re.search(r"https?://\S+", entry))
        has_private_id = bool(
            re.search(r"\bMCP\b|私有来源|原生记录|record[_ -]?id", entry, re.IGNORECASE)
        )
        if not entry or not (has_public_url or has_private_id):
            errors.append(f"来源 [{number}] 缺少完整 URL 或 MCP 记录标识")
    return errors


def _limit_research_memo(value: str, limit: int) -> str:
    """Keep a generated memo within its delivery contract and retain sources."""

    if len(value) <= limit:
        return value
    marker = "\n\n> 证据备忘录超过长度上限，已保留主要结论与末尾来源。\n\n"
    available = max(limit - len(marker), 0)
    head = int(available * 0.75)
    return value[:head] + marker + value[-(available - head) :]


def _finalize_research_memo(
    content: Any,
    research_topic: str,
    raw_notes: Sequence[str],
    limit: int,
) -> tuple[str, bool]:
    """Return a non-empty memo and report whether deterministic fallback was used."""

    generated = _text(content).strip()
    if generated:
        return _limit_research_memo(generated, limit), False
    return _deterministic_research_memo(research_topic, raw_notes, limit), True


def _deterministic_research_memo(
    research_topic: str,
    notes: Sequence[str],
    limit: int,
) -> str:
    """Produce a small evidence memo without an LLM when compression is unavailable.

    This is deliberately lossy: it keeps unique evidence fragments and a deduplicated
    source register instead of returning a large tool transcript under a misleading
    ``compressed_research`` label.
    """

    memo_limit = max(limit, 1_000)
    source_pattern = re.compile(r"https?://[^\s<>\]\[)\"']+")
    sources: list[str] = []
    seen_sources: set[str] = set()
    fragments: list[str] = []
    seen_fragments: set[str] = set()

    for note in notes:
        text = str(note or "").strip()
        for url in source_pattern.findall(text):
            normalized_url = url.rstrip(".,;:，。；：")
            if normalized_url and normalized_url not in seen_sources:
                seen_sources.add(normalized_url)
                sources.append(normalized_url)

        # Tool payloads are often one very long JSON line. Sentence/paragraph
        # boundaries yield compact evidence snippets without needing to understand
        # every provider-specific schema.
        candidates = re.split(r"(?:\r?\n){2,}|(?<=[。！？.!?])\s+", text)
        for candidate in candidates:
            compact = re.sub(r"\s+", " ", candidate).strip()
            if not compact or compact in {"{}", "[]"}:
                continue
            fingerprint = re.sub(r"\W+", "", compact).casefold()
            if not fingerprint or fingerprint in seen_fragments:
                continue
            seen_fragments.add(fingerprint)
            fragments.append(_truncate_text(compact, 1_600))

    header = (
        f"## 研究单元：{research_topic}\n\n"
        "### 确定性压缩证据\n"
        "模型压缩预算不可用；以下内容由程序去重并限长，最终写作必须降低结论强度并保留局限。\n\n"
    )
    source_block = ""
    if sources:
        source_block = "\n\n### 去重来源\n" + "\n".join(
            f"- {url}" for url in sources
        )

    evidence_budget = max(memo_limit - len(header) - len(source_block), 0)
    selected: list[str] = []
    used = 0
    for fragment in fragments:
        rendered = f"- {fragment}"
        if used + len(rendered) + 1 > evidence_budget:
            remaining = evidence_budget - used
            if remaining > 120:
                selected.append(_truncate_text(rendered, remaining))
            break
        selected.append(rendered)
        used += len(rendered) + 1

    evidence = "\n".join(selected) or "- 没有取得可保留的外部证据。"
    return _truncate_text(header + evidence + source_block, memo_limit)


def _fallback_outline(request: str) -> dict[str, Any]:
    """Provide a valid general outline when the brief call budget is unavailable."""

    return {
        "title": request.splitlines()[0][:80] or "深度研究报告",
        "thesis": request[:300] or "围绕用户问题形成可核验的综合结论。",
        "sections": [
            {
                "section_id": "S1",
                "title": "研究背景与问题边界",
                "objective": "明确核心概念、范围和判断标准。",
                "research_questions": ["该问题的关键概念、范围与评价标准是什么？"],
                "subsections": [],
            },
            {
                "section_id": "S2",
                "title": "核心证据与综合分析",
                "objective": "取得多来源证据并比较一致性与冲突。",
                "research_questions": ["现有可靠证据支持哪些结论，存在哪些冲突？"],
                "subsections": [],
            },
            {
                "section_id": "S3",
                "title": "结论、局限与建议",
                "objective": "回答研究目标并说明不确定性。",
                "research_questions": ["可以形成哪些结论，其局限和后续建议是什么？"],
                "subsections": [],
            },
        ],
    }


def build_deep_research_graph(
    settings: Settings,
    artifact_store: ArtifactStore,
    *,
    event_store: ConversationStore | None = None,
    model_factory: ModelFactory | None = None,
    tool_loader: ToolLoader | None = None,
    followup_handler: FollowupHandler | None = None,
    checkpointer: Any | None = None,
):
    """Compile the full main graph and its parallel supervisor/researcher subgraphs."""

    if model_factory is None:
        def make_model(max_tokens: int) -> BaseChatModel:
            return build_chat_model(settings, max_tokens=max_tokens)

        def make_structured_model(max_tokens: int) -> BaseChatModel:
            return build_chat_model(
                settings,
                max_tokens=max_tokens,
                disable_thinking=True,
            )

        def make_compression_model() -> BaseChatModel:
            # Evidence compression is extraction and organization rather than
            # open-ended reasoning. Disabling thinking prevents Hy3 from using
            # the entire completion budget on hidden reasoning tokens.
            return build_chat_model(
                settings,
                max_tokens=settings.compression_max_tokens,
                disable_thinking=True,
            )

        def make_no_thinking_model(max_tokens: int) -> BaseChatModel:
            return build_chat_model(
                settings,
                max_tokens=max_tokens,
                disable_thinking=True,
            )
    else:
        make_model = model_factory
        # Injected test/custom models decide their own reasoning policy.
        make_structured_model = model_factory

        def make_compression_model() -> BaseChatModel:
            return model_factory(settings.compression_max_tokens)

        # Injected models control their own reasoning behavior. Recreating the
        # model still lets tests and custom factories supply a fresh retry.
        make_no_thinking_model = model_factory
    budget_manager = Hy3CallBudgetManager(settings.max_hy3_calls_per_research, reserved=3)
    default_tool_provider = ResearchToolProvider(settings, budget_manager)

    def emit_progress(
        thread_id: str,
        event_type: str,
        stage: str,
        title: str,
        *,
        detail: str = "",
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Publish a sanitized progress event without exposing model reasoning."""

        event = {
            "event_type": event_type,
            "stage": stage,
            "title": title,
            "detail": _truncate_text(detail.strip(), 500),
            "payload": payload or {},
        }
        if event_store is not None:
            try:
                event_store.add_event(thread_id, **event)
            except Exception:
                # Progress telemetry must never make the research Run fail.
                pass
        try:
            get_stream_writer()(event)
        except RuntimeError:
            # Direct unit tests and non-streaming invocations have no writer.
            pass

    def tool_display_name(name: str) -> str:
        return {
            "openalex_search": "OpenAlex 文献检索",
            "openalex_fetch_fulltext": "OpenAlex 论文正文读取",
            "tavily_search": "Tavily 网页检索",
            "think_tool": "证据反思",
        }.get(name, name.replace("_", " "))

    async def acquire_model_call(
        thread_id: str,
        stage: str,
        *,
        essential: bool = False,
    ) -> bool:
        return await budget_manager.acquire(thread_id, stage, essential=essential)

    async def load_tools() -> list[BaseTool]:
        if tool_loader is not None:
            return await tool_loader()
        return await default_tool_provider.get_tools()

    async def researcher_agent(
        state: ResearcherState,
        config: RunnableConfig,
    ) -> dict[str, Any]:
        thread_id = state["thread_id"]
        if state.get("tool_iteration", 0) == 0:
            emit_progress(
                thread_id,
                "researcher_analyzing",
                "research",
                "Researcher 正在分析研究任务",
                detail=state["research_topic"],
            )
        if not await acquire_model_call(thread_id, "researcher"):
            return {
                "researcher_messages": [
                    AIMessage(
                        content="全局 Hy3 调用预算已达到研究阶段上限，使用现有证据收束。",
                        tool_calls=[
                            {
                                "name": "ResearchComplete",
                                "args": {},
                                "id": f"budget-complete-{state.get('tool_iteration', 0)}",
                            }
                        ],
                    )
                ],
                "tool_iteration": state.get("tool_iteration", 0) + 1,
                "hy3_call_budget_exhausted": True,
            }
        tools = await load_tools()
        awaiting_reflection = state.get("awaiting_reflection", False)
        search_tool_calls = state.get("search_tool_calls", 0)
        if awaiting_reflection:
            # The reference workflow requires a dedicated think-tool turn after
            # every search. Expose only that tool so retrieval and reflection
            # cannot be mixed in one parallel model response.
            researcher_tool_specs = [think_tool]
        elif search_tool_calls >= settings.max_search_tool_calls:
            researcher_tool_specs = [ResearchComplete]
        else:
            researcher_tool_specs = [*tools, ResearchComplete]
        model = make_model(settings.researcher_max_tokens).bind_tools(
            researcher_tool_specs,
            parallel_tool_calls=True,
        )
        system = RESEARCHER_PROMPT.format(
            research_brief=state["research_brief"],
            research_topic=state["research_topic"],
            max_search_tool_calls=settings.max_search_tool_calls,
            mcp_prompt=settings.mcp_prompt,
        )
        if awaiting_reflection:
            system += (
                "\n\n当前必须先完成一次独立反思。只调用 think_tool，概括刚取得的证据、"
                "缺口、冲突和是否应继续；本回合不得调用任何检索工具或结束工具。"
            )
        elif search_tool_calls >= settings.max_search_tool_calls:
            system += (
                "\n\n当前研究单元已经达到外部检索工具调用上限。不得继续检索；"
                "请调用 ResearchComplete，未解决的缺口交给证据备忘录说明。"
            )
        context_limit = settings.researcher_context_max_characters
        compact_context = _bounded_notes(state.get("raw_notes", []), context_limit // 2)
        messages: list[AnyMessage] = [
            SystemMessage(content=system),
            HumanMessage(content=state["research_topic"]),
        ]
        if compact_context:
            messages.append(
                HumanMessage(
                    content=(
                        "以下是前几轮工具结果的压缩记录。基于这些证据反思缺口，只在确有必要时"
                        "继续调用工具；不要要求重新返回已经取得的全文。\n\n"
                        + compact_context
                    )
                )
            )
        history = state.get("researcher_messages", [])
        last_tool_call_index = next(
            (
                index
                for index in range(len(history) - 1, -1, -1)
                if isinstance(history[index], AIMessage) and history[index].tool_calls
            ),
            -1,
        )
        if last_tool_call_index >= 0:
            recent = history[last_tool_call_index:]
            tool_messages = [item for item in recent if isinstance(item, ToolMessage)]
            per_tool_limit = max((context_limit // 2) // max(len(tool_messages), 1), 500)
            messages.append(recent[0])
            messages.extend(
                ToolMessage(
                    content=_truncate_text(_text(item.content), per_tool_limit),
                    tool_call_id=item.tool_call_id,
                    status=item.status,
                )
                for item in tool_messages
            )
        response = await model.ainvoke(
            messages,
            config={**config, "tags": [*config.get("tags", []), "researcher", "reflection"]},
        )
        response_calls = _tool_calls([response])
        if awaiting_reflection:
            reflection_calls = [
                call
                for call in response_calls
                if str(call.get("name") or "") == think_tool.name
            ]
            if not reflection_calls:
                # Some OpenAI-compatible providers do not reliably honor a
                # single-tool binding. Keep the model's own assessment and only
                # repair the tool protocol so the reflection is still recorded.
                reflection = _text(getattr(response, "content", "")).strip()
                response = AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": think_tool.name,
                            "args": {
                                "reflection": reflection
                                or "已复核刚取得的证据；下一步仅补足会改变结论的关键缺口。"
                            },
                            "id": f"forced-reflection-{state.get('tool_iteration', 0)}",
                        }
                    ],
                )
            elif len(response_calls) != len(reflection_calls):
                response = AIMessage(
                    content=_text(response.content),
                    tool_calls=reflection_calls,
                )
        else:
            external_calls = [
                call
                for call in response_calls
                if str(call.get("name") or "")
                not in {think_tool.name, ResearchComplete.__name__}
            ]
            if search_tool_calls >= settings.max_search_tool_calls and external_calls:
                response = AIMessage(
                    content="外部检索次数已经达到上限，使用现有证据结束研究单元。",
                    tool_calls=[
                        {
                            "name": ResearchComplete.__name__,
                            "args": {},
                            "id": f"search-limit-complete-{state.get('tool_iteration', 0)}",
                        }
                    ],
                )
            elif external_calls and any(
                str(call.get("name") or "") == think_tool.name
                for call in response_calls
            ):
                # Execute retrieval first. The next graph turn is the mandatory
                # isolated reflection turn.
                response = AIMessage(
                    content=_text(response.content),
                    tool_calls=external_calls,
                )
        if _is_empty_ai_response(response):
            emit_progress(
                thread_id,
                "model_empty_retry",
                "research",
                "Researcher 返回空结果，正在关闭思考后重试",
                detail=state["research_topic"],
                payload={"stage": "researcher"},
            )
            retry_allowed = await acquire_model_call(thread_id, "researcher_empty_retry")
            if retry_allowed:
                retry_model = make_no_thinking_model(
                    settings.researcher_max_tokens
                ).bind_tools(
                    researcher_tool_specs,
                    parallel_tool_calls=True,
                )
                response = await retry_model.ainvoke(
                    messages,
                    config={
                        **config,
                        "tags": [
                            *config.get("tags", []),
                            "researcher",
                            "reflection",
                            "empty-retry",
                            "no-thinking",
                        ],
                    },
                )
            if _is_empty_ai_response(response):
                if state.get("raw_notes"):
                    response = AIMessage(
                        content="模型连续返回空结果，使用已取得的工具证据进入压缩。",
                        tool_calls=[
                            {
                                "name": "ResearchComplete",
                                "args": {},
                                "id": f"researcher-empty-fallback-{state.get('tool_iteration', 0)}",
                            }
                        ],
                    )
                else:
                    raise RuntimeError(
                        "Researcher 连续返回空结果，且尚未取得任何证据；已停止本次运行，"
                        "避免生成误导性的无证据报告。"
                    )
        return {
            # ResearcherState uses replacement semantics: only the current model
            # response or the current AI/tool protocol pair remains in state.
            "researcher_messages": [response],
            "tool_iteration": state.get("tool_iteration", 0) + 1,
        }

    def route_researcher(state: ResearcherState) -> str:
        calls = _tool_calls(state.get("researcher_messages", []))
        if not calls:
            return "compress_research"
        if any(_is_call(call, ResearchComplete) for call in calls):
            return "compress_research"
        return "researcher_tools"

    async def researcher_tools(
        state: ResearcherState,
        config: RunnableConfig,
    ) -> dict[str, Any]:
        tools = {item.name: item for item in await load_tools()}
        # ``think_tool`` is part of the graph protocol even when tests or custom
        # deployments inject only their domain-specific retrieval tools.
        tools.setdefault(think_tool.name, think_tool)
        calls = _tool_calls(state.get("researcher_messages", []))

        async def execute(
            call: dict[str, Any],
        ) -> tuple[ToolMessage, str | None, str | None]:
            name = str(call.get("name") or "")
            call_id = str(call.get("id") or name)
            if _is_call(call, ResearchComplete):
                return (
                    ToolMessage(content="已接受研究完成信号。", tool_call_id=call_id),
                    None,
                    None,
                )
            selected = tools.get(name)
            if selected is None:
                error = f"当前配置中没有可用工具“{name}”。"
                return (
                    ToolMessage(content=error, tool_call_id=call_id, status="error"),
                    f"## {name} 调用失败\n{error}",
                    None,
                )
            try:
                args = call.get("args") or {}
                visible_input = str(
                    args.get("query")
                    or args.get("url")
                    or args.get("work_id")
                    or state["research_topic"]
                )
                emit_progress(
                    state["thread_id"],
                    "tool_started",
                    "research",
                    f"正在使用 {tool_display_name(name)}",
                    detail=visible_input,
                    payload={"tool": name, "research_topic": state["research_topic"]},
                )
                token = budget_manager.bind(state["thread_id"])
                try:
                    result = await selected.ainvoke(call.get("args") or {}, config=config)
                finally:
                    budget_manager.reset_binding(token)
                content = _compact_tool_output(
                    result,
                    settings.researcher_tool_result_max_characters,
                )
                if name == think_tool.name:
                    reflection = str((call.get("args") or {}).get("reflection") or "").strip()
                    note = f"## 研究员反思\n{reflection}" if reflection else None
                else:
                    note = f"## 工具：{name}\n{content}"
                emit_progress(
                    state["thread_id"],
                    "tool_completed",
                    "research",
                    f"{tool_display_name(name)} 已完成",
                    detail=state["research_topic"],
                    payload={"tool": name, "research_topic": state["research_topic"]},
                )
                return ToolMessage(content=content, tool_call_id=call_id), note, name
            except Exception as exc:  # A failed source must not discard the whole research unit.
                error = f"{type(exc).__name__}: {exc}"
                emit_progress(
                    state["thread_id"],
                    "tool_failed",
                    "research",
                    f"{tool_display_name(name)} 调用失败",
                    detail=f"{state['research_topic']}；{error}",
                    payload={"tool": name, "error": _truncate_text(error, 300)},
                )
                return (
                    ToolMessage(content=error, tool_call_id=call_id, status="error"),
                    f"## {name} 调用失败\n{error}",
                    name,
                )

        outcomes = await asyncio.gather(*(execute(call) for call in calls))
        external_call_count = sum(
            1
            for call in calls
            if str(call.get("name") or "")
            not in {think_tool.name, ResearchComplete.__name__}
        )
        reflected = any(
            str(call.get("name") or "") == think_tool.name for call in calls
        )
        current_ai = next(
            (
                message
                for message in reversed(state.get("researcher_messages", []))
                if isinstance(message, AIMessage)
            ),
            None,
        )
        protocol_messages: list[AnyMessage] = []
        if current_ai is not None:
            protocol_messages.append(current_ai)
        protocol_messages.extend(item[0] for item in outcomes)
        return {
            "researcher_messages": protocol_messages,
            "raw_notes": [item[1] for item in outcomes if item[1]],
            "tools_used": [item[2] for item in outcomes if item[2]],
            "search_tool_calls": state.get("search_tool_calls", 0)
            + external_call_count,
            "awaiting_reflection": True if external_call_count else (
                False if reflected else state.get("awaiting_reflection", False)
            ),
        }

    def route_after_researcher_tools(state: ResearcherState) -> str:
        if state.get("awaiting_reflection", False):
            return "researcher"
        if state.get("search_tool_calls", 0) >= settings.max_search_tool_calls:
            return "compress_research"
        if state.get("tool_iteration", 0) >= settings.max_researcher_iterations:
            return "compress_research"
        return "researcher"

    async def compress_research(
        state: ResearcherState,
        config: RunnableConfig,
    ) -> dict[str, Any]:
        emit_progress(
            state["thread_id"],
            "compression_started",
            "compression",
            "正在整理证据备忘录",
            detail=state["research_topic"],
        )
        raw_notes = _bounded_notes(
            state.get("raw_notes", []),
            settings.compression_input_max_characters,
        )
        if not raw_notes:
            raw_notes = "没有取得可用的外部来源；请在证据备忘录中明确说明这一限制。"
        if not await budget_manager.acquire_reserved(
            state["thread_id"],
            "research_compression",
        ):
            result = {
                "compressed_research": _deterministic_research_memo(
                    state["research_topic"],
                    state.get("raw_notes", []),
                    settings.research_memo_max_characters,
                ),
                "hy3_call_budget_exhausted": True,
            }
            emit_progress(
                state["thread_id"],
                "compression_completed",
                "compression",
                "证据备忘录已完成",
                detail=state["research_topic"],
                payload={"fallback": True},
            )
            return result
        prompt = COMPRESSION_PROMPT.format(
            research_brief=state["research_brief"],
            research_topic=state["research_topic"],
            raw_notes=raw_notes,
            memo_target_characters=settings.research_memo_target_characters,
            memo_max_characters=settings.research_memo_max_characters,
        )
        response = await make_compression_model().ainvoke(
            [SystemMessage(content=prompt)],
            config={**config, "tags": [*config.get("tags", []), "research-compression"]},
        )
        memo, used_fallback = _finalize_research_memo(
            response.content,
            state["research_topic"],
            state.get("raw_notes", []),
            settings.research_memo_max_characters,
        )
        result = {"compressed_research": memo}
        emit_progress(
            state["thread_id"],
            "compression_completed",
            "compression",
            "证据备忘录已完成",
            detail=state["research_topic"],
            payload={"fallback": used_fallback},
        )
        return result

    researcher_builder = StateGraph(
        ResearcherState,
        output_schema=ResearcherOutputState,
    )
    researcher_builder.add_node("researcher", researcher_agent)
    researcher_builder.add_node("researcher_tools", researcher_tools)
    researcher_builder.add_node("compress_research", compress_research)
    researcher_builder.add_edge(START, "researcher")
    researcher_builder.add_conditional_edges(
        "researcher",
        route_researcher,
        {"researcher_tools": "researcher_tools", "compress_research": "compress_research"},
    )
    researcher_builder.add_conditional_edges(
        "researcher_tools",
        route_after_researcher_tools,
        {"researcher": "researcher", "compress_research": "compress_research"},
    )
    researcher_builder.add_edge("compress_research", END)
    researcher_graph = researcher_builder.compile()

    async def manager(
        state: ManagerState,
        config: RunnableConfig,
    ) -> dict[str, Any]:
        iteration = state.get("manager_iteration", 0) + 1
        final_review_reserved = state.get("manager_final_review_reserved", False)
        if not final_review_reserved:
            # Protect the mandatory final review before any parallel
            # researcher can consume the ordinary call pool. An empty-response
            # retry may use remaining ordinary capacity, but cannot consume the
            # two calls protected for final report delivery.
            final_review_reserved = (
                await budget_manager.reserve_stage(
                    state["thread_id"],
                    "research_manager_final_review",
                    1,
                )
                == 1
            )
        emit_progress(
            state["thread_id"],
            "manager_reviewing",
            "management",
            "Manager 正在复核证据覆盖",
            detail=f"第 {iteration} 轮研究决策",
        )
        if iteration > settings.max_supervisor_iterations:
            return {
                "manager_messages": [
                    AIMessage(
                        content="统筹循环已达到上限，请使用现有证据进入综合写作。"
                    )
                ],
                "manager_iteration": iteration,
                "manager_final_review_reserved": final_review_reserved,
            }
        if iteration > 1 and state.get("manager_review_reserved", False):
            manager_call_acquired = await budget_manager.acquire_reserved(
                state["thread_id"],
                "research_manager_review",
            )
        else:
            manager_call_acquired = await acquire_model_call(
                state["thread_id"], "research_manager"
            )
        if not manager_call_acquired:
            return {
                "manager_messages": [
                    AIMessage(
                        content="全局 Hy3 调用预算已达到研究阶段上限，进入综合写作。",
                        tool_calls=[
                            {
                                "name": "ResearchComplete",
                                "args": {},
                                "id": f"supervisor-budget-{iteration}",
                            }
                        ],
                    )
                ],
                "manager_iteration": iteration,
                "manager_final_review_reserved": final_review_reserved,
                "hy3_call_budget_exhausted": True,
            }
        manager_tool_specs = [ConductResearch, ResearchComplete, think_tool]
        model = make_model(settings.supervisor_max_tokens).bind_tools(
            manager_tool_specs,
            parallel_tool_calls=True,
        )
        stored_manager_messages = state.get("manager_messages", [])
        manager_context: list[AnyMessage] = [
            SystemMessage(
                content=SUPERVISOR_PROMPT.format(research_brief=state["research_brief"])
            ),
            *(
                stored_manager_messages
                or [HumanMessage(content=state["research_brief"])]
            ),
        ]
        compact_notes = _bounded_notes(
            state.get("notes", []),
            min(
                settings.researcher_context_max_characters,
                settings.final_report_context_max_characters,
            ),
        )
        if compact_notes:
            manager_context.append(
                HumanMessage(
                    content=(
                        "以下是已经完成的研究单元证据备忘录。请据此复核证据缺口；"
                        "不要要求研究员重复返回已有来源。\n\n" + compact_notes
                    )
                )
            )
        response = await model.ainvoke(
            manager_context,
            config={**config, "tags": [*config.get("tags", []), "research-manager"]},
        )
        if _is_empty_ai_response(response):
            emit_progress(
                state["thread_id"],
                "model_empty_retry",
                "management",
                "Manager 返回空结果，正在关闭思考后重试",
                detail=f"第 {iteration} 轮研究决策",
                payload={"stage": "research_manager"},
            )
            retry_allowed = await acquire_model_call(
                state["thread_id"],
                "research_manager_empty_retry",
            )
            if retry_allowed:
                retry_model = make_no_thinking_model(
                    settings.supervisor_max_tokens
                ).bind_tools(
                    manager_tool_specs,
                    parallel_tool_calls=True,
                )
                response = await retry_model.ainvoke(
                    manager_context,
                    config={
                        **config,
                        "tags": [
                            *config.get("tags", []),
                            "research-manager",
                            "empty-retry",
                            "no-thinking",
                        ],
                    },
                )
            if _is_empty_ai_response(response):
                if state.get("notes"):
                    response = AIMessage(
                        content="Manager 连续返回空结果，使用已取得证据进入综合写作。",
                        tool_calls=[
                            {
                                "name": "ResearchComplete",
                                "args": {},
                                "id": f"manager-empty-complete-{iteration}",
                            }
                        ],
                    )
                else:
                    response = AIMessage(
                        content="Manager 连续返回空结果，启用保底研究任务以取得外部证据。",
                        tool_calls=[
                            {
                                "name": "ConductResearch",
                                "args": {
                                    "research_topic": (
                                        "围绕以下完整研究简报取得至少两个可核验来源，"
                                        "覆盖核心事实、差异、局限与反方证据：\n"
                                        + state["research_brief"]
                                    )
                                },
                                "id": f"manager-empty-research-{iteration}",
                            }
                        ],
                    )
                emit_progress(
                    state["thread_id"],
                    "model_empty_fallback",
                    "management",
                    "Manager 连续空响应，已启用确定性保底决策",
                    payload={"has_existing_evidence": bool(state.get("notes"))},
                )
        delegated = [
            call for call in _tool_calls([response]) if _is_call(call, ConductResearch)
        ]
        if delegated:
            emit_progress(
                state["thread_id"],
                "manager_delegated",
                "management",
                f"Manager 启动 {len(delegated)} 个并行研究单元",
                detail="将分别检索互不重复的证据维度",
                payload={"count": len(delegated)},
            )
        elif any(_is_call(call, ResearchComplete) for call in _tool_calls([response])):
            emit_progress(
                state["thread_id"],
                "manager_completed",
                "management",
                "Manager 已确认研究证据可以进入写作",
            )
        return {
            "manager_messages": [response],
            "manager_iteration": iteration,
            "manager_review_reserved": False if iteration > 1 else state.get(
                "manager_review_reserved", False
            ),
            "manager_final_review_reserved": final_review_reserved,
        }

    async def manager_final_review(
        state: ManagerState,
        config: RunnableConfig,
    ) -> dict[str, Any]:
        """Review every completed memo after delegation has permanently stopped."""

        emit_progress(
            state["thread_id"],
            "manager_final_reviewing",
            "management",
            "Manager 正在最终复核全部研究结果",
            detail=f"共收到 {len(state.get('notes', []))} 份证据备忘录",
        )
        notes = _bounded_notes(
            state.get("notes", []),
            min(
                settings.researcher_context_max_characters,
                settings.final_report_context_max_characters,
            ),
        )
        prompt = MANAGER_FINAL_REVIEW_PROMPT.format(
            research_brief=state["research_brief"],
            notes=notes or "没有取得可用证据备忘录。",
        )
        acquired = await budget_manager.acquire_reserved(
            state["thread_id"],
            "research_manager_final_review",
        )
        review = ""
        if acquired:
            response = await make_model(settings.supervisor_max_tokens).ainvoke(
                [SystemMessage(content=prompt)],
                config={
                    **config,
                    "tags": [
                        *config.get("tags", []),
                        "research-manager-final-review",
                    ],
                },
            )
            review = _text(response.content).strip()
        if not review:
            retry_acquired = await acquire_model_call(
                state["thread_id"],
                "research_manager_final_review_empty_retry",
            )
            if retry_acquired:
                response = await make_no_thinking_model(
                    settings.supervisor_max_tokens
                ).ainvoke(
                    [SystemMessage(content=prompt)],
                    config={
                        **config,
                        "tags": [
                            *config.get("tags", []),
                            "research-manager-final-review",
                            "empty-retry",
                            "no-thinking",
                        ],
                    },
                )
                review = _text(response.content).strip()
        if not review:
            review = (
                "现有研究结果已全部送达报告阶段，但 Manager 最终复核未返回有效文本。"
                "报告只能依据证据备忘录写作，并应明确保留其中记录的证据缺口与不确定性。"
            )
        emit_progress(
            state["thread_id"],
            "manager_final_review_completed",
            "management",
            "Manager 已完成全部研究结果的最终复核",
            payload={"memo_count": len(state.get("notes", []))},
        )
        return {
            "notes": ["## Manager 最终复核意见\n\n" + review],
            "manager_final_review_reserved": False,
            "hy3_call_budget_exhausted": (
                state.get("hy3_call_budget_exhausted", False) or not acquired
            ),
        }

    def route_manager(state: ManagerState) -> str:
        calls = _tool_calls(state.get("manager_messages", []))
        if not calls or state.get("manager_iteration", 0) > settings.max_supervisor_iterations:
            return "manager_final_review"
        return "tools"

    async def tools(
        state: ManagerState,
        config: RunnableConfig,
    ) -> dict[str, Any]:
        calls = _tool_calls(state.get("manager_messages", []))
        requested_research_calls = [
            call for call in calls if _is_call(call, ConductResearch)
        ][: settings.max_concurrent_research_units]
        # A delegated unit must retain enough budget to produce its evidence
        # memo. Reserve compression before an optional intermediate Manager
        # review; the separate mandatory final review is already protected.
        reserved_compressions = await budget_manager.reserve_stage(
            state["thread_id"],
            "research_compression",
            len(requested_research_calls),
        )
        research_calls = requested_research_calls[:reserved_compressions]
        compression_reservation_shortfall = (
            reserved_compressions < len(requested_research_calls)
        )
        manager_review_reserved = state.get("manager_review_reserved", False)
        if (
            research_calls
            and not manager_review_reserved
            and state.get("manager_iteration", 0) < settings.max_supervisor_iterations
        ):
            manager_review_reserved = bool(
                await budget_manager.reserve_stage(
                    state["thread_id"],
                    "research_manager_review",
                    1,
                )
            )

        async def conduct(call: dict[str, Any]) -> tuple[dict[str, Any], str]:
            topic = str((call.get("args") or {}).get("research_topic") or "").strip()
            emit_progress(
                state["thread_id"],
                "researcher_started",
                "research",
                "Researcher 已开始工作",
                detail=topic,
            )
            result = await researcher_graph.ainvoke(
                {
                    "thread_id": state["thread_id"],
                    "research_topic": topic,
                    "research_brief": state["research_brief"],
                    "raw_notes": [],
                    "tools_used": [],
                    "tool_iteration": 0,
                    "search_tool_calls": 0,
                    "awaiting_reflection": False,
                },
                config=config,
            )
            emit_progress(
                state["thread_id"],
                "researcher_completed",
                "research",
                "Researcher 已完成研究单元",
                detail=topic,
                payload={"tools": list(dict.fromkeys(result.get("tools_used", [])))},
            )
            return result, str(call.get("id") or "conduct-research")

        results = await asyncio.gather(*(conduct(call) for call in research_calls))
        completed_by_id = {call_id: result for result, call_id in results}
        messages: list[ToolMessage] = []
        notes: list[str] = []
        used: list[str] = []
        for call in calls:
            call_id = str(call.get("id") or call.get("name") or "tool")
            if _is_call(call, ConductResearch):
                result = completed_by_id.get(call_id)
                if result is None:
                    messages.append(
                        ToolMessage(
                            content=(
                                "本轮并行研究单元或专属压缩额度已达到上限，该委派未执行。"
                            ),
                            tool_call_id=call_id,
                        )
                    )
                    continue
                memo = str(result.get("compressed_research") or "研究员没有返回证据备忘录。")
                notes.append(memo)
                used.extend(result.get("tools_used", []))
                memo_index = len(state.get("notes", [])) + len(notes)
                messages.append(
                    ToolMessage(
                        content=f"研究单元已完成，证据备忘录已保存为 #{memo_index}。",
                        tool_call_id=call_id,
                    )
                )
            elif _is_call(call, ResearchComplete):
                messages.append(ToolMessage(content="已接受完成信号。", tool_call_id=call_id))
            elif str(call.get("name") or "") == think_tool.name:
                reflection = str((call.get("args") or {}).get("reflection") or "")
                messages.append(
                    ToolMessage(content=f"已记录反思：{reflection}", tool_call_id=call_id)
                )
            else:
                messages.append(
                    ToolMessage(
                        content="统筹节点调用了不受支持的工具。",
                        tool_call_id=call_id,
                        status="error",
                    )
                )
        return {
            "manager_messages": messages,
            "notes": notes,
            "research_unit_count": state.get("research_unit_count", 0) + len(results),
            "tools_used": used,
            "manager_review_reserved": manager_review_reserved,
            "hy3_call_budget_exhausted": (
                state.get("hy3_call_budget_exhausted", False)
                or compression_reservation_shortfall
                or any(result.get("hy3_call_budget_exhausted", False) for result, _ in results)
            ),
        }

    def route_after_tools(state: ManagerState) -> str:
        calls = _tool_calls(state.get("manager_messages", []))
        if any(_is_call(call, ResearchComplete) for call in calls):
            return "manager_final_review"
        if state.get("manager_iteration", 0) >= settings.max_supervisor_iterations:
            return "manager_final_review"
        return "manager"

    manager_builder = StateGraph(
        ManagerState,
        output_schema=ManagerOutputState,
    )
    manager_builder.add_node("manager", manager)
    manager_builder.add_node("tools", tools)
    manager_builder.add_node("manager_final_review", manager_final_review)
    manager_builder.add_edge(START, "manager")
    manager_builder.add_conditional_edges(
        "manager",
        route_manager,
        {"tools": "tools", "manager_final_review": "manager_final_review"},
    )
    manager_builder.add_conditional_edges(
        "tools",
        route_after_tools,
        {"manager": "manager", "manager_final_review": "manager_final_review"},
    )
    manager_builder.add_edge("manager_final_review", END)
    manager_graph = manager_builder.compile()

    async def clarify_with_user(
        state: DeepResearchState,
        config: RunnableConfig,
    ) -> dict[str, Any]:
        if state.get("mode") == "followup":
            emit_progress(
                state["thread_id"],
                "followup_started",
                "followup",
                "正在结合报告回答追问",
            )
            if followup_handler is None:
                return {"error": "尚未配置报告追问处理器。", "status": "failed"}
            result = await asyncio.to_thread(followup_handler, state["thread_id"])
            emit_progress(
                state["thread_id"],
                "followup_completed",
                "followup",
                "报告追问已回答",
            )
            return {
                "followup_answer": result["answer"],
                "followup_metadata": result.get("metadata", {}),
                "status": "complete",
            }
        await budget_manager.ensure(state["thread_id"])
        emit_progress(
            state["thread_id"],
            "clarification_started",
            "clarification",
            "正在明确研究意图",
        )
        if not settings.allow_clarification or state.get("clarification_completed"):
            emit_progress(
                state["thread_id"],
                "clarification_completed",
                "clarification",
                "研究意图已经明确",
            )
            return {"clarification_completed": True}
        if not await acquire_model_call(state["thread_id"], "clarification"):
            return {
                "clarification_completed": True,
                "hy3_call_budget_exhausted": True,
            }
        model = make_structured_model(
            settings.research_brief_max_tokens
        ).with_structured_output(
            ClarificationDecision,
            method="json_schema",
        )
        decision = await model.ainvoke(
            [SystemMessage(content=CLARIFICATION_PROMPT), *state.get("messages", [])],
            config={**config, "tags": [*config.get("tags", []), "clarification"]},
        )
        if not decision.needs_clarification:
            emit_progress(
                state["thread_id"],
                "clarification_completed",
                "clarification",
                "研究意图已经明确",
            )
            return {"clarification_completed": True}
        emit_progress(
            state["thread_id"],
            "input_required",
            "clarification",
            "需要补充一项研究要求",
        )
        resumed = interrupt(
            {
                "kind": "clarification",
                "questions": [decision.question],
                "round": 1,
                "max_rounds": 1,
            }
        )
        answers = resumed.get("answers", []) if isinstance(resumed, dict) else []
        emit_progress(
            state["thread_id"],
            "clarification_completed",
            "clarification",
            "补充信息已接收",
        )
        return {
            "messages": [HumanMessage(content="用户补充信息：" + "\n".join(map(str, answers)))],
            "clarification_completed": True,
        }

    def route_after_clarification(state: DeepResearchState) -> str:
        return "__end__" if state.get("mode") == "followup" else "write_research_outline"

    async def write_research_outline(
        state: DeepResearchState,
        config: RunnableConfig,
    ) -> dict[str, Any]:
        emit_progress(
            state["thread_id"],
            "outline_started",
            "outline",
            "正在生成研究大纲",
        )
        if not await acquire_model_call(state["thread_id"], "research_brief"):
            request = state.get("original_request") or _text(
                state.get("messages", [])[-1].content if state.get("messages") else ""
            )
            result = {
                "research_brief": request,
                "research_outline": state.get("research_outline") or _fallback_outline(request),
                "outline_confirmed": False,
                "notes": [],
                "research_unit_count": 0,
                "hy3_call_budget_exhausted": True,
            }
            emit_progress(
                state["thread_id"],
                "outline_completed",
                "outline",
                "研究大纲已生成",
                payload={"fallback": True},
            )
            return result
        model = make_structured_model(
            settings.research_brief_max_tokens
        ).with_structured_output(
            ResearchQuestion,
            method="json_schema",
        )
        result = await model.ainvoke(
            [SystemMessage(content=RESEARCH_BRIEF_PROMPT), *state.get("messages", [])],
            config={**config, "tags": [*config.get("tags", []), "research-brief"]},
        )
        output = {
            "research_brief": result.research_brief,
            "research_outline": result.research_outline.model_dump(mode="json"),
            "outline_confirmed": False,
            "notes": [],
            "research_unit_count": 0,
        }
        emit_progress(
            state["thread_id"],
            "outline_completed",
            "outline",
            "研究大纲已生成",
            detail=f"共 {len(result.research_outline.sections)} 个主要章节",
            payload={"section_count": len(result.research_outline.sections)},
        )
        return output

    async def confirm_outline_with_user(state: DeepResearchState) -> dict[str, Any]:
        """Pause for approval; revision feedback returns to the same brief node."""

        outline = state["research_outline"]
        if not settings.require_outline_confirmation:
            approved_brief = (
                state["research_brief"]
                + "\n\n用户确认的报告大纲：\n"
                + json.dumps(outline, ensure_ascii=False)
            )
            emit_progress(
                state["thread_id"],
                "outline_approved",
                "outline",
                "研究大纲已确认",
            )
            return {
                "research_brief": approved_brief,
                "outline_confirmed": True,
            }

        emit_progress(
            state["thread_id"],
            "outline_confirmation_required",
            "outline",
            "等待确认研究大纲",
        )
        resumed = _parse_outline_resume(
            interrupt(
                {
                    "kind": "outline_confirmation",
                    "outline": outline,
                    "message": "请确认大纲；也可以给出修改意见后重新生成。",
                    "revision_count": state.get("outline_revision_count", 0),
                    "max_revisions": settings.max_outline_revisions,
                }
            )
        )
        action = str(resumed.get("action") or "").strip().casefold()
        feedback = str(resumed.get("feedback") or "").strip()
        if action not in {"approve", "revise"}:
            raise ValueError("大纲操作必须是 approve 或 revise")
        if action == "revise" and not feedback:
            raise ValueError("修改大纲时必须提供 feedback")

        revision_count = state.get("outline_revision_count", 0)
        if action == "revise" and revision_count >= settings.max_outline_revisions:
            await budget_manager.release(state["thread_id"])
            return {
                "outline_confirmed": False,
                "status": "failed",
                "error": (
                    f"大纲已经修改 {settings.max_outline_revisions} 次，仍未确认。"
                    "请新建会话并在初始需求中明确大纲要求。"
                ),
            }
        if action == "revise":
            emit_progress(
                state["thread_id"],
                "outline_revision_requested",
                "outline",
                "正在根据反馈修改研究大纲",
            )
            return {
                "messages": [HumanMessage(content="用户对大纲的修改意见：" + feedback)],
                "outline_confirmed": False,
                "outline_revision_count": revision_count + 1,
                "status": "outline_revision_requested",
            }

        approved_brief = (
            state["research_brief"]
            + "\n\n用户确认的报告大纲：\n"
            + json.dumps(outline, ensure_ascii=False)
        )
        emit_progress(
            state["thread_id"],
            "outline_approved",
            "outline",
            "研究大纲已确认，开始检索",
        )
        return {
            "research_brief": approved_brief,
            "outline_confirmed": True,
            "status": "outline_confirmed",
        }

    def route_after_outline_confirmation(state: DeepResearchState) -> str:
        if state.get("status") == "failed":
            return "__end__"
        return "research_manager" if state.get("outline_confirmed") else "write_research_outline"

    async def report_generation(
        state: DeepResearchState,
        config: RunnableConfig,
    ) -> dict[str, Any]:
        emit_progress(
            state["thread_id"],
            "report_started",
            "report",
            "正在撰写最终报告",
        )
        artifact_store.put_text(
            state["thread_id"],
            "research_report_draft",
            "",
            content_type="text/markdown",
        )
        notes = _bounded_notes(
            state.get("notes", []),
            settings.final_report_context_max_characters,
        )
        prompt = FINAL_REPORT_PROMPT.format(
            research_brief=state["research_brief"],
            notes=notes or "没有生成任何证据备忘录，请在报告中明确说明这一限制。",
        )
        acquired = await acquire_model_call(
            state["thread_id"],
            "final_report",
            essential=True,
        )
        report = ""
        finish_reason: str | None = None
        completeness_check: dict[str, Any] | None = None
        if acquired:
            report_model = make_model(settings.final_report_max_tokens)
            report_config = {
                **config,
                "tags": [*config.get("tags", []), "final-report"],
            }
            report_parts: list[str] = []
            persisted_length = 0
            if hasattr(report_model, "astream"):
                async for chunk in report_model.astream(
                    [SystemMessage(content=prompt)],
                    config=report_config,
                ):
                    finish_reason = _response_finish_reason(chunk) or finish_reason
                    text_chunk = _text(getattr(chunk, "content", chunk))
                    if not text_chunk:
                        continue
                    report_parts.append(text_chunk)
                    current = "".join(report_parts)
                    if len(current) - persisted_length >= 160:
                        artifact_store.put_text(
                            state["thread_id"],
                            "research_report_draft",
                            current,
                            content_type="text/markdown",
                        )
                        persisted_length = len(current)
                report = "".join(report_parts)
            else:
                response = await report_model.ainvoke(
                    [SystemMessage(content=prompt)],
                    config=report_config,
                )
                report = _text(response.content)
                finish_reason = _response_finish_reason(response)

            report = _repair_numbered_citations(report)
            validation_errors = _validate_numbered_report(
                report,
                finish_reason=finish_reason,
            )
            if validation_errors:
                empty_response = not report.strip()
                retry_stage = (
                    "final_report_empty_retry"
                    if empty_response
                    else "final_report_validation_retry"
                )
                emit_progress(
                    state["thread_id"],
                    "model_empty_retry" if empty_response else "report_validation_retry",
                    "report",
                    (
                        "最终报告返回空结果，正在关闭思考后重试"
                        if empty_response
                        else "最终报告引用不完整，正在关闭思考后局部修复引用"
                    ),
                    detail="；".join(validation_errors),
                    payload={"stage": "final_report", "errors": validation_errors},
                )
                retry_acquired = await acquire_model_call(
                    state["thread_id"],
                    retry_stage,
                    essential=True,
                )
                if retry_acquired:
                    retry_model = make_no_thinking_model(
                        settings.final_report_max_tokens
                    )
                    retry_prompt = (
                        prompt
                        if empty_response
                        else CITATION_REPAIR_PROMPT.format(
                            validation_errors="；".join(validation_errors),
                            report=report,
                            notes=notes or "没有可用于补充引用的证据备忘录。",
                        )
                    )
                    response = await retry_model.ainvoke(
                        [SystemMessage(content=retry_prompt)],
                        config={
                            **report_config,
                            "tags": [
                                *report_config.get("tags", []),
                                "empty-retry" if empty_response else "validation-retry",
                                "no-thinking",
                            ],
                        },
                    )
                    report = _text(response.content)
                    finish_reason = _response_finish_reason(response)
                    report = _repair_numbered_citations(report)
                    validation_errors = _validate_numbered_report(
                        report,
                        finish_reason=finish_reason,
                    )
                elif not empty_response:
                    validation_errors.append("Hy3 调用预算不足，无法执行完整性重写")
        else:
            validation_errors = ["Hy3 调用预算已经耗尽，无法撰写最终报告"]

        if not validation_errors:
            emit_progress(
                state["thread_id"],
                "report_completeness_check_started",
                "report",
                "正在检查最终报告内容完整性",
            )
            check_acquired = await acquire_model_call(
                state["thread_id"],
                "report_completeness_check",
                essential=True,
            )
            if not check_acquired:
                validation_errors.append("Hy3 调用预算不足，无法执行报告完整性检查")
            else:
                question = str(state.get("original_request") or "").strip()
                if not question and state.get("messages"):
                    question = _text(state["messages"][-1].content).strip()
                check_input = {
                    "question": question,
                    "research_brief": state.get("research_brief", ""),
                    "research_outline": state.get("research_outline", {}),
                    "final_report": report,
                }
                try:
                    checker = make_structured_model(
                        settings.research_brief_max_tokens
                    ).with_structured_output(
                        ReportCompletenessCheck,
                        method="json_schema",
                    )
                    raw_check = await checker.ainvoke(
                        [
                            SystemMessage(content=REPORT_COMPLETENESS_CHECK_PROMPT),
                            HumanMessage(
                                content=json.dumps(check_input, ensure_ascii=False)
                            ),
                        ],
                        config={
                            **config,
                            "tags": [
                                *config.get("tags", []),
                                "report-completeness-check",
                                "no-thinking",
                            ],
                        },
                    )
                    checked = (
                        raw_check
                        if isinstance(raw_check, ReportCompletenessCheck)
                        else ReportCompletenessCheck.model_validate(raw_check)
                    )
                    completeness_check = checked.model_dump(mode="json")
                    artifact_store.put_text(
                        state["thread_id"],
                        "report_completeness_check",
                        json.dumps(completeness_check, ensure_ascii=False, indent=2),
                        content_type="application/json",
                    )
                    passed = (
                        checked.score >= 3
                        and checked.is_complete
                        and not checked.truncation_detected
                        and not checked.missing_sections
                        and not checked.structural_defects
                    )
                    emit_progress(
                        state["thread_id"],
                        "report_completeness_check_completed",
                        "report",
                        "最终报告内容完整性检查已完成",
                        detail=checked.reason,
                        payload={
                            "score": checked.score,
                            "is_complete": checked.is_complete,
                            "passed": passed,
                            "truncation_detected": checked.truncation_detected,
                        },
                    )
                    if not passed:
                        validation_errors.append(
                            "LLM 完整性检查未通过："
                            f"score={checked.score}；{checked.reason}"
                        )
                except Exception as exc:
                    validation_errors.append(
                        "LLM 完整性检查调用失败："
                        f"{type(exc).__name__}: {exc}"
                    )

        artifact_store.put_text(
            state["thread_id"],
            "research_report_draft",
            report,
            content_type="text/markdown",
        )
        if validation_errors:
            artifact_store.put_text(
                state["thread_id"],
                "research_report_incomplete",
                report,
                content_type="text/markdown",
            )
            snapshot = await budget_manager.snapshot(state["thread_id"])
            error = "最终报告完整性检查失败：" + "；".join(validation_errors)
            emit_progress(
                state["thread_id"],
                "report_failed",
                "report",
                "最终报告未通过完整性检查，已保留残稿供排查",
                detail="；".join(validation_errors),
            )
            result = {
                "notes": {"type": "override", "value": []},
                "research_outline": {},
                "status": "failed",
                "error": error,
                "source_count": 0,
                "tools_used": list(dict.fromkeys(state.get("tools_used", []))),
                "hy3_calls_used": snapshot.used,
                "hy3_call_budget": snapshot.limit,
                "hy3_call_budget_exhausted": snapshot.exhausted or not acquired,
                "hy3_calls_by_stage": snapshot.calls_by_stage,
            }
            if completeness_check is not None:
                result["report_completeness_check"] = completeness_check
            await budget_manager.release(state["thread_id"])
            return result

        artifact_store.put_text(
            state["thread_id"],
            "research_report",
            report,
            content_type="text/markdown",
        )
        # Report metadata should describe sources actually delivered to the
        # user, not every URL that happened to appear in an internal memo.
        urls = set(re.findall(r"https?://[^\s)>\]]+", report))
        snapshot = await budget_manager.snapshot(state["thread_id"])
        result = {
            "final_report": report,
            "notes": {"type": "override", "value": []},
            "research_outline": {},
            "status": "complete",
            "source_count": len(urls),
            "tools_used": list(dict.fromkeys(state.get("tools_used", []))),
            "hy3_calls_used": snapshot.used,
            "hy3_call_budget": snapshot.limit,
            "hy3_call_budget_exhausted": snapshot.exhausted or not acquired,
            "hy3_calls_by_stage": snapshot.calls_by_stage,
        }
        if completeness_check is not None:
            result["report_completeness_check"] = completeness_check
        emit_progress(
            state["thread_id"],
            "report_completed",
            "report",
            "最终报告已经生成",
            detail=f"已整理 {len(urls)} 个可追溯来源",
            payload={"source_count": len(urls)},
        )
        await budget_manager.release(state["thread_id"])
        return result

    main = StateGraph(
        DeepResearchState,
        input_schema=DeepResearchInputState,
    )
    main.add_node("clarify_with_user", clarify_with_user)
    main.add_node("write_research_outline", write_research_outline)
    main.add_node("confirm_outline_with_user", confirm_outline_with_user)
    main.add_node("research_manager", manager_graph)
    main.add_node("report_generation", report_generation)
    main.add_edge(START, "clarify_with_user")
    main.add_conditional_edges(
        "clarify_with_user",
        route_after_clarification,
        {"write_research_outline": "write_research_outline", "__end__": END},
    )
    main.add_edge("write_research_outline", "confirm_outline_with_user")
    main.add_conditional_edges(
        "confirm_outline_with_user",
        route_after_outline_confirmation,
        {
            "write_research_outline": "write_research_outline",
            "research_manager": "research_manager",
            "__end__": END,
        },
    )
    main.add_edge("research_manager", "report_generation")
    main.add_edge("report_generation", END)
    return main.compile(checkpointer=checkpointer)
