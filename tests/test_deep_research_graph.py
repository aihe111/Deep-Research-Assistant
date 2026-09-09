import asyncio
import json
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from pydantic import BaseModel

from deep_research_assistant.artifact_store import ArtifactStore
from deep_research_assistant.config import Settings
from deep_research_assistant.conversation_store import ConversationStore
from deep_research_assistant.deep_research_graph import (
    _bounded_notes,
    _compact_tool_output,
    _deterministic_research_memo,
    _finalize_research_memo,
    _is_empty_ai_response,
    _limit_research_memo,
    _normalize_numbered_sources,
    _parse_outline_resume,
    _repair_numbered_citations,
    _validate_numbered_report,
    build_deep_research_graph,
)
from deep_research_assistant.deep_research_prompts import (
    CITATION_REPAIR_PROMPT,
    COMPRESSION_PROMPT,
    FINAL_REPORT_PROMPT,
    MANAGER_FINAL_REVIEW_PROMPT,
    REPORT_COMPLETENESS_CHECK_PROMPT,
    RESEARCHER_PROMPT,
    SUPERVISOR_PROMPT,
)
from deep_research_assistant.deep_research_state import (
    ClarificationDecision,
    DeepResearchState,
    ManagerState,
    ReportCompletenessCheck,
    ResearchComplete,
    ResearcherState,
    ResearchQuestion,
)


class SearchInput(BaseModel):
    query: str


def test_outline_resume_accepts_studio_json_strings() -> None:
    expected = {"action": "approve", "feedback": ""}

    assert _parse_outline_resume(expected) == expected
    assert _parse_outline_resume('{"action":"approve","feedback":""}') == expected
    assert _parse_outline_resume('"approve"') == expected
    assert _parse_outline_resume("approve") == expected


def test_redundant_deep_research_state_contracts_are_removed() -> None:
    assert "followup_question" not in DeepResearchState.__annotations__
    assert "outline_feedback" not in DeepResearchState.__annotations__
    assert "manager_messages" not in DeepResearchState.__annotations__
    assert "raw_notes" not in DeepResearchState.__annotations__
    assert "raw_notes" not in ManagerState.__annotations__
    assert "context_notes" not in ResearcherState.__annotations__
    assert "verification" not in ClarificationDecision.model_fields
    assert not ResearchComplete.model_fields


def test_empty_compression_uses_non_empty_deterministic_memo() -> None:
    memo, used_fallback = _finalize_research_memo(
        "",
        "Compare PostgreSQL and MySQL",
        ["Evidence https://example.com/database-comparison"],
        5_000,
    )

    assert used_fallback is True
    assert "Compare PostgreSQL and MySQL" in memo
    assert "https://example.com/database-comparison" in memo


def test_non_empty_compression_keeps_model_memo() -> None:
    memo, used_fallback = _finalize_research_memo(
        "Grounded memo [source](https://example.com/source).",
        "Research topic",
        ["Unused fallback evidence"],
        5_000,
    )

    assert used_fallback is False
    assert memo == "Grounded memo [source](https://example.com/source)."


def test_empty_ai_response_requires_both_text_and_tool_calls_to_be_missing() -> None:
    assert _is_empty_ai_response(AIMessage(content="")) is True
    assert _is_empty_ai_response(AIMessage(content="Finished")) is False
    assert (
        _is_empty_ai_response(
            AIMessage(
                content="",
                tool_calls=[{"name": "ResearchComplete", "args": {}, "id": "done"}],
            )
        )
        is False
    )


def test_research_prompts_require_comprehensive_findings_and_global_citations() -> None:
    assert "完整研究发现" in COMPRESSION_PROMPT
    assert "尽量保留所有" in COMPRESSION_PROMPT
    assert "本地编号" in COMPRESSION_PROMPT
    assert "允许使用模型已有知识补充" in COMPRESSION_PROMPT
    assert "不需要额外标注" in COMPRESSION_PROMPT
    assert "https://doi.org/DOI" in COMPRESSION_PROMPT
    assert "全局连续编号" in FINAL_REPORT_PROMPT
    assert "不要直接沿用" in FINAL_REPORT_PROMPT
    assert "- [编号] 来源标题: URL" in FINAL_REPORT_PROMPT
    assert "不得设置固定来源数量" in FINAL_REPORT_PROMPT
    assert "不要把报告压缩成简短摘要" in FINAL_REPORT_PROMPT
    assert "章节长度应与问题重要性" in FINAL_REPORT_PROMPT
    assert "用户明确指定的篇幅要求优先" in FINAL_REPORT_PROMPT
    assert "只有用户未指定篇幅时" in FINAL_REPORT_PROMPT
    assert "不必把报告严格限制为工具返回内容的复述" in FINAL_REPORT_PROMPT
    assert "不需要额外添加“未核验”" in FINAL_REPORT_PROMPT
    assert "可以直接使用模型知识补全分析和建议" in FINAL_REPORT_PROMPT
    assert "研究员反思转述" in FINAL_REPORT_PROMPT
    assert "不得从头重写报告" in CITATION_REPAIR_PROMPT
    assert "最小范围修复" in CITATION_REPAIR_PROMPT
    assert "禁止提出或委派新的研究任务" in MANAGER_FINAL_REVIEW_PROMPT
    assert "全部证据备忘录" in MANAGER_FINAL_REVIEW_PROMPT


def test_joined_sources_are_sorted_and_put_on_separate_lines() -> None:
    report = (
        "# Report\n\nClaim [1][2].\n\n## 主要来源\n"
        "[2] Second: https://example.com/2 [1] First: https://example.com/1"
    )

    normalized = _normalize_numbered_sources(report)

    assert normalized.endswith(
        "## 主要来源\n\n"
        "- [1] First: https://example.com/1\n"
        "- [2] Second: https://example.com/2"
    )
    assert _validate_numbered_report(normalized) == []


def test_report_validation_detects_missing_and_incomplete_sources() -> None:
    report = (
        "# Report\n\nClaims [1][2].\n\n## 主要来源\n\n"
        "- [1] Complete: https://example.com/1\n"
        "- [2] Incomplete source title"
    )

    errors = _validate_numbered_report(report)

    assert "来源 [2] 缺少完整 URL 或 MCP 记录标识" in errors

    missing = "# Report\n\nClaims [1][44].\n\n## 主要来源\n- [1] One: https://example.com/1"
    errors = _validate_numbered_report(missing)
    assert any("[44]" in error for error in errors)


def test_citation_repair_canonicalizes_doi_drops_unused_and_renumbers() -> None:
    report = (
        "# Report\n\nSecond claim [8]. First claim [3].\n\n## 主要来源\n"
        "- [3] First paper (DOI:10.1111/ina.12785)\n"
        "- [7] Unused: https://example.com/unused\n"
        "- [8] Second paper: 10.1016/j.scitotenv.2022.158026"
    )

    repaired = _repair_numbered_citations(report)

    assert "Second claim [1]. First claim [2]." in repaired
    assert "https://doi.org/10.1016/j.scitotenv.2022.158026" in repaired
    assert "https://doi.org/10.1111/ina.12785" in repaired
    assert "Unused" not in repaired
    assert _validate_numbered_report(repaired) == []


def test_citation_repair_keeps_unaddressed_reflection_invalid_for_local_repair() -> None:
    report = (
        "# Report\n\nUnsupported claim [1].\n\n## 主要来源\n"
        "- [1] WHO guidance – 研究员反思转述"
    )

    repaired = _repair_numbered_citations(report)

    assert "研究员反思转述" in repaired
    assert _validate_numbered_report(repaired) == [
        "来源 [1] 缺少完整 URL 或 MCP 记录标识"
    ]


def test_search_stopping_rules_are_bounded() -> None:
    researcher_prompt = RESEARCHER_PROMPT.format(
        research_brief="brief",
        research_topic="topic",
        max_search_tool_calls=5,
        mcp_prompt="",
    )

    assert "为每个比较对象分别启动一个 Researcher" in SUPERVISOR_PROMPT
    assert "非比较类问题仍优先使用一个 Researcher" in SUPERVISOR_PROMPT
    assert "2～3 次" in researcher_prompt
    assert "最多使用 5 次" in researcher_prompt
    assert "3 个以上" in researcher_prompt
    assert "最近两次搜索返回高度相似" in researcher_prompt


class FakeModel:
    def __init__(self, stage: str, shared: dict[str, Any]) -> None:
        self.stage = stage
        self.shared = shared
        self.schema = None
        self.bound_tool_names: set[str] = set()

    def with_structured_output(self, schema, **kwargs):
        self.schema = schema
        return self

    def bind_tools(self, tools, **kwargs):
        self.bound_tool_names = {
            str(getattr(item, "name", None) or getattr(item, "__name__", ""))
            for item in tools
        }
        return self

    async def ainvoke(self, messages, config=None):
        if self.schema is ReportCompletenessCheck:
            self.shared.setdefault("completeness_prompts", []).append(
                [str(message.content) for message in messages]
            )
            if self.shared.get("incomplete_report_check"):
                return ReportCompletenessCheck(
                    score=1,
                    is_complete=False,
                    truncation_detected=True,
                    missing_sections=["Cross-chain bridge audit defenses"],
                    unanswered_questions=["Which defense patterns are current?"],
                    structural_defects=["A dangling section heading has no body"],
                    reason="The report stops after a dangling heading.",
                )
            return ReportCompletenessCheck(
                score=4,
                is_complete=True,
                truncation_detected=False,
                missing_sections=[],
                unanswered_questions=[],
                structural_defects=[],
                reason="All requested sections are present and complete.",
            )
        if self.schema is ResearchQuestion:
            return ResearchQuestion(
                research_brief="Compare two independent evidence dimensions.",
                research_outline={
                    "title": "Comparison report",
                    "thesis": "Compare two independent evidence dimensions.",
                    "sections": [
                        {
                            "section_id": f"S{index}",
                            "title": f"Section {index}",
                            "objective": f"Evaluate dimension {index}",
                            "research_questions": [f"What does dimension {index} show?"],
                        }
                        for index in range(1, 4)
                    ],
                },
            )
        if self.stage == "supervisor":
            if not self.bound_tool_names and self.shared.get("capture_final_review"):
                self.shared["final_review_messages"] = messages
                return AIMessage(content="Final evidence audit completed")
            if self.shared.get("empty_supervisor_remaining", 0) > 0:
                self.shared["empty_supervisor_remaining"] -= 1
                return AIMessage(content="")
            self.shared["supervisor_calls"] += 1
            if self.shared.get("delegate_twice") and self.shared["supervisor_calls"] <= 2:
                index = self.shared["supervisor_calls"]
                return AIMessage(
                    content=f"Delegating round {index}",
                    tool_calls=[
                        {
                            "name": "ConductResearch",
                            "args": {"research_topic": f"Round {index} evidence"},
                            "id": f"round-{index}",
                        }
                    ],
                )
            if self.shared["supervisor_calls"] == 1:
                return AIMessage(
                    content="Delegating in parallel",
                    tool_calls=[
                        {
                            "name": "ConductResearch",
                            "args": {"research_topic": "Dimension A with primary evidence"},
                            "id": "unit-a",
                        },
                        {
                            "name": "ConductResearch",
                            "args": {"research_topic": "Dimension B with primary evidence"},
                            "id": "unit-b",
                        },
                    ],
                )
            return AIMessage(
                content="Evidence is sufficient",
                tool_calls=[{"name": "ResearchComplete", "args": {}, "id": "done"}],
            )
        if self.stage == "researcher":
            if self.shared.get("empty_researcher_remaining", 0) > 0:
                self.shared["empty_researcher_remaining"] -= 1
                return AIMessage(content="")
            if self.bound_tool_names == {"think_tool"}:
                return AIMessage(
                    content="Reflecting",
                    tool_calls=[
                        {
                            "name": "think_tool",
                            "args": {
                                "reflection": (
                                    "Evidence reviewed; only material gaps justify "
                                    "another search."
                                )
                            },
                            "id": "reflection",
                        }
                    ],
                )
            if self.bound_tool_names == {"ResearchComplete"}:
                return AIMessage(
                    content="Search cap reached",
                    tool_calls=[
                        {"name": "ResearchComplete", "args": {}, "id": "r-limit"}
                    ],
                )
            if self.shared.get("always_search_researcher"):
                search_index = self.shared.get("planned_search_calls", 0) + 1
                self.shared["planned_search_calls"] = search_index
                return AIMessage(
                    content="Searching another evidence gap",
                    tool_calls=[
                        {
                            "name": "mock_search",
                            "args": {"query": f"gap {search_index}"},
                            "id": f"search-gap-{search_index}",
                        }
                    ],
                )
            if isinstance(messages[-1], HumanMessage):
                query = messages[-1].content
                return AIMessage(
                    content="Searching",
                    tool_calls=[
                        {
                            "name": "mock_search",
                            "args": {"query": query},
                            "id": f"search-{query[-1]}",
                        }
                    ],
                )
            assert isinstance(messages[-1], ToolMessage)
            return AIMessage(
                content="Enough evidence",
                tool_calls=[{"name": "ResearchComplete", "args": {}, "id": "r-done"}],
            )
        if self.stage == "compression":
            return AIMessage(
                content=(
                    "## Complete findings\nEvidence memo [1].\n\n"
                    "## Sources\n[1] Evidence: https://example.com/evidence"
                )
            )
        if self.stage == "final":
            self.shared.setdefault("final_prompts", []).append(
                str(messages[-1].content)
            )
            if self.shared.get("empty_final_remaining", 0) > 0:
                self.shared["empty_final_remaining"] -= 1
                return AIMessage(content="")
            if self.shared.get("invalid_final_remaining", 0) > 0:
                self.shared["invalid_final_remaining"] -= 1
                return AIMessage(
                    content=(
                        "# Report\n\nGrounded result [1][2].\n\n## Sources\n"
                        "[1] Evidence: https://example.com/evidence [2] Truncated"
                    )
                )
            if self.shared.get("truncated_but_citation_valid"):
                return AIMessage(
                    content=(
                        "# Report\n\nGrounded result [1].\n\n"
                        "## Dangling\n\n## Sources\n"
                        "[1] Evidence: https://example.com/evidence"
                    )
                )
            return AIMessage(
                content=(
                    "# Report\n\nGrounded result [1].\n\n## Sources\n"
                    "[1] Evidence: https://example.com/evidence"
                )
            )
        raise AssertionError(f"Unexpected stage: {self.stage}")


def test_parallel_researchers_complete_and_persist_report(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        allow_clarification=False,
        require_outline_confirmation=False,
        database_url=f"sqlite:///{tmp_path / 'deep.db'}",
        research_brief_max_tokens=1001,
        supervisor_max_tokens=1002,
        researcher_max_tokens=1003,
        compression_max_tokens=1004,
        final_report_max_tokens=1005,
    )
    shared = {"supervisor_calls": 0, "active": 0, "max_active": 0}

    async def mock_search(query: str) -> str:
        shared["active"] += 1
        shared["max_active"] = max(shared["max_active"], shared["active"])
        await asyncio.sleep(0.02)
        shared["active"] -= 1
        return f"Result for {query}: https://example.com/evidence"

    search_tool = StructuredTool.from_function(
        coroutine=mock_search,
        name="mock_search",
        description="Return test evidence.",
        args_schema=SearchInput,
    )

    stages = {
        1001: "brief",
        1002: "supervisor",
        1003: "researcher",
        1004: "compression",
        1005: "final",
    }
    store = ArtifactStore(settings.database_url)
    event_store = ConversationStore(settings.database_url)
    event_store.create_conversation("parallel-test")
    graph = build_deep_research_graph(
        settings,
        store,
        event_store=event_store,
        model_factory=lambda tokens: FakeModel(stages[tokens], shared),
        tool_loader=lambda: asyncio.sleep(0, result=[search_tool]),
    )
    assert set(graph.get_input_jsonschema()["properties"]) == {
        "thread_id",
        "mode",
        "original_request",
        "messages",
        "clarification_completed",
    }
    try:
        result = asyncio.run(
            graph.ainvoke(
                {
                    "thread_id": "parallel-test",
                    "mode": "research",
                    "messages": [{"role": "user", "content": "Research both dimensions"}],
                }
            )
        )
        assert result["status"] == "complete"
        assert result["research_unit_count"] == 2
        assert result["source_count"] == 1
        assert result["tools_used"] == ["mock_search", "think_tool"]
        assert shared["max_active"] == 2
        assert result["hy3_calls_used"] == 14
        assert result["hy3_calls_by_stage"]["report_completeness_check"] == 1
        assert result["report_completeness_check"]["score"] == 4
        assert result["hy3_calls_by_stage"]["research_manager_review"] == 1
        assert result["hy3_calls_by_stage"]["research_manager_final_review"] == 1
        assert result["hy3_call_budget_exhausted"] is False
        assert result["notes"] == []
        assert result["research_outline"] == {}
        assert "raw_notes" not in result
        assert "manager_messages" not in result
        assert store.get_text("parallel-test", "research_report").startswith("# Report")
        event_types = {
            event.event_type for event in event_store.list_events("parallel-test")
        }
        assert {
            "clarification_completed",
            "outline_completed",
            "researcher_started",
            "tool_completed",
            "compression_completed",
            "manager_completed",
            "manager_final_review_completed",
            "report_completed",
        } <= event_types
    finally:
        event_store.close()
        store.close()


def test_last_delegation_round_is_always_followed_by_final_manager_review(
    tmp_path: Path,
) -> None:
    settings = Settings(
        _env_file=None,
        allow_clarification=False,
        require_outline_confirmation=False,
        max_concurrent_research_units=1,
        max_supervisor_iterations=2,
        database_url=f"sqlite:///{tmp_path / 'final-review.db'}",
        research_brief_max_tokens=1001,
        supervisor_max_tokens=1002,
        researcher_max_tokens=1003,
        compression_max_tokens=1004,
        final_report_max_tokens=1005,
    )
    shared = {
        "supervisor_calls": 0,
        "active": 0,
        "max_active": 0,
        "delegate_twice": True,
        "capture_final_review": True,
    }

    async def mock_search(query: str) -> str:
        return f"Evidence for {query}: https://example.com/final-review"

    search_tool = StructuredTool.from_function(
        coroutine=mock_search,
        name="mock_search",
        description="Return test evidence.",
        args_schema=SearchInput,
    )
    stages = {
        1001: "brief",
        1002: "supervisor",
        1003: "researcher",
        1004: "compression",
        1005: "final",
    }
    store = ArtifactStore(settings.database_url)
    graph = build_deep_research_graph(
        settings,
        store,
        model_factory=lambda tokens: FakeModel(stages[tokens], shared),
        tool_loader=lambda: asyncio.sleep(0, result=[search_tool]),
    )
    try:
        result = asyncio.run(
            graph.ainvoke(
                {
                    "thread_id": "final-review-test",
                    "mode": "research",
                    "messages": [{"role": "user", "content": "Research two rounds"}],
                }
            )
        )

        assert result["status"] == "complete"
        assert result["research_unit_count"] == 2
        assert shared["supervisor_calls"] == 2
        final_review_prompt = shared["final_review_messages"][0].content
        assert final_review_prompt.count("## Complete findings") == 2
        assert result["hy3_calls_by_stage"]["research_manager_final_review"] == 1
    finally:
        store.close()


def test_researcher_hard_caps_searches_and_reflects_after_each_batch(
    tmp_path: Path,
) -> None:
    settings = Settings(
        _env_file=None,
        allow_clarification=False,
        require_outline_confirmation=False,
        max_concurrent_research_units=1,
        max_search_tool_calls=2,
        database_url=f"sqlite:///{tmp_path / 'search-cap.db'}",
        research_brief_max_tokens=1001,
        supervisor_max_tokens=1002,
        researcher_max_tokens=1003,
        compression_max_tokens=1004,
        final_report_max_tokens=1005,
    )
    shared = {
        "supervisor_calls": 0,
        "active": 0,
        "max_active": 0,
        "search_calls": 0,
        "always_search_researcher": True,
    }

    async def mock_search(query: str) -> str:
        shared["search_calls"] += 1
        return f"Evidence for {query}: https://example.com/search-cap"

    search_tool = StructuredTool.from_function(
        coroutine=mock_search,
        name="mock_search",
        description="Return test evidence.",
        args_schema=SearchInput,
    )
    stages = {
        1001: "brief",
        1002: "supervisor",
        1003: "researcher",
        1004: "compression",
        1005: "final",
    }
    store = ArtifactStore(settings.database_url)
    graph = build_deep_research_graph(
        settings,
        store,
        model_factory=lambda tokens: FakeModel(stages[tokens], shared),
        tool_loader=lambda: asyncio.sleep(0, result=[search_tool]),
    )
    try:
        result = asyncio.run(
            graph.ainvoke(
                {
                    "thread_id": "search-cap-test",
                    "mode": "research",
                    "messages": [{"role": "user", "content": "Research one dimension"}],
                }
            )
        )

        assert result["status"] == "complete"
        assert shared["search_calls"] == settings.max_search_tool_calls
        assert result["tools_used"] == ["mock_search", "think_tool"]
        assert result["hy3_calls_by_stage"]["researcher"] == 4
        assert result["hy3_calls_by_stage"]["research_manager_review"] == 1
    finally:
        store.close()


def test_empty_manager_and_researcher_responses_retry_and_finish(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        allow_clarification=False,
        require_outline_confirmation=False,
        database_url=f"sqlite:///{tmp_path / 'empty-retry.db'}",
        research_brief_max_tokens=1001,
        supervisor_max_tokens=1002,
        researcher_max_tokens=1003,
        compression_max_tokens=1004,
        final_report_max_tokens=1005,
    )
    shared = {
        "supervisor_calls": 0,
        "active": 0,
        "max_active": 0,
        "empty_supervisor_remaining": 1,
        "empty_researcher_remaining": 1,
        "empty_final_remaining": 1,
    }

    async def mock_search(query: str) -> str:
        return f"Evidence for {query}: https://example.com/empty-retry"

    search_tool = StructuredTool.from_function(
        coroutine=mock_search,
        name="mock_search",
        description="Return test evidence.",
        args_schema=SearchInput,
    )
    stages = {
        1001: "brief",
        1002: "supervisor",
        1003: "researcher",
        1004: "compression",
        1005: "final",
    }
    store = ArtifactStore(settings.database_url)
    event_store = ConversationStore(settings.database_url)
    event_store.create_conversation("empty-retry-test")
    graph = build_deep_research_graph(
        settings,
        store,
        event_store=event_store,
        model_factory=lambda tokens: FakeModel(stages[tokens], shared),
        tool_loader=lambda: asyncio.sleep(0, result=[search_tool]),
    )
    try:
        result = asyncio.run(
            graph.ainvoke(
                {
                    "thread_id": "empty-retry-test",
                    "mode": "research",
                    "messages": [
                        {"role": "user", "content": "Research both dimensions"}
                    ],
                }
            )
        )

        assert result["status"] == "complete"
        assert result["research_unit_count"] == 2
        assert result["hy3_calls_by_stage"]["research_manager_empty_retry"] == 1
        assert result["hy3_calls_by_stage"]["researcher_empty_retry"] == 1
        assert result["hy3_calls_by_stage"]["final_report_empty_retry"] == 1
        assert store.get_text("empty-retry-test", "research_report").startswith(
            "# Report"
        )
        retry_events = [
            event
            for event in event_store.list_events("empty-retry-test")
            if event.event_type == "model_empty_retry"
        ]
        assert {event.stage for event in retry_events} == {
            "management",
            "research",
            "report",
        }
    finally:
        event_store.close()
        store.close()


def test_incomplete_final_report_is_rewritten_and_normalized(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        allow_clarification=False,
        require_outline_confirmation=False,
        database_url=f"sqlite:///{tmp_path / 'report-validation.db'}",
        research_brief_max_tokens=1001,
        supervisor_max_tokens=1002,
        researcher_max_tokens=1003,
        compression_max_tokens=1004,
        final_report_max_tokens=1005,
    )
    shared = {
        "supervisor_calls": 0,
        "active": 0,
        "max_active": 0,
        "invalid_final_remaining": 1,
    }

    async def mock_search(query: str) -> str:
        return f"Evidence for {query}: https://example.com/evidence"

    search_tool = StructuredTool.from_function(
        coroutine=mock_search,
        name="mock_search",
        description="Return test evidence.",
        args_schema=SearchInput,
    )
    stages = {
        1001: "brief",
        1002: "supervisor",
        1003: "researcher",
        1004: "compression",
        1005: "final",
    }
    store = ArtifactStore(settings.database_url)
    event_store = ConversationStore(settings.database_url)
    event_store.create_conversation("report-validation-test")
    graph = build_deep_research_graph(
        settings,
        store,
        event_store=event_store,
        model_factory=lambda tokens: FakeModel(stages[tokens], shared),
        tool_loader=lambda: asyncio.sleep(0, result=[search_tool]),
    )
    try:
        result = asyncio.run(
            graph.ainvoke(
                {
                    "thread_id": "report-validation-test",
                    "mode": "research",
                    "messages": [{"role": "user", "content": "Research both dimensions"}],
                }
            )
        )

        assert result["status"] == "complete"
        assert result["hy3_calls_by_stage"]["final_report_validation_retry"] == 1
        assert "不得从头重写报告" in shared["final_prompts"][1]
        assert "最小范围修复" in shared["final_prompts"][1]
        report = store.get_text("report-validation-test", "research_report")
        assert report.endswith(
            "## Sources\n\n- [1] Evidence: https://example.com/evidence"
        )
        assert _validate_numbered_report(report) == []
        assert "report_validation_retry" in {
            event.event_type
            for event in event_store.list_events("report-validation-test")
        }
    finally:
        event_store.close()
        store.close()


def test_llm_completeness_check_rejects_structurally_valid_truncation(
    tmp_path: Path,
) -> None:
    settings = Settings(
        _env_file=None,
        allow_clarification=False,
        require_outline_confirmation=False,
        database_url=f"sqlite:///{tmp_path / 'llm-completeness.db'}",
        research_brief_max_tokens=1001,
        supervisor_max_tokens=1002,
        researcher_max_tokens=1003,
        compression_max_tokens=1004,
        final_report_max_tokens=1005,
    )
    shared = {
        "supervisor_calls": 0,
        "active": 0,
        "max_active": 0,
        "truncated_but_citation_valid": True,
        "incomplete_report_check": True,
    }

    async def mock_search(query: str) -> str:
        return f"Evidence for {query}: https://example.com/evidence"

    search_tool = StructuredTool.from_function(
        coroutine=mock_search,
        name="mock_search",
        description="Return test evidence.",
        args_schema=SearchInput,
    )
    stages = {
        1001: "brief",
        1002: "supervisor",
        1003: "researcher",
        1004: "compression",
        1005: "final",
    }
    store = ArtifactStore(settings.database_url)
    graph = build_deep_research_graph(
        settings,
        store,
        model_factory=lambda tokens: FakeModel(stages[tokens], shared),
        tool_loader=lambda: asyncio.sleep(0, result=[search_tool]),
    )
    try:
        result = asyncio.run(
            graph.ainvoke(
                {
                    "thread_id": "llm-completeness-test",
                    "mode": "research",
                    "messages": [{"role": "user", "content": "Research both dimensions"}],
                }
            )
        )

        assert result["status"] == "failed"
        assert "LLM 完整性检查未通过" in result["error"]
        assert result["report_completeness_check"]["score"] == 1
        assert result["hy3_calls_by_stage"]["report_completeness_check"] == 1
        assert REPORT_COMPLETENESS_CHECK_PROMPT in shared["completeness_prompts"][0][0]
        assert store.get_text(
            "llm-completeness-test", "research_report_incomplete"
        ).startswith("# Report")
    finally:
        store.close()


def test_global_hy3_budget_stops_research_and_preserves_final_report(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        allow_clarification=False,
        require_outline_confirmation=False,
        database_url=f"sqlite:///{tmp_path / 'budget.db'}",
        max_hy3_calls_per_research=7,
        research_brief_max_tokens=1001,
        supervisor_max_tokens=1002,
        researcher_max_tokens=1003,
        compression_max_tokens=1004,
        final_report_max_tokens=1005,
    )
    shared = {"supervisor_calls": 0, "active": 0, "max_active": 0}

    async def mock_search(query: str) -> str:
        return f"Evidence for {query}: https://example.com/budget"

    search_tool = StructuredTool.from_function(
        coroutine=mock_search,
        name="mock_search",
        description="Return test evidence.",
        args_schema=SearchInput,
    )
    stages = {
        1001: "brief",
        1002: "supervisor",
        1003: "researcher",
        1004: "compression",
        1005: "final",
    }
    store = ArtifactStore(settings.database_url)
    graph = build_deep_research_graph(
        settings,
        store,
        model_factory=lambda tokens: FakeModel(stages[tokens], shared),
        tool_loader=lambda: asyncio.sleep(0, result=[search_tool]),
    )
    try:
        result = asyncio.run(
            graph.ainvoke(
                {
                    "thread_id": "budget-test",
                    "mode": "research",
                    "messages": [{"role": "user", "content": "Research both dimensions"}],
                }
            )
        )
        assert result["status"] == "complete"
        assert result["hy3_calls_used"] <= settings.max_hy3_calls_per_research
        assert result["hy3_call_budget_exhausted"] is True
        assert result["hy3_calls_by_stage"]["research_compression"] == 1
        assert result["hy3_calls_by_stage"]["research_manager_final_review"] == 1
        assert result["research_unit_count"] == 1
        assert "final_report" in result["hy3_calls_by_stage"]
        assert store.get_text("budget-test", "research_report").startswith("# Report")
    finally:
        store.close()


def test_tool_results_and_research_context_are_deterministically_bounded() -> None:
    payload = {
        "provider": "test",
        "results": [
            {
                "title": f"Source {index}",
                "url": f"https://example.com/{index}",
                "content": "x" * 5000,
            }
            for index in range(4)
        ],
    }
    compacted = _compact_tool_output(payload, 4000)
    context = _bounded_notes([compacted, "y" * 8000], 5000)

    assert len(compacted) <= 4000
    assert len(context) <= 5000
    compacted_payload = json.loads(compacted)
    assert [item["url"] for item in compacted_payload["results"]] == [
        f"https://example.com/{index}" for index in range(4)
    ]


def test_deterministic_research_memo_deduplicates_and_never_returns_raw_context() -> None:
    repeated = (
        "关键发现：样本量为 120，效果提升 18%。"
        "来源 https://example.com/study。"
    )
    memo = _deterministic_research_memo(
        "比较干预效果",
        [repeated, repeated, "无关背景 " + "x" * 20_000],
        3_000,
    )

    assert len(memo) <= 3_000
    assert memo.count("https://example.com/study") <= 2  # evidence plus source register
    assert memo.count("关键发现") == 1
    assert "确定性压缩证据" in memo
    assert "x" * 5_000 not in memo


def test_generated_research_memo_has_a_hard_delivery_limit() -> None:
    memo = "主要证据" + "x" * 8_000 + "\n\n### 来源\nhttps://example.com/source"

    bounded = _limit_research_memo(memo, 5_000)

    assert len(bounded) == 5_000
    assert "证据备忘录超过长度上限" in bounded
    assert bounded.endswith("https://example.com/source")


def test_outline_confirmation_can_revise_then_approve(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        allow_clarification=False,
        require_outline_confirmation=True,
        max_outline_revisions=3,
        database_url=f"sqlite:///{tmp_path / 'outline.db'}",
        research_brief_max_tokens=1001,
        supervisor_max_tokens=1002,
        researcher_max_tokens=1003,
        compression_max_tokens=1004,
        final_report_max_tokens=1005,
    )
    shared = {"supervisor_calls": 0, "active": 0, "max_active": 0}

    async def mock_search(query: str) -> str:
        return f"Evidence for {query}: https://example.com/outline"

    search_tool = StructuredTool.from_function(
        coroutine=mock_search,
        name="mock_search",
        description="Return test evidence.",
        args_schema=SearchInput,
    )
    stages = {
        1001: "brief",
        1002: "supervisor",
        1003: "researcher",
        1004: "compression",
        1005: "final",
    }
    store = ArtifactStore(settings.database_url)
    graph = build_deep_research_graph(
        settings,
        store,
        model_factory=lambda tokens: FakeModel(stages[tokens], shared),
        tool_loader=lambda: asyncio.sleep(0, result=[search_tool]),
        checkpointer=MemorySaver(),
    )
    config = {"configurable": {"thread_id": "outline-test"}}
    initial = {
        "thread_id": "outline-test",
        "mode": "research",
        "messages": [{"role": "user", "content": "Research both dimensions"}],
    }
    try:
        first = asyncio.run(graph.ainvoke(initial, config=config))
        assert first["__interrupt__"][0].value["kind"] == "outline_confirmation"
        assert first["__interrupt__"][0].value["revision_count"] == 0

        revised = asyncio.run(
            graph.ainvoke(
                Command(resume={"action": "revise", "feedback": "加强反方证据"}),
                config=config,
            )
        )
        assert revised["__interrupt__"][0].value["revision_count"] == 1

        completed = asyncio.run(
            graph.ainvoke(Command(resume={"action": "approve"}), config=config)
        )
        assert completed["status"] == "complete"
        assert completed["outline_confirmed"] is True
        assert completed["outline_revision_count"] == 1
        assert "用户确认的报告大纲" in completed["research_brief"]
    finally:
        store.close()
