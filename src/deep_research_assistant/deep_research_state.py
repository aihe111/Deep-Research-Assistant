"""State and tool-call contracts for the general deep-research graph."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from deep_research_assistant.models import ResearchOutline


def override_reducer(current_value: list[Any], new_value: Any) -> list[Any]:
    """Append normally, while allowing a node to explicitly clear consumed state."""

    if isinstance(new_value, dict) and new_value.get("type") == "override":
        return list(new_value.get("value") or [])
    return operator.add(list(current_value or []), list(new_value or []))


class ClarificationDecision(BaseModel):
    """判断是否存在必须由用户补充的关键歧义。"""

    needs_clarification: bool
    question: str = Field(default="")


class ResearchQuestion(BaseModel):
    """无需读取原对话即可独立执行的研究简报与待确认大纲。"""

    research_brief: str = Field(min_length=10)
    research_outline: ResearchOutline


class ConductResearch(BaseModel):
    """把一个边界清楚的独立研究单元委派给并行研究员。"""

    research_topic: str = Field(
        min_length=5,
        description="完整、自包含的研究任务，并说明需要取得什么证据。",
    )


class ResearchComplete(BaseModel):
    """表示当前证据已经足够进入综合写作。"""

    pass


class ReportCompletenessCheck(BaseModel):
    """最终报告生成后的结构化完整性判定。"""

    score: int = Field(ge=0, le=4)
    is_complete: bool
    truncation_detected: bool
    missing_sections: list[str] = Field(default_factory=list)
    unanswered_questions: list[str] = Field(default_factory=list)
    structural_defects: list[str] = Field(default_factory=list)
    reason: str = Field(min_length=1)


class DeepResearchInputState(TypedDict, total=False):
    """Public graph input; internal checkpoints must not leak into Studio inputs."""

    thread_id: str
    mode: Literal["research", "followup"]
    original_request: str
    messages: list[AnyMessage]
    clarification_completed: bool


class DeepResearchState(TypedDict, total=False):
    """Top-level state kept compact enough for durable LangGraph checkpoints."""

    thread_id: str
    mode: Literal["research", "followup"]
    original_request: str
    followup_answer: str
    followup_metadata: dict[str, Any]
    messages: Annotated[list[AnyMessage], add_messages]
    clarification_completed: bool
    research_brief: str
    research_outline: dict[str, Any]
    outline_confirmed: bool
    outline_revision_count: int
    notes: Annotated[list[str], override_reducer]
    final_report: str
    report_completeness_check: dict[str, Any]
    status: str
    research_unit_count: int
    source_count: int
    tools_used: list[str]
    hy3_calls_used: int
    hy3_call_budget: int
    hy3_call_budget_exhausted: bool
    hy3_calls_by_stage: dict[str, int]
    error: str


class ManagerState(TypedDict, total=False):
    """State local to the manager subgraph."""

    thread_id: str
    research_brief: str
    manager_messages: Annotated[list[AnyMessage], add_messages]
    notes: Annotated[list[str], operator.add]
    manager_iteration: int
    manager_review_reserved: bool
    manager_final_review_reserved: bool
    research_unit_count: int
    tools_used: Annotated[list[str], operator.add]
    hy3_call_budget_exhausted: bool


class ResearcherState(TypedDict, total=False):
    """State local to one independent researcher subgraph."""

    thread_id: str
    research_topic: str
    research_brief: str
    researcher_messages: list[AnyMessage]
    raw_notes: Annotated[list[str], operator.add]
    tool_iteration: int
    search_tool_calls: int
    awaiting_reflection: bool
    compressed_research: str
    tools_used: Annotated[list[str], operator.add]
    hy3_call_budget_exhausted: bool


class ManagerOutputState(TypedDict, total=False):
    """Only durable research results may leave the manager subgraph."""

    notes: list[str]
    research_unit_count: int
    tools_used: list[str]
    hy3_call_budget_exhausted: bool


class ResearcherOutputState(TypedDict, total=False):
    """Only the cleaned memo and small metrics may leave a researcher subgraph."""

    compressed_research: str
    tools_used: list[str]
    hy3_call_budget_exhausted: bool


class SourceRecord(BaseModel):
    """Portable source envelope whose metadata remains provider-specific."""

    provider: str
    source_type: str
    evidence_level: str = "metadata"
    title: str
    url: str | None = None
    content: str = ""
    published_at: str | None = None
    authors: list[str] = Field(default_factory=list)
    identifiers: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class WebpageSummary(BaseModel):
    """从网页原文中提炼、且保留关键原句的结构化摘要。"""

    summary: str = Field(description="与查询直接相关的事实性摘要。")
    key_excerpts: list[str] = Field(
        default_factory=list,
        max_length=5,
        description="用于核验摘要的简短关键原文片段。",
    )
