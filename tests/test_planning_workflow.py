from typing import Any

import pytest
from pydantic import BaseModel

from deep_research_assistant.models import ResearchIntent, ResearchOutline
from deep_research_assistant.workflow import ClarificationRequired, PlanningWorkflow


class FakeStructuredClient:
    def chat_structured(
        self,
        messages: list[dict[str, str]],
        response_model: type[BaseModel],
        **_: Any,
    ) -> BaseModel:
        if response_model is ResearchIntent:
            return ResearchIntent(
                topic="RAG 评测方法",
                goal="梳理主要评测维度与公开数据集",
                focus_areas=["检索质量", "生成忠实性"],
                target_word_count=3000,
            )
        if response_model is ResearchOutline:
            return ResearchOutline.model_validate(
                {
                    "title": "RAG 评测方法调研",
                    "thesis": "如何系统评价 RAG 系统",
                    "sections": [
                        {
                            "section_id": "S1",
                            "title": "问题背景",
                            "objective": "定义评测对象",
                            "research_questions": ["RAG 评测包含哪些环节？"],
                        },
                        {
                            "section_id": "S2",
                            "title": "主要方法",
                            "objective": "比较评测路线",
                            "research_questions": ["有哪些自动评测方法？"],
                        },
                        {
                            "section_id": "S3",
                            "title": "局限与趋势",
                            "objective": "归纳开放问题",
                            "research_questions": ["当前方法存在哪些不足？"],
                        },
                    ],
                }
            )
        raise AssertionError("unexpected response model")


def test_planning_workflow_returns_valid_plan() -> None:
    workflow = PlanningWorkflow(client=FakeStructuredClient())  # type: ignore[arg-type]

    plan = workflow.run(
        "调研 2023-2026 年 RAG 的主要评测方法，面向大模型开发者，"
        "重点关注公开数据集，生成3000字中文标准深度报告"
    )

    assert plan.intent.topic == "RAG 评测方法"
    assert plan.intent.target_word_count == 3000
    assert [section.section_id for section in plan.outline.sections] == ["S1", "S2", "S3"]


def test_planning_workflow_waits_for_clarification() -> None:
    workflow = PlanningWorkflow(client=FakeStructuredClient())  # type: ignore[arg-type]

    with pytest.raises(ClarificationRequired) as exc_info:
        workflow.run("调研 RAG")

    assert exc_info.value.intent.clarification_questions
