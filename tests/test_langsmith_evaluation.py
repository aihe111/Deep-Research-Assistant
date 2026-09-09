import asyncio
import json
from types import SimpleNamespace

import pytest

from scripts.run_langsmith_evaluation import (
    JUDGE_DIMENSIONS,
    _parse_judge_response,
    citation_quality,
    llm_quality_evaluator,
    rule_evaluator,
)


def _run(report: str) -> SimpleNamespace:
    return SimpleNamespace(outputs={"final_report": report})


def test_numbered_citation_report_receives_full_integrity_score() -> None:
    report = (
        "# 测试报告\n\n"
        "第一个事实[1]，第二个事实[2]。\n\n"
        "## 主要来源\n\n"
        "- [1] 官方资料: https://example.com/one\n"
        "- [2] 研究论文: https://doi.org/10.1000/example"
    )

    result = citation_quality(_run(report), None)

    assert result["score"] == 1.0
    assert "校验错误=无" in result["comment"]


def test_missing_numbered_source_is_penalized_and_explained() -> None:
    report = (
        "# 测试报告\n\n"
        "两个事实分别引用[1][2]。\n\n"
        "## 主要来源\n\n"
        "- [1] 官方资料: https://example.com/one"
    )

    result = citation_quality(_run(report), None)

    assert result["score"] < 1.0
    assert "正文与来源双向对应" in result["comment"]
    assert "[2]" in result["comment"]


def test_empty_report_receives_zero_integrity_score() -> None:
    result = citation_quality(_run(""), None)

    assert result["score"] == 0.0
    assert "报告为空" in result["comment"]


def test_complete_rule_evaluator_returns_six_named_metrics() -> None:
    report = (
        "# 架构比较报告\n\n"
        "## 成本比较\n\n事实[1]。\n\n"
        "## 运维与扩展性\n\n事实[2]。\n\n"
        "## 选择建议\n\n综合建议。\n\n"
        "## 主要来源\n\n"
        "- [1] 官方资料: https://example.com/one\n"
        "- [2] 研究论文: https://doi.org/10.1000/example"
    )
    run = SimpleNamespace(
        outputs={
            "status": "complete",
            "final_report": report,
            "research_unit_count": 3,
            "source_count": 2,
            "tools_used": ["tavily_search", "openalex_fetch_fulltext", "think_tool"],
            "evaluation_outline": {
                "title": "架构比较报告",
                "thesis": "比较三类架构",
                "sections": [
                    {"section_id": "S1", "title": "成本比较"},
                    {"section_id": "S2", "title": "运维与扩展性"},
                    {"section_id": "S3", "title": "选择建议"},
                ],
            },
        }
    )

    results = rule_evaluator(run, None)
    by_key = {result["key"]: result for result in results}

    assert set(by_key) == {
        "run_completion",
        "research_process",
        "citation_quality",
        "outline_structure_coverage",
        "report_format",
        "rule_overall",
    }
    assert all(result["score"] == 1.0 for result in results)


def test_rule_evaluator_exposes_partial_failures() -> None:
    run = SimpleNamespace(
        outputs={
            "status": "failed",
            "final_report": "",
            "research_unit_count": 0,
            "source_count": 0,
            "tools_used": [],
            "evaluation_outline": {
                "sections": [{"section_id": "S1", "title": "缺失章节"}]
            },
            "error": "模型调用失败",
        }
    )

    results = rule_evaluator(run, None)
    by_key = {result["key"]: result for result in results}

    assert by_key["run_completion"]["score"] == 0.0
    assert by_key["research_process"]["score"] == 0.0
    assert by_key["citation_quality"]["score"] == 0.0
    assert by_key["outline_structure_coverage"]["score"] == 0.0
    assert by_key["report_format"]["score"] == 0.0
    assert by_key["rule_overall"]["score"] == 0.0


class _FakeCompletions:
    def __init__(self, content: str | list[str]) -> None:
        self.contents = content if isinstance(content, list) else [content]
        self.kwargs: dict = {}
        self.call_count = 0

    async def create(self, **kwargs):
        self.kwargs = kwargs
        index = min(self.call_count, len(self.contents) - 1)
        self.call_count += 1
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self.contents[index]),
                    finish_reason="stop",
                )
            ]
        )


class _FakeJudgeClient:
    def __init__(self, content: str | list[str]) -> None:
        self.completions = _FakeCompletions(content)
        self.chat = SimpleNamespace(completions=self.completions)


def test_llm_quality_evaluator_uses_one_non_thinking_json_request() -> None:
    payload = {
        dimension: {"score": index % 5, "comment": f"{dimension} 的具体理由"}
        for index, dimension in enumerate(JUDGE_DIMENSIONS)
    }
    client = _FakeJudgeClient(json.dumps(payload, ensure_ascii=False))
    run = SimpleNamespace(
        outputs={
            "final_report": "# 完整报告\n\n正文[1]。",
            "research_brief": "研究大纲包含效果、成本与建议。",
        }
    )
    example = SimpleNamespace(inputs={"question": "比较三种方案"})

    results = asyncio.run(
        llm_quality_evaluator(
            run,
            example,
            client=client,
            model="deepseek-v4-flash",
            max_tokens=1500,
        )
    )
    by_key = {result["key"]: result for result in results}

    assert set(by_key) == {*JUDGE_DIMENSIONS, "judge_overall"}
    assert by_key["factual_accuracy"]["score"] == 0
    assert by_key["outline_semantic_coverage"]["score"] == 1
    assert client.completions.kwargs["response_format"] == {"type": "json_object"}
    assert client.completions.kwargs["extra_body"] == {
        "thinking": {"type": "disabled"}
    }
    request_text = client.completions.kwargs["messages"][1]["content"]
    assert "比较三种方案" in request_text
    assert "研究大纲包含效果、成本与建议" in request_text
    assert "# 完整报告" in request_text
    assert client.completions.call_count == 1


def test_llm_quality_evaluator_retries_invalid_json_response() -> None:
    payload = {
        dimension: {"score": 4, "comment": f"{dimension} 的具体理由"}
        for dimension in JUDGE_DIMENSIONS
    }
    client = _FakeJudgeClient(
        ["{\"factual_accuracy\":", json.dumps(payload, ensure_ascii=False)]
    )

    results = asyncio.run(
        llm_quality_evaluator(
            SimpleNamespace(outputs={"final_report": "# 完整报告\n\n正文[1]。"}),
            SimpleNamespace(inputs={"question": "测试问题"}),
            client=client,
            model="deepseek-v4-flash",
            max_tokens=3000,
            parse_retries=1,
        )
    )

    assert client.completions.call_count == 2
    assert {result["key"] for result in results} == {
        *JUDGE_DIMENSIONS,
        "judge_overall",
    }
    assert results[-1]["score"] == 4.0


def test_llm_quality_evaluator_reports_final_invalid_response() -> None:
    client = _FakeJudgeClient(["not json", "still not json"])

    with pytest.raises(ValueError, match="连续 2 次.*finish_reason='stop'"):
        asyncio.run(
            llm_quality_evaluator(
                SimpleNamespace(outputs={"final_report": "# 完整报告\n\n正文[1]。"}),
                SimpleNamespace(inputs={"question": "测试问题"}),
                client=client,
                model="deepseek-v4-flash",
                max_tokens=3000,
                parse_retries=1,
            )
        )


def test_empty_report_skips_llm_quality_request() -> None:
    client = _FakeJudgeClient("should not be used")

    results = asyncio.run(
        llm_quality_evaluator(
            SimpleNamespace(outputs={"final_report": ""}),
            SimpleNamespace(inputs={"question": "测试问题"}),
            client=client,
            model="deepseek-v4-flash",
            max_tokens=1500,
        )
    )

    assert len(results) == 9
    assert all(result["score"] == 0 for result in results)
    assert client.completions.kwargs == {}


def test_invalid_judge_score_is_rejected_instead_of_logged_as_low_quality() -> None:
    payload = {
        dimension: {"score": 4, "comment": "具体理由"}
        for dimension in JUDGE_DIMENSIONS
    }
    payload["factual_accuracy"]["score"] = 5

    with pytest.raises(ValueError, match="0 到 4"):
        _parse_judge_response(json.dumps(payload, ensure_ascii=False))
