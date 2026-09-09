import inspect
import json
from types import SimpleNamespace

import pytest

from scripts.run_evaluator_calibration import (
    async_calibration_target,
    calibration_rule_evaluator,
    calibration_target,
    validate_calibration_inputs,
)


def test_async_calibration_target_is_compatible_with_aevaluate() -> None:
    assert inspect.iscoroutinefunction(async_calibration_target)


def _inputs(**overrides):
    values = {
        "question": "比较三种架构",
        "outline": {
            "sections": [
                {"section_id": "S1", "title": "成本比较"},
                {"section_id": "S2", "title": "选择建议"},
            ]
        },
        "final_report": (
            "# 架构比较\n\n"
            "## 成本比较\n\n事实[1]。\n\n"
            "## 选择建议\n\n建议。\n\n"
            "## 主要来源\n\n"
            "- [1] 官方资料: https://example.com/one"
        ),
    }
    values.update(overrides)
    return values


def test_calibration_target_copies_prewritten_report_without_generation() -> None:
    inputs = _inputs()

    outputs = calibration_target(inputs)

    assert outputs["original_request"] == "比较三种架构"
    assert outputs["final_report"] == inputs["final_report"]
    assert outputs["evaluation_outline"] == inputs["outline"]
    assert json.loads(outputs["research_brief"]) == inputs["outline"]
    assert outputs["status"] == "complete"


def test_calibration_input_requires_question_outline_and_report() -> None:
    with pytest.raises(ValueError, match="outline"):
        validate_calibration_inputs(
            {"question": "问题", "final_report": "报告"},
            example_id="example-1",
        )


def test_content_only_calibration_excludes_unavailable_process_score() -> None:
    outputs = calibration_target(_inputs())

    results = calibration_rule_evaluator(SimpleNamespace(outputs=outputs), None)
    by_key = {result["key"]: result for result in results}

    assert "research_process" not in by_key
    assert set(by_key) == {
        "run_completion",
        "citation_quality",
        "outline_structure_coverage",
        "report_format",
        "rule_overall",
    }
    assert "未提供研究过程字段" in by_key["rule_overall"]["comment"]


def test_calibration_scores_process_when_optional_evidence_is_present() -> None:
    outputs = calibration_target(
        _inputs(
            research_unit_count=2,
            source_count=1,
            tools_used=["tavily_search", "think_tool"],
        )
    )

    results = calibration_rule_evaluator(SimpleNamespace(outputs=outputs), None)
    by_key = {result["key"]: result for result in results}

    assert by_key["research_process"]["score"] == 1.0
    assert by_key["rule_overall"]["score"] == 1.0
