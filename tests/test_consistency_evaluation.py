from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from scripts.run_consistency_evaluation import (
    ALL_METRICS,
    build_result_row,
    consistency_target,
    dataset_fingerprint,
    validate_examples,
)


def _example(case_id: str = "academic_001") -> SimpleNamespace:
    return SimpleNamespace(
        id=f"dataset-{case_id}",
        inputs={
            "question": "测试问题",
            "original_request": "测试问题",
            "research_brief": "研究简报",
            "research_outline": json.dumps({"sections": [{"title": "章节"}]}),
            "evaluation_outline": "{}",
            "final_report": "# 报告\n\n## 章节\n内容[1]\n\n## 主要来源\n- [1] 来源: https://x.test",
            "status": "complete",
            "error": "",
            "research_unit_count": "3",
            "source_count": "5",
            "tools_used": '["tavily_search","openalex_search"]',
            "report_completeness_check": '{"is_complete":true}',
        },
        metadata={
            "case_id": case_id,
            "domain": "academic",
            "difficulty": "easy",
            "example_id": f"source-{case_id}",
            "source_run_id": f"run-{case_id}",
            "judge_input_sha256": (case_id + "0" * 64)[:64],
        },
    )


def test_validate_and_target_preserve_formal_evaluator_fields() -> None:
    example = _example()

    validated = validate_examples([example], expected_count=1)
    outputs = consistency_target(validated[0].inputs)

    assert outputs["status"] == "complete"
    assert outputs["research_unit_count"] == 3
    assert outputs["source_count"] == 5
    assert outputs["tools_used"] == ["tavily_search", "openalex_search"]
    assert outputs["research_outline"]["sections"][0]["title"] == "章节"
    assert outputs["report_completeness_check"]["is_complete"] is True


def test_validation_rejects_duplicate_case_id() -> None:
    with pytest.raises(ValueError, match="case_id 重复"):
        validate_examples([_example(), _example()], expected_count=2)


def test_dataset_fingerprint_is_order_independent() -> None:
    first = _example("academic_001")
    second = _example("finance_001")

    assert dataset_fingerprint([first, second]) == dataset_fingerprint([second, first])


def test_build_result_row_flattens_all_15_scores() -> None:
    example = _example()
    run = SimpleNamespace(id="evaluation-run-1")
    evaluation_results = {
        "results": [
            SimpleNamespace(key=metric, score=index / 10, comment=f"评语-{metric}")
            for index, metric in enumerate(ALL_METRICS)
        ]
    }
    experiment_row = {
        "run": run,
        "example": example,
        "evaluation_results": evaluation_results,
    }

    row, missing = build_result_row(
        experiment_row,
        round_number=2,
        experiment_name="consistency-round-2-abc",
        experiment_id="experiment-2",
        experiment_url="https://example.test/experiment-2",
        judge_model="deepseek-v4-flash",
        version="version123",
        evaluated_at="2026-09-08T00:00:00+00:00",
    )

    assert missing == []
    assert row["case_id"] == "academic_001"
    assert row["evaluation_round"] == 2
    assert row["judge_overall"] == (len(ALL_METRICS) - 1) / 10
    assert row["judge_overall_comment"] == "评语-judge_overall"
