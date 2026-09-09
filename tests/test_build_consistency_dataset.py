from __future__ import annotations

import csv
import json
from types import SimpleNamespace

import pytest

from scripts.build_consistency_dataset import (
    CSV_FIELDS,
    build_row,
    select_latest_reports,
    write_csv,
)


def _run(
    run_id: str,
    case_id: str,
    started: str,
    *,
    error: str | None = None,
    status: str = "complete",
    report: str = "# 报告\n\n## 正文\n内容[1]\n\n## 主要来源\n- [1] 来源: https://example.com",
) -> SimpleNamespace:
    return SimpleNamespace(
        id=run_id,
        start_time=started,
        error=error,
        reference_example_id=f"example-{case_id}",
        inputs={"question": f"问题-{case_id}"},
        metadata={
            "ls_example_case_id": case_id,
            "ls_example_domain": "academic",
            "ls_example_difficulty": "hard",
        },
        outputs={
            "status": status,
            "error": "" if error is None else error,
            "original_request": f"问题-{case_id}",
            "research_brief": "研究简报\n用户确认的报告大纲：{\"sections\":[]}",
            "research_outline": {"sections": [{"title": "正文"}]},
            "final_report": report,
            "research_unit_count": 3,
            "source_count": 5,
            "tools_used": ["tavily_search", "openalex_search"],
            "report_completeness_check": {"is_complete": True, "score": 4},
        },
    )


def test_select_latest_reports_excludes_failed_and_old_runs() -> None:
    selected = select_latest_reports(
        [
            _run("old", "academic_001", "2026-09-01T10:00:00Z"),
            _run("new", "academic_001", "2026-09-01T11:00:00Z"),
            _run("failed", "academic_002", "2026-09-01T12:00:00Z", error="boom"),
            _run("empty", "academic_003", "2026-09-01T12:00:00Z", report=""),
        ]
    )

    assert set(selected) == {"academic_001"}
    assert selected["academic_001"].id == "new"


def test_build_row_preserves_evaluator_inputs_and_process_fields() -> None:
    row = build_row(
        _run("run-1", "academic_001", "2026-09-01T11:00:00Z"),
        experiment="benchmark-reports-v1",
        frozen_at="2026-09-02T00:00:00+00:00",
    )

    assert row["case_id"] == "academic_001"
    assert row["question"] == "问题-academic_001"
    assert row["research_unit_count"] == 3
    assert row["source_count"] == 5
    assert json.loads(row["tools_used"]) == ["tavily_search", "openalex_search"]
    assert json.loads(row["research_outline"])["sections"][0]["title"] == "正文"
    assert len(row["report_sha256"]) == 64
    assert len(row["judge_input_sha256"]) == 64


def test_build_row_rejects_missing_research_brief() -> None:
    run = _run("run-1", "academic_001", "2026-09-01T11:00:00Z")
    run.outputs["research_brief"] = ""

    with pytest.raises(ValueError, match="research_brief"):
        build_row(run, experiment="experiment", frozen_at="now")


def test_write_csv_is_complete_and_excel_friendly(tmp_path) -> None:
    row = build_row(
        _run("run-1", "academic_001", "2026-09-01T11:00:00Z"),
        experiment="benchmark-reports-v1",
        frozen_at="2026-09-02T00:00:00+00:00",
    )
    output = tmp_path / "frozen.csv"

    write_csv(output, [row])

    assert output.read_bytes().startswith(b"\xef\xbb\xbf")
    with output.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert tuple(reader.fieldnames or ()) == CSV_FIELDS
    assert len(rows) == 1
    assert rows[0]["final_report"].startswith("# 报告")
