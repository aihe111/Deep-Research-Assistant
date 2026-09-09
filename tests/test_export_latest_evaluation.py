from __future__ import annotations

import csv
from datetime import UTC, datetime
from types import SimpleNamespace

from scripts.export_latest_evaluation import (
    REQUIRED_METRICS,
    SCORE_EXPORT_ORDER,
    build_export_row,
    latest_scored_feedback,
    select_latest_successful_runs,
    write_csv,
)


def _run(
    run_id: str,
    case_id: str,
    started: str,
    *,
    error: str | None = None,
    status: str = "complete",
    report: str = "完整报告",
) -> SimpleNamespace:
    return SimpleNamespace(
        id=run_id,
        start_time=started,
        error=error,
        inputs={"question": f"问题-{case_id}"},
        outputs={
            "status": status,
            "final_report": report,
            "research_brief": "简报",
            "research_outline": "大纲",
        },
        metadata={
            "ls_example_case_id": case_id,
            "ls_example_domain": "finance",
            "ls_example_difficulty": "hard",
        },
        reference_example_id=f"example-{case_id}",
    )


def _feedback(key: str, score: float | None, created: str, comment: str = ""):
    return SimpleNamespace(
        id=f"{key}-{created}",
        key=key,
        score=score,
        comment=comment,
        created_at=created,
        modified_at=created,
    )


def test_selects_latest_success_and_excludes_failures() -> None:
    runs = [
        _run("old", "finance_001", "2026-09-01T10:00:00Z"),
        _run("new", "finance_001", "2026-09-01T11:00:00Z"),
        _run("failed", "finance_002", "2026-09-01T12:00:00Z", error="boom"),
        _run("empty", "finance_003", "2026-09-01T12:00:00Z", report=""),
        _run("ok", "finance_004", "2026-09-01T12:00:00Z"),
    ]

    selected, summary = select_latest_successful_runs(runs)

    assert set(selected) == {"finance_001", "finance_004"}
    assert selected["finance_001"].id == "new"
    assert summary.root_runs == 5
    assert summary.successful_runs == 3
    assert summary.excluded_runs == 2
    assert summary.superseded_runs == 1


def test_latest_scored_feedback_ignores_errors_and_old_scores() -> None:
    feedbacks = [
        _feedback("factual_accuracy", 2, "2026-09-01T10:00:00Z", "旧评分"),
        _feedback("factual_accuracy", 4, "2026-09-01T11:00:00Z", "新评分"),
        _feedback("factual_accuracy", None, "2026-09-01T12:00:00Z", "解析错误"),
        _feedback("quality_judge", None, "2026-09-01T13:00:00Z", "错误反馈"),
    ]

    selected = latest_scored_feedback(feedbacks)

    assert set(selected) == {"factual_accuracy"}
    assert selected["factual_accuracy"].score == 4
    assert selected["factual_accuracy"].comment == "新评分"


def test_build_row_has_all_latest_metrics() -> None:
    run = _run("run-1", "finance_001", "2026-09-01T11:00:00Z")
    feedbacks = [
        _feedback(key, float(index), f"2026-09-01T11:{index:02d}:00Z", key)
        for index, key in enumerate(REQUIRED_METRICS)
    ]

    row, missing = build_export_row(run, feedbacks)

    assert missing == []
    assert row["case_id"] == "finance_001"
    assert row["domain"] == "finance"
    assert row["difficulty"] == "hard"
    assert row["factual_accuracy_comment"] == "factual_accuracy"
    assert row["judge_overall"] == float(REQUIRED_METRICS.index("judge_overall"))


def test_write_csv_is_excel_friendly_utf8(tmp_path) -> None:
    row, _ = build_export_row(
        _run("run-1", "finance_001", "2026-09-01T11:00:00Z"),
        [
            _feedback(key, 4, datetime.now(UTC).isoformat(), "中文评语")
            for key in REQUIRED_METRICS
        ],
    )
    output = tmp_path / "latest.csv"

    write_csv(output, [row])

    assert output.read_bytes().startswith(b"\xef\xbb\xbf")
    with output.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["case_id"] == "finance_001"
    assert rows[0]["factual_accuracy_comment"] == "中文评语"


def test_write_scores_only_matches_langsmith_score_columns(tmp_path) -> None:
    row, _ = build_export_row(
        _run("run-1", "finance_001", "2026-09-01T11:00:00Z"),
        [
            _feedback(key, 4, datetime.now(UTC).isoformat(), "不应导出的评语")
            for key in REQUIRED_METRICS
        ],
    )
    output = tmp_path / "scores.csv"

    write_csv(output, [row], scores_only=True)

    with output.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert reader.fieldnames == [
        "case_id",
        "domain",
        "difficulty",
        "example_id",
        "run_id",
        "run_start_time",
        *SCORE_EXPORT_ORDER,
    ]
    assert rows[0]["judge_overall"] == "4"
    assert "factual_accuracy_comment" not in rows[0]
