"""Export the latest valid LangSmith evaluation for every benchmark case."""

from __future__ import annotations

import argparse
import csv
import re
import uuid
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langsmith import Client as LangSmithClient

from deep_research_assistant.config import get_settings

RULE_METRICS = (
    "run_completion",
    "research_process",
    "citation_quality",
    "outline_structure_coverage",
    "report_format",
    "rule_overall",
)

JUDGE_METRICS = (
    "factual_accuracy",
    "outline_semantic_coverage",
    "citation_faithfulness",
    "comparison_reasoning",
    "recommendation_actionability",
    "terminology_correctness",
    "user_readability",
    "safety_compliance",
    "judge_overall",
)

REQUIRED_METRICS = (*RULE_METRICS, *JUDGE_METRICS)

# Match the score-column order used by LangSmith's experiment CSV export.
SCORE_EXPORT_ORDER = (
    "citation_faithfulness",
    "citation_quality",
    "comparison_reasoning",
    "factual_accuracy",
    "judge_overall",
    "outline_semantic_coverage",
    "outline_structure_coverage",
    "recommendation_actionability",
    "report_format",
    "research_process",
    "rule_overall",
    "run_completion",
    "safety_compliance",
    "terminology_correctness",
    "user_readability",
)


@dataclass(frozen=True)
class SelectionSummary:
    """Counts explaining how the canonical runs were selected."""

    root_runs: int
    successful_runs: int
    excluded_runs: int
    superseded_runs: int


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _metadata(run: Any) -> dict[str, Any]:
    metadata = getattr(run, "metadata", None)
    if isinstance(metadata, dict):
        return metadata
    extra = _mapping(getattr(run, "extra", None))
    return _mapping(extra.get("metadata"))


def _text_field(run: Any, name: str) -> str:
    """Read a benchmark field from trace metadata, outputs, or inputs."""

    metadata = _metadata(run)
    outputs = _mapping(getattr(run, "outputs", None))
    inputs = _mapping(getattr(run, "inputs", None))
    candidates = (
        metadata.get(f"ls_example_{name}"),
        metadata.get(name),
        outputs.get(name),
        inputs.get(name),
    )
    return next((str(value).strip() for value in candidates if value not in (None, "")), "")


def _run_order(run: Any) -> tuple[str, str]:
    return (
        str(getattr(run, "start_time", "") or ""),
        str(getattr(run, "id", "") or ""),
    )


def _is_complete_report(run: Any) -> bool:
    outputs = _mapping(getattr(run, "outputs", None))
    return (
        not getattr(run, "error", None)
        and outputs.get("status") == "complete"
        and bool(str(outputs.get("final_report") or "").strip())
        and bool(_text_field(run, "case_id"))
    )


def select_latest_successful_runs(
    runs: Iterable[Any],
) -> tuple[dict[str, Any], SelectionSummary]:
    """Keep only the newest successful report run for each case_id."""

    all_runs = list(runs)
    successful = [run for run in all_runs if _is_complete_report(run)]
    selected: dict[str, Any] = {}
    for run in successful:
        case_id = _text_field(run, "case_id")
        previous = selected.get(case_id)
        if previous is None or _run_order(run) > _run_order(previous):
            selected[case_id] = run
    return selected, SelectionSummary(
        root_runs=len(all_runs),
        successful_runs=len(successful),
        excluded_runs=len(all_runs) - len(successful),
        superseded_runs=len(successful) - len(selected),
    )


def _feedback_order(feedback: Any) -> tuple[str, str, str]:
    return (
        str(getattr(feedback, "modified_at", "") or ""),
        str(getattr(feedback, "created_at", "") or ""),
        str(getattr(feedback, "id", "") or ""),
    )


def latest_scored_feedback(feedbacks: Iterable[Any]) -> dict[str, Any]:
    """Choose the newest numeric feedback for every metric key."""

    selected: dict[str, Any] = {}
    for feedback in feedbacks:
        key = str(getattr(feedback, "key", "") or "")
        if key not in REQUIRED_METRICS or getattr(feedback, "score", None) is None:
            continue
        previous = selected.get(key)
        if previous is None or _feedback_order(feedback) > _feedback_order(previous):
            selected[key] = feedback
    return selected


def _iso(value: Any) -> str:
    if value is None:
        return ""
    isoformat = getattr(value, "isoformat", None)
    return str(isoformat() if callable(isoformat) else value)


def build_export_row(run: Any, feedbacks: Iterable[Any]) -> tuple[dict[str, Any], list[str]]:
    """Build one auditable CSV row and report any missing score keys."""

    outputs = _mapping(getattr(run, "outputs", None))
    latest = latest_scored_feedback(feedbacks)
    missing = [key for key in REQUIRED_METRICS if key not in latest]
    row: dict[str, Any] = {
        "case_id": _text_field(run, "case_id"),
        "domain": _text_field(run, "domain"),
        "difficulty": _text_field(run, "difficulty"),
        "example_id": str(getattr(run, "reference_example_id", "") or ""),
        "run_id": str(getattr(run, "id", "") or ""),
        "run_start_time": _iso(getattr(run, "start_time", None)),
        "question": _text_field(run, "question"),
        "research_brief": str(outputs.get("research_brief") or ""),
        "research_outline": str(outputs.get("research_outline") or ""),
        "final_report": str(outputs.get("final_report") or ""),
    }
    for key in REQUIRED_METRICS:
        feedback = latest.get(key)
        row[key] = getattr(feedback, "score", "") if feedback is not None else ""
        row[f"{key}_comment"] = (
            str(getattr(feedback, "comment", "") or "") if feedback is not None else ""
        )
        row[f"{key}_created_at"] = (
            _iso(getattr(feedback, "created_at", None)) if feedback is not None else ""
        )
    return row, missing


def _fieldnames() -> list[str]:
    fields = [
        "case_id",
        "domain",
        "difficulty",
        "example_id",
        "run_id",
        "run_start_time",
        "question",
        "research_brief",
        "research_outline",
        "final_report",
    ]
    for key in REQUIRED_METRICS:
        fields.extend((key, f"{key}_comment", f"{key}_created_at"))
    return fields


def _score_fieldnames() -> list[str]:
    return [
        "case_id",
        "domain",
        "difficulty",
        "example_id",
        "run_id",
        "run_start_time",
        *SCORE_EXPORT_ORDER,
    ]


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    scores_only: bool = False,
) -> None:
    """Write an Excel-friendly UTF-8 CSV atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = _score_fieldnames() if scores_only else _fieldnames()
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _experiment_project(client: LangSmithClient, experiment: str) -> Any:
    try:
        uuid.UUID(experiment)
    except ValueError:
        return client.read_project(project_name=experiment)
    return client.read_project(project_id=experiment)


def _safe_name(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-._")
    return normalized or "langsmith-evaluation"


def _default_output(experiment: str) -> Path:
    return Path("outputs") / f"{_safe_name(experiment)}-latest.csv"


def export_latest_evaluation(
    client: LangSmithClient,
    experiment: str,
    output: Path,
    expected_count: int | None,
    scores_only: bool = False,
) -> tuple[list[dict[str, Any]], SelectionSummary]:
    """Read LangSmith without running models and export canonical evaluations."""

    project = _experiment_project(client, experiment)
    # The current async Runs API is incompatible with the httpx2 timeout object
    # bundled with this Python 3.14 environment. The stable sync reader remains
    # supported until 2027 and is the proven path for this local export utility.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"list_runs\(\) is deprecated.*",
            category=DeprecationWarning,
        )
        runs = list(client.list_runs(project_id=project.id, is_root=True))
    selected, summary = select_latest_successful_runs(runs)
    if expected_count is not None and len(selected) != expected_count:
        raise RuntimeError(
            f"唯一成功 case_id 数量为 {len(selected)}，与预期 {expected_count} 不一致；"
            "为避免输出不完整文件，已停止导出。"
        )

    rows: list[dict[str, Any]] = []
    missing_by_case: dict[str, list[str]] = {}
    for case_id in sorted(selected):
        run = selected[case_id]
        feedbacks = list(client.list_feedback(run_ids=[run.id], limit=100))
        row, missing = build_export_row(run, feedbacks)
        rows.append(row)
        if missing:
            missing_by_case[case_id] = missing

    if missing_by_case:
        details = "; ".join(
            f"{case_id}: {','.join(keys)}" for case_id, keys in missing_by_case.items()
        )
        raise RuntimeError(
            "下列最新成功报告缺少完整评分；为避免导出混合结果，已停止：" + details
        )

    write_csv(output, rows, scores_only=scores_only)
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="导出每个 case_id 最新成功报告的最新有效评分；不调用任何模型。"
    )
    parser.add_argument("experiment", help="LangSmith experiment 名称或 UUID")
    parser.add_argument("--output", type=Path, help="输出 CSV 路径")
    parser.add_argument(
        "--scores-only",
        action="store_true",
        help="只导出标识字段和与 LangSmith 表格对应的 15 个最新评分",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=30,
        help="预期唯一 case_id 数量；设为 0 可关闭数量校验，默认 30",
    )
    args = parser.parse_args()
    if args.expected_count < 0:
        parser.error("--expected-count 不能为负数")

    settings = get_settings()
    client = LangSmithClient(
        api_url=settings.langsmith_endpoint,
        api_key=settings.langsmith_api_key or None,
    )
    output = args.output or _default_output(args.experiment)
    rows, summary = export_latest_evaluation(
        client,
        args.experiment,
        output,
        args.expected_count or None,
        scores_only=args.scores_only,
    )
    print(f"实验：{args.experiment}")
    print(
        "根运行数="
        f"{summary.root_runs}，成功候选={summary.successful_runs}，"
        f"排除失败/不完整={summary.excluded_runs}，排除旧重复={summary.superseded_runs}"
    )
    print(f"已导出 {len(rows)} 个唯一 case_id，每条包含 {len(REQUIRED_METRICS)} 个评分。")
    print(f"输出：{output.resolve()}")
    print(f"导出时间：{datetime.now(UTC).isoformat()}")
if __name__ == "__main__":
    main()
