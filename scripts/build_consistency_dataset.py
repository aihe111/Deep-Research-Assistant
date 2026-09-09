"""Freeze the latest successful reports into a consistency-evaluation CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import uuid
import warnings
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langsmith import Client as LangSmithClient

from deep_research_assistant.config import get_settings

CSV_FIELDS = (
    "case_id",
    "domain",
    "difficulty",
    "example_id",
    "source_run_id",
    "source_run_start_time",
    "source_experiment",
    "frozen_at",
    "question",
    "original_request",
    "research_brief",
    "research_outline",
    "evaluation_outline",
    "final_report",
    "status",
    "error",
    "research_unit_count",
    "source_count",
    "tools_used",
    "report_completeness_check",
    "report_sha256",
    "judge_input_sha256",
)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _metadata(run: Any) -> dict[str, Any]:
    metadata = getattr(run, "metadata", None)
    if isinstance(metadata, dict):
        return metadata
    return _mapping(_mapping(getattr(run, "extra", None)).get("metadata"))


def _field(run: Any, name: str) -> str:
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


def _is_successful_report(run: Any) -> bool:
    outputs = _mapping(getattr(run, "outputs", None))
    return (
        not getattr(run, "error", None)
        and outputs.get("status") == "complete"
        and bool(str(outputs.get("final_report") or "").strip())
        and bool(_field(run, "case_id"))
    )


def _run_order(run: Any) -> tuple[str, str]:
    return (
        str(getattr(run, "start_time", "") or ""),
        str(getattr(run, "id", "") or ""),
    )


def select_latest_reports(runs: Iterable[Any]) -> dict[str, Any]:
    """Select the newest successful report run for each case_id."""

    selected: dict[str, Any] = {}
    for run in runs:
        if not _is_successful_report(run):
            continue
        case_id = _field(run, "case_id")
        previous = selected.get(case_id)
        if previous is None or _run_order(run) > _run_order(previous):
            selected[case_id] = run
    return selected


def _json_cell(value: Any, default: Any) -> str:
    normalized = default if value in (None, "") else value
    if isinstance(normalized, str):
        try:
            normalized = json.loads(normalized)
        except json.JSONDecodeError:
            pass
    return json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _iso(value: Any) -> str:
    if value is None:
        return ""
    isoformat = getattr(value, "isoformat", None)
    return str(isoformat() if callable(isoformat) else value)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_row(
    run: Any,
    *,
    experiment: str,
    frozen_at: str,
) -> dict[str, Any]:
    """Build one frozen row containing every field used by the evaluators."""

    outputs = _mapping(getattr(run, "outputs", None))
    question = _field(run, "question")
    final_report = str(outputs.get("final_report") or "").strip()
    research_brief = str(outputs.get("research_brief") or "").strip()
    case_id = _field(run, "case_id")
    missing = [
        name
        for name, value in (
            ("case_id", case_id),
            ("question", question),
            ("research_brief", research_brief),
            ("final_report", final_report),
        )
        if not value
    ]
    if missing:
        raise ValueError(f"运行 {getattr(run, 'id', '')} 缺少字段：{', '.join(missing)}")

    judge_input = json.dumps(
        {
            "question": question,
            "research_brief": research_brief,
            "final_report": final_report,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "case_id": case_id,
        "domain": _field(run, "domain"),
        "difficulty": _field(run, "difficulty"),
        "example_id": str(getattr(run, "reference_example_id", "") or ""),
        "source_run_id": str(getattr(run, "id", "") or ""),
        "source_run_start_time": _iso(getattr(run, "start_time", None)),
        "source_experiment": experiment,
        "frozen_at": frozen_at,
        "question": question,
        "original_request": str(outputs.get("original_request") or question).strip(),
        "research_brief": research_brief,
        "research_outline": _json_cell(outputs.get("research_outline"), {}),
        "evaluation_outline": _json_cell(outputs.get("evaluation_outline"), {}),
        "final_report": final_report,
        "status": str(outputs.get("status") or ""),
        "error": str(outputs.get("error") or ""),
        "research_unit_count": int(outputs.get("research_unit_count") or 0),
        "source_count": int(outputs.get("source_count") or 0),
        "tools_used": _json_cell(outputs.get("tools_used"), []),
        "report_completeness_check": _json_cell(
            outputs.get("report_completeness_check"), {}
        ),
        "report_sha256": _sha256(final_report),
        "judge_input_sha256": _sha256(judge_input),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write the frozen dataset atomically with an Excel-friendly UTF-8 BOM."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _experiment_project(client: LangSmithClient, experiment: str) -> Any:
    try:
        uuid.UUID(experiment)
    except ValueError:
        return client.read_project(project_name=experiment)
    return client.read_project(project_id=experiment)


def freeze_latest_reports(
    client: LangSmithClient,
    experiment: str,
    output: Path,
    *,
    expected_count: int | None,
) -> list[dict[str, Any]]:
    """Read, validate, and freeze the latest successful report for every case."""

    project = _experiment_project(client, experiment)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"list_runs\(\) is deprecated.*",
            category=DeprecationWarning,
        )
        all_runs = list(client.list_runs(project_id=project.id, is_root=True))
    selected = select_latest_reports(all_runs)
    if expected_count is not None and len(selected) != expected_count:
        raise RuntimeError(
            f"最新成功报告数为 {len(selected)}，与预期 {expected_count} 不一致；"
            "为避免生成不完整数据集，已停止。"
        )

    frozen_at = datetime.now(UTC).isoformat()
    rows = [
        build_row(selected[case_id], experiment=experiment, frozen_at=frozen_at)
        for case_id in sorted(selected)
    ]
    hashes = [row["judge_input_sha256"] for row in rows]
    if len(hashes) != len(set(hashes)):
        raise RuntimeError("存在重复的评审输入内容；为避免一致性分析重复计权，已停止。")
    write_csv(output, rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="冻结 LangSmith 实验中每个 case_id 的最新成功报告，不调用模型。"
    )
    parser.add_argument("experiment", help="报告生成 experiment 名称或 UUID")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/consistency/benchmark-consistency-input.csv"),
        help="冻结 CSV 输出路径",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=30,
        help="预期唯一样本数；设为0可关闭数量校验，默认30",
    )
    args = parser.parse_args()
    if args.expected_count < 0:
        parser.error("--expected-count 不能为负数")

    settings = get_settings()
    client = LangSmithClient(
        api_url=settings.langsmith_endpoint,
        api_key=settings.langsmith_api_key or None,
    )
    rows = freeze_latest_reports(
        client,
        args.experiment,
        args.output,
        expected_count=args.expected_count or None,
    )
    print(f"已冻结 {len(rows)} 篇最新成功报告；未生成报告，未运行评分器。")
    print(f"输出：{args.output.resolve()}")


if __name__ == "__main__":
    main()
