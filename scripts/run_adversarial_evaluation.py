"""Evaluate frozen adversarial reports with the formal full evaluator.

This script never generates or mutates a research report. It evaluates the reports
already stored in a LangSmith dataset, then compares each adversarial score with
the corresponding base report's mean score from the consistency experiment.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from typing import Any

from langsmith import Client as LangSmithClient
from langsmith import aevaluate
from openai import AsyncOpenAI

from deep_research_assistant.config import get_settings
from scripts.run_consistency_evaluation import (
    ALL_METRICS,
    _integer,
    _mapping,
    _parse_json,
    async_consistency_target,
    evaluator_version,
)
from scripts.run_langsmith_evaluation import (
    llm_quality_evaluator,
    rule_evaluator,
)

IDENTITY_FIELDS = (
    "case_id",
    "base_case_id",
    "domain",
    "difficulty",
    "adversarial_type",
    "severity",
    "target_dimensions",
    "mutation_summary",
    "expected_behavior",
    "question",
    "experiment_name",
    "experiment_id",
    "experiment_url",
    "dataset_example_id",
    "base_example_id",
    "base_source_run_id",
    "parent_report_sha256",
    "report_sha256",
    "judge_input_sha256",
    "judge_model",
    "evaluator_version",
    "evaluated_at",
    "baseline_round_count",
)

SAMPLE_DESCRIPTOR_FIELDS = (
    "case_id",
    "base_case_id",
    "base_example_id",
    "base_source_run_id",
    "domain",
    "difficulty",
    "adversarial_type",
    "severity",
    "target_dimensions",
    "mutation_summary",
    "expected_behavior",
    "parent_report_sha256",
    "report_sha256",
    "judge_input_sha256",
)


def _sample_value(
    inputs: dict[str, Any],
    metadata: dict[str, Any],
    name: str,
) -> Any:
    """Read imported sample descriptors from metadata or, as a fallback, inputs."""

    metadata_value = metadata.get(name)
    return metadata_value if metadata_value not in (None, "") else inputs.get(name)


def _text_sha256(value: Any) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def hydrate_sample_descriptors(examples: list[Any], sample_csv: Path) -> int:
    """Fill descriptors omitted by LangSmith CSV mapping from a frozen local CSV.

    Matching is performed with the report SHA-256, so row order is irrelevant and
    a descriptor can never be attached to a different report accidentally. The
    fetched LangSmith examples are changed in memory only; the dataset is not
    mutated.
    """

    if not sample_csv.exists():
        return 0
    with sample_csv.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    manifest_by_report_hash: dict[str, dict[str, str]] = {}
    for row in rows:
        report_hash = str(row.get("report_sha256") or "").strip().casefold()
        computed_hash = _text_sha256(row.get("final_report"))
        if not report_hash:
            report_hash = computed_hash
        elif report_hash != computed_hash:
            raise ValueError(
                f"本地样本清单 {sample_csv.resolve()} 中 {row.get('case_id') or ''} "
                "的 report_sha256 与 final_report 不一致。"
            )
        if report_hash in manifest_by_report_hash:
            raise ValueError(f"本地样本清单存在重复报告哈希：{report_hash}")
        manifest_by_report_hash[report_hash] = row

    hydrated = 0
    unmatched: list[str] = []
    for example in examples:
        raw_inputs = getattr(example, "inputs", None)
        inputs = raw_inputs if isinstance(raw_inputs, dict) else {}
        metadata = _mapping(getattr(example, "metadata", None))
        if all(
            _sample_value(inputs, metadata, name) not in (None, "")
            for name in SAMPLE_DESCRIPTOR_FIELDS
        ):
            continue
        report_hash = _text_sha256(inputs.get("final_report"))
        manifest = manifest_by_report_hash.get(report_hash)
        if manifest is None:
            unmatched.append(str(getattr(example, "id", "") or ""))
            continue
        for name in SAMPLE_DESCRIPTOR_FIELDS:
            if _sample_value(inputs, metadata, name) in (None, ""):
                value = manifest.get(name)
                if value not in (None, ""):
                    inputs[name] = value
        hydrated += 1

    if unmatched:
        raise ValueError(
            "以下 LangSmith 样本无法按 final_report 哈希匹配本地清单："
            + ", ".join(unmatched)
        )
    return hydrated


def result_fields() -> list[str]:
    fields = list(IDENTITY_FIELDS)
    for metric in ALL_METRICS:
        fields.extend(
            (
                metric,
                f"{metric}_comment",
                f"baseline_{metric}_mean",
                f"delta_{metric}",
            )
        )
    fields.extend(("adversarial_pass", "adversarial_check_details"))
    return fields


def validate_examples(examples: list[Any], expected_count: int) -> list[Any]:
    if len(examples) != expected_count:
        raise ValueError(
            f"对抗数据集样本数为 {len(examples)}，与预期 {expected_count} 不一致。"
        )

    seen: set[str] = set()
    for example in examples:
        inputs = _mapping(getattr(example, "inputs", None))
        metadata = _mapping(getattr(example, "metadata", None))
        case_id = str(_sample_value(inputs, metadata, "case_id") or "").strip()
        required_inputs = (
            "question",
            "research_brief",
            "final_report",
            "status",
            "research_unit_count",
            "source_count",
            "tools_used",
        )
        required_metadata = (
            "case_id",
            "base_case_id",
            "base_source_run_id",
            "adversarial_type",
            "severity",
            "target_dimensions",
            "expected_behavior",
            "parent_report_sha256",
            "report_sha256",
            "judge_input_sha256",
        )
        missing = [name for name in required_inputs if inputs.get(name) in (None, "")]
        missing.extend(
            f"metadata或inputs.{name}"
            for name in required_metadata
            if _sample_value(inputs, metadata, name) in (None, "")
        )
        if missing:
            raise ValueError(
                f"样本 {getattr(example, 'id', '')} 缺少字段：{', '.join(missing)}"
            )
        if case_id in seen:
            raise ValueError(f"case_id 重复：{case_id}")
        seen.add(case_id)

        _integer(inputs["research_unit_count"], "research_unit_count")
        _integer(inputs["source_count"], "source_count")
        tools = _parse_json(inputs["tools_used"], [])
        if not isinstance(tools, list):
            raise ValueError(f"样本 {case_id} 的 tools_used 必须是 JSON 数组")
        for name in ("research_outline", "evaluation_outline"):
            parsed = _parse_json(inputs.get(name), {})
            if not isinstance(parsed, dict):
                raise ValueError(f"样本 {case_id} 的 {name} 必须是 JSON 对象")

    return sorted(
        examples,
        key=lambda example: str(
            _sample_value(
                _mapping(getattr(example, "inputs", None)),
                _mapping(getattr(example, "metadata", None)),
                "case_id",
            )
            or ""
        ),
    )


def load_baseline(
    path: Path,
    *,
    expected_rounds: int,
) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"找不到一致性基线结果：{path.resolve()}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("一致性基线结果为空。")

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("case_id") or "")].append(row)

    baseline: dict[str, dict[str, Any]] = {}
    for case_id, case_rows in grouped.items():
        rounds = {str(row.get("evaluation_round") or "") for row in case_rows}
        if len(rounds) != expected_rounds:
            raise ValueError(
                f"基线样本 {case_id} 的轮数为 {len(rounds)}，预期 {expected_rounds}。"
            )
        means: dict[str, float] = {}
        for metric in ALL_METRICS:
            try:
                values = [float(row[metric]) for row in case_rows]
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"基线样本 {case_id} 缺少有效评分：{metric}") from exc
            means[metric] = round(fmean(values), 4)
        baseline[case_id] = {
            "round_count": len(rounds),
            "means": means,
        }
    return baseline


def _evaluation_items(value: Any) -> list[Any]:
    if isinstance(value, dict):
        return list(value.get("results") or [])
    return list(getattr(value, "results", None) or [])


def _adversarial_checks(
    adversarial_type: str,
    deltas: dict[str, float],
) -> tuple[bool, str]:
    if adversarial_type == "length_padding":
        thresholds = {
            "judge_overall": 0.0,
            "user_readability": 0.0,
            "factual_accuracy": 0.0,
            "citation_faithfulness": 0.0,
        }
    elif adversarial_type == "jargon_stuffing":
        thresholds = {
            "judge_overall": 0.0,
            "terminology_correctness": -1.0,
            "user_readability": -1.0,
        }
    elif adversarial_type == "fabricated_citations":
        thresholds = {
            "judge_overall": 0.0,
            "citation_faithfulness": -1.0,
            "factual_accuracy": -1.0,
        }
    else:
        raise ValueError(f"不支持的 adversarial_type：{adversarial_type}")

    checks = []
    for metric, maximum_delta in thresholds.items():
        delta = deltas[metric]
        passed = delta <= maximum_delta + 1e-12
        checks.append(
            f"{metric}:delta={delta:+.4f},要求<={maximum_delta:+.4f},"
            f"{'通过' if passed else '失败'}"
        )
    return all("通过" in check for check in checks), "；".join(checks)


def build_result_row(
    experiment_row: Any,
    *,
    baseline: dict[str, dict[str, Any]],
    experiment_name: str,
    experiment_id: str,
    experiment_url: str,
    judge_model: str,
    version: str,
    evaluated_at: str,
) -> tuple[dict[str, Any], list[str]]:
    example = experiment_row["example"]
    inputs = _mapping(getattr(example, "inputs", None))
    metadata = _mapping(getattr(example, "metadata", None))
    case_id = str(_sample_value(inputs, metadata, "case_id") or "")
    base_case_id = str(_sample_value(inputs, metadata, "base_case_id") or "")
    base = baseline.get(base_case_id)
    if base is None:
        raise ValueError(f"样本 {case_id} 找不到基线报告：{base_case_id}")

    results = _evaluation_items(experiment_row["evaluation_results"])
    by_key = {
        str(getattr(result, "key", "") or ""): result
        for result in results
        if getattr(result, "score", None) is not None
    }
    missing = [metric for metric in ALL_METRICS if metric not in by_key]

    row: dict[str, Any] = {
        "case_id": case_id,
        "base_case_id": base_case_id,
        "domain": str(_sample_value(inputs, metadata, "domain") or ""),
        "difficulty": str(_sample_value(inputs, metadata, "difficulty") or ""),
        "adversarial_type": str(
            _sample_value(inputs, metadata, "adversarial_type") or ""
        ),
        "severity": str(_sample_value(inputs, metadata, "severity") or ""),
        "target_dimensions": str(
            _sample_value(inputs, metadata, "target_dimensions") or ""
        ),
        "mutation_summary": str(
            _sample_value(inputs, metadata, "mutation_summary") or ""
        ),
        "expected_behavior": str(
            _sample_value(inputs, metadata, "expected_behavior") or ""
        ),
        "question": str(inputs.get("question") or ""),
        "experiment_name": experiment_name,
        "experiment_id": experiment_id,
        "experiment_url": experiment_url,
        "dataset_example_id": str(getattr(example, "id", "") or ""),
        "base_example_id": str(
            _sample_value(inputs, metadata, "base_example_id") or ""
        ),
        "base_source_run_id": str(
            _sample_value(inputs, metadata, "base_source_run_id") or ""
        ),
        "parent_report_sha256": str(
            _sample_value(inputs, metadata, "parent_report_sha256") or ""
        ),
        "report_sha256": str(
            _sample_value(inputs, metadata, "report_sha256") or ""
        ),
        "judge_input_sha256": str(
            _sample_value(inputs, metadata, "judge_input_sha256") or ""
        ),
        "judge_model": judge_model,
        "evaluator_version": version,
        "evaluated_at": evaluated_at,
        "baseline_round_count": base["round_count"],
    }

    deltas: dict[str, float] = {}
    for metric in ALL_METRICS:
        result = by_key.get(metric)
        score = float(getattr(result, "score", 0.0)) if result is not None else 0.0
        baseline_mean = float(base["means"][metric])
        delta = round(score - baseline_mean, 4)
        row[metric] = score if result is not None else ""
        row[f"{metric}_comment"] = (
            str(getattr(result, "comment", "") or "") if result is not None else ""
        )
        row[f"baseline_{metric}_mean"] = baseline_mean
        row[f"delta_{metric}"] = delta
        deltas[metric] = delta

    if missing:
        row["adversarial_pass"] = ""
        row["adversarial_check_details"] = "缺失评分：" + ",".join(missing)
    else:
        passed, details = _adversarial_checks(row["adversarial_type"], deltas)
        row["adversarial_pass"] = passed
        row["adversarial_check_details"] = details
    return row, missing


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=result_fields(), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="对冻结的对抗报告运行正式规则评分器和八维LLM Judge。"
    )
    parser.add_argument("dataset", help="已导入 LangSmith 的对抗样本数据集名称")
    parser.add_argument(
        "--baseline-csv",
        type=Path,
        default=Path(
            "outputs/consistency/github-export/"
            "benchmark-consistency-three-rounds-grouped.csv"
        ),
    )
    parser.add_argument("--baseline-rounds", type=int, default=3)
    parser.add_argument("--expected-count", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument(
        "--sample-csv",
        type=Path,
        default=Path("outputs/adversarial/benchmark-adversarial-samples.csv"),
        help=(
            "冻结的对抗样本清单；当 LangSmith 导入时未保留 metadata 字段，"
            "脚本会按 final_report SHA-256 从该文件补齐"
        ),
    )
    parser.add_argument(
        "--prefix",
        default="benchmark-adversarial-v1",
        help="LangSmith experiment 名称前缀",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/adversarial/evaluation-v1"),
    )
    parser.add_argument("--judge-max-tokens", type=int, default=None)
    parser.add_argument("--judge-parse-retries", type=int, default=1)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只验证数据集和基线，不创建实验、不调用模型",
    )
    args = parser.parse_args()
    if args.expected_count < 1:
        parser.error("--expected-count 必须大于等于1")
    if args.baseline_rounds < 1:
        parser.error("--baseline-rounds 必须大于等于1")
    if args.concurrency < 1:
        parser.error("--concurrency 必须大于等于1")
    if not 0 <= args.judge_parse_retries <= 5:
        parser.error("--judge-parse-retries 必须在0到5之间")

    settings = get_settings()
    judge_max_tokens = args.judge_max_tokens or settings.evaluation_judge_max_tokens
    if not 256 <= judge_max_tokens <= 8000:
        parser.error("--judge-max-tokens 必须在256到8000之间")
    if not args.dry_run and not settings.deepseek_api_key:
        parser.error("未配置 DEEPSEEK_API_KEY，不能运行 LLM Judge。")

    baseline = load_baseline(args.baseline_csv, expected_rounds=args.baseline_rounds)
    client = LangSmithClient(
        api_url=settings.langsmith_endpoint,
        api_key=settings.langsmith_api_key or None,
    )
    examples = list(client.list_examples(dataset_name=args.dataset))
    hydrated_count = hydrate_sample_descriptors(examples, args.sample_csv)
    examples = validate_examples(examples, expected_count=args.expected_count)
    missing_bases = sorted(
        {
            str(
                _sample_value(
                    _mapping(getattr(example, "inputs", None)),
                    _mapping(getattr(example, "metadata", None)),
                    "base_case_id",
                )
                or ""
            )
            for example in examples
        }
        - set(baseline)
    )
    if missing_bases:
        raise ValueError("一致性基线缺少原始样本：" + ", ".join(missing_bases))

    version = evaluator_version()
    print(
        f"对抗样本={len(examples)}，每条完整评分={len(ALL_METRICS)}项，"
        f"预计LLM Judge调用={len(examples)}次；不会生成研究报告。"
    )
    print(
        f"基线={args.baseline_csv.resolve()}，每个原始样本={args.baseline_rounds}轮均值，"
        f"评测器版本={version}，模型={settings.evaluation_judge_model}。"
    )
    if hydrated_count:
        print(
            f"已从本地冻结清单补齐 {hydrated_count} 条样本的描述字段："
            f"{args.sample_csv.resolve()}"
        )
    if args.dry_run:
        print("dry-run验证通过：未创建实验，未调用模型，未产生评测费用。")
        return

    judge_client = AsyncOpenAI(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        timeout=settings.evaluation_judge_timeout_seconds,
        max_retries=settings.evaluation_judge_max_retries,
    )

    async def quality_judge(run: Any, example: Any) -> list[dict[str, Any]]:
        return await llm_quality_evaluator(
            run,
            example,
            client=judge_client,
            model=settings.evaluation_judge_model,
            max_tokens=judge_max_tokens,
            parse_retries=args.judge_parse_retries,
        )

    try:
        results = await aevaluate(
            async_consistency_target,
            data=examples,
            evaluators=[rule_evaluator, quality_judge],
            experiment_prefix=args.prefix,
            description=(
                "Adversarial validation of frozen reports; full deterministic "
                "rules and eight-dimension LLM judge; no report generation."
            ),
            max_concurrency=args.concurrency,
            num_repetitions=1,
            metadata={
                "purpose": "evaluator-adversarial-validation",
                "judge_model": settings.evaluation_judge_model,
                "evaluator_version": version,
            },
            client=client,
            error_handling="log",
        )
        await results.wait()
        experiment_rows = [row async for row in results]
    finally:
        await judge_client.close()

    evaluated_at = datetime.now(UTC).isoformat()
    flat_rows: list[dict[str, Any]] = []
    missing_by_case: dict[str, list[str]] = {}
    for experiment_row in experiment_rows:
        row, missing = build_result_row(
            experiment_row,
            baseline=baseline,
            experiment_name=results.experiment_name,
            experiment_id=str(results.experiment_id),
            experiment_url=str(results.url or ""),
            judge_model=settings.evaluation_judge_model,
            version=version,
            evaluated_at=evaluated_at,
        )
        flat_rows.append(row)
        if missing:
            missing_by_case[row["case_id"]] = missing

    flat_rows.sort(key=lambda row: str(row["case_id"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "adversarial-results.csv"
    write_csv(result_path, flat_rows)
    if len(flat_rows) != len(examples) or missing_by_case:
        details = "; ".join(
            f"{case_id}:{','.join(keys)}"
            for case_id, keys in missing_by_case.items()
        )
        raise RuntimeError(
            f"对抗评测不完整：结果数={len(flat_rows)}/{len(examples)}；"
            f"缺失评分={details or '无'}；诊断文件={result_path.resolve()}"
        )

    passed = sum(str(row["adversarial_pass"]).casefold() == "true" for row in flat_rows)
    summary = {
        "dataset": args.dataset,
        "experiment_name": results.experiment_name,
        "experiment_id": str(results.experiment_id),
        "experiment_url": str(results.url or ""),
        "sample_count": len(flat_rows),
        "passed_count": passed,
        "failed_count": len(flat_rows) - passed,
        "baseline_csv": str(args.baseline_csv.resolve()),
        "baseline_rounds": args.baseline_rounds,
        "judge_model": settings.evaluation_judge_model,
        "evaluator_version": version,
        "evaluated_at": evaluated_at,
        "criteria": {
            "length_padding": (
                "judge_overall/user_readability/factual_accuracy/"
                "citation_faithfulness均不得高于基线均值"
            ),
            "jargon_stuffing": (
                "terminology_correctness和user_readability至少下降1分，"
                "judge_overall不得上升"
            ),
            "fabricated_citations": (
                "citation_faithfulness和factual_accuracy至少下降1分，"
                "judge_overall不得上升"
            ),
        },
    }
    summary_path = args.output_dir / "adversarial-summary.json"
    write_json(summary_path, summary)
    print(
        f"对抗评测完成：通过={passed}/{len(flat_rows)}；"
        f"结果={result_path.resolve()}；摘要={summary_path.resolve()}"
    )
    print(f"LangSmith实验：{results.experiment_name}；{results.url or ''}")


if __name__ == "__main__":
    asyncio.run(main())
