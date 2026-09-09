"""Run three isolated full-evaluation rounds on frozen LangSmith reports."""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import inspect
import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langsmith import Client as LangSmithClient
from langsmith import aevaluate
from openai import AsyncOpenAI

from deep_research_assistant.config import get_settings
from scripts.run_langsmith_evaluation import (
    JUDGE_DIMENSIONS,
    QUALITY_JUDGE_PROMPT,
    citation_quality,
    llm_quality_evaluator,
    outline_structure_coverage,
    report_format,
    research_process,
    rule_evaluator,
    run_completion,
)

RULE_METRICS = (
    "run_completion",
    "research_process",
    "citation_quality",
    "outline_structure_coverage",
    "report_format",
    "rule_overall",
)
JUDGE_METRICS = (*JUDGE_DIMENSIONS, "judge_overall")
ALL_METRICS = (*RULE_METRICS, *JUDGE_METRICS)
MANIFEST_VERSION = 1

RESULT_ID_FIELDS = (
    "case_id",
    "domain",
    "difficulty",
    "evaluation_round",
    "experiment_name",
    "experiment_id",
    "experiment_url",
    "dataset_example_id",
    "source_example_id",
    "source_run_id",
    "judge_input_sha256",
    "judge_model",
    "evaluator_version",
    "evaluated_at",
)


def result_fields() -> list[str]:
    fields = list(RESULT_ID_FIELDS)
    for metric in ALL_METRICS:
        fields.extend((metric, f"{metric}_comment"))
    return fields


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _parse_json(value: Any, default: Any) -> Any:
    if value in (None, ""):
        return default
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"无法解析 JSON 字段：{value[:120]!r}") from exc


def _integer(value: Any, field: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} 必须是整数，实际为 {value!r}") from exc


def validate_examples(
    examples: Iterable[Any],
    *,
    expected_count: int | None,
) -> list[Any]:
    """Validate the imported frozen dataset and return it in case_id order."""

    normalized = list(examples)
    if expected_count is not None and len(normalized) != expected_count:
        raise ValueError(
            f"数据集样本数为 {len(normalized)}，与预期 {expected_count} 不一致。"
        )
    seen: set[str] = set()
    for example in normalized:
        inputs = _mapping(getattr(example, "inputs", None))
        metadata = _mapping(getattr(example, "metadata", None))
        case_id = str(metadata.get("case_id") or "").strip()
        required_inputs = (
            "question",
            "research_brief",
            "final_report",
            "status",
            "research_unit_count",
            "source_count",
            "tools_used",
        )
        missing = [name for name in required_inputs if inputs.get(name) in (None, "")]
        required_metadata = ("case_id", "source_run_id", "judge_input_sha256")
        missing.extend(
            f"metadata.{name}"
            for name in required_metadata
            if metadata.get(name) in (None, "")
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
        for name in (
            "research_outline",
            "evaluation_outline",
            "report_completeness_check",
        ):
            parsed = _parse_json(inputs.get(name), {})
            if not isinstance(parsed, dict):
                raise ValueError(f"样本 {case_id} 的 {name} 必须是 JSON 对象")
    return sorted(
        normalized,
        key=lambda example: str(_mapping(example.metadata).get("case_id") or ""),
    )


def consistency_target(inputs: dict[str, Any]) -> dict[str, Any]:
    """Return the frozen report and process evidence without generating content."""

    tools = _parse_json(inputs.get("tools_used"), [])
    if not isinstance(tools, list):
        raise ValueError("tools_used 必须是 JSON 数组")
    outputs = {
        "original_request": str(inputs.get("original_request") or inputs["question"]),
        "research_brief": str(inputs["research_brief"]),
        "research_outline": _parse_json(inputs.get("research_outline"), {}),
        "evaluation_outline": _parse_json(inputs.get("evaluation_outline"), {}),
        "final_report": str(inputs["final_report"]),
        "status": str(inputs.get("status") or "complete"),
        "error": str(inputs.get("error") or ""),
        "research_unit_count": _integer(
            inputs.get("research_unit_count"), "research_unit_count"
        ),
        "source_count": _integer(inputs.get("source_count"), "source_count"),
        "tools_used": tools,
        "report_completeness_check": _parse_json(
            inputs.get("report_completeness_check"), {}
        ),
    }
    return outputs


async def async_consistency_target(inputs: dict[str, Any]) -> dict[str, Any]:
    return consistency_target(inputs)


def evaluator_version() -> str:
    source = "\n".join(
        (
            QUALITY_JUDGE_PROMPT,
            inspect.getsource(run_completion),
            inspect.getsource(research_process),
            inspect.getsource(citation_quality),
            inspect.getsource(outline_structure_coverage),
            inspect.getsource(report_format),
            inspect.getsource(rule_evaluator),
            inspect.getsource(llm_quality_evaluator),
        )
    )
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]


def dataset_fingerprint(examples: Iterable[Any]) -> str:
    pairs = sorted(
        (
            str(_mapping(example.metadata).get("case_id") or ""),
            str(_mapping(example.metadata).get("judge_input_sha256") or ""),
        )
        for example in examples
    )
    payload = json.dumps(pairs, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _evaluation_items(value: Any) -> list[Any]:
    if isinstance(value, dict):
        return list(value.get("results") or [])
    return list(getattr(value, "results", None) or [])


def build_result_row(
    experiment_row: Any,
    *,
    round_number: int,
    experiment_name: str,
    experiment_id: str,
    experiment_url: str,
    judge_model: str,
    version: str,
    evaluated_at: str,
) -> tuple[dict[str, Any], list[str]]:
    """Flatten one LangSmith result without averaging duplicate feedback."""

    example = experiment_row["example"]
    metadata = _mapping(getattr(example, "metadata", None))
    results = _evaluation_items(experiment_row["evaluation_results"])
    by_key = {
        str(getattr(result, "key", "") or ""): result
        for result in results
        if getattr(result, "score", None) is not None
    }
    missing = [metric for metric in ALL_METRICS if metric not in by_key]
    row: dict[str, Any] = {
        "case_id": str(metadata.get("case_id") or ""),
        "domain": str(metadata.get("domain") or ""),
        "difficulty": str(metadata.get("difficulty") or ""),
        "evaluation_round": round_number,
        "experiment_name": experiment_name,
        "experiment_id": experiment_id,
        "experiment_url": experiment_url,
        "dataset_example_id": str(getattr(example, "id", "") or ""),
        "source_example_id": str(metadata.get("example_id") or ""),
        "source_run_id": str(metadata.get("source_run_id") or ""),
        "judge_input_sha256": str(metadata.get("judge_input_sha256") or ""),
        "judge_model": judge_model,
        "evaluator_version": version,
        "evaluated_at": evaluated_at,
    }
    for metric in ALL_METRICS:
        result = by_key.get(metric)
        row[metric] = getattr(result, "score", "") if result is not None else ""
        row[f"{metric}_comment"] = (
            str(getattr(result, "comment", "") or "") if result is not None else ""
        )
    return row, missing


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=result_fields(), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_manifest(
    path: Path,
    *,
    dataset: str,
    prefix: str,
    fingerprint: str,
    configuration_hash: str,
) -> dict[str, Any]:
    if not path.exists():
        return {
            "version": MANIFEST_VERSION,
            "dataset": dataset,
            "prefix": prefix,
            "dataset_fingerprint": fingerprint,
            "configuration_hash": configuration_hash,
            "rounds": {},
        }
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "version": MANIFEST_VERSION,
        "dataset": dataset,
        "prefix": prefix,
        "dataset_fingerprint": fingerprint,
        "configuration_hash": configuration_hash,
    }
    mismatched = [key for key, value in expected.items() if manifest.get(key) != value]
    if mismatched:
        raise RuntimeError(
            "现有一致性检查点与本次数据或配置不一致："
            + ", ".join(mismatched)
            + "。请更换 --output-dir 或 --prefix，保留旧实验结果。"
        )
    return manifest


def _configuration_hash(configuration: dict[str, Any]) -> str:
    payload = json.dumps(configuration, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def merge_completed_rounds(
    manifest: dict[str, Any],
    *,
    rounds: int,
    output_dir: Path,
) -> Path:
    merged: list[dict[str, Any]] = []
    for round_number in range(1, rounds + 1):
        state = _mapping(_mapping(manifest.get("rounds")).get(str(round_number)))
        result_path = Path(str(state.get("result_path") or ""))
        if state.get("status") != "complete" or not result_path.exists():
            raise RuntimeError(f"第 {round_number} 轮尚未完整完成，不能合并。")
        with result_path.open(encoding="utf-8-sig", newline="") as handle:
            merged.extend(csv.DictReader(handle))
    expected_rows = rounds * len(
        {str(row.get("case_id") or "") for row in merged}
    )
    if len(merged) != expected_rows:
        raise RuntimeError("各轮样本数量或 case_id 集合不一致，已停止合并。")
    output_path = output_dir / "consistency-results-long.csv"
    write_csv(output_path, merged)
    return output_path


async def _run_round(
    *,
    round_number: int,
    examples: list[Any],
    client: LangSmithClient,
    judge_client: AsyncOpenAI,
    judge_model: str,
    judge_max_tokens: int,
    judge_parse_retries: int,
    prefix: str,
    concurrency: int,
    output_dir: Path,
    version: str,
) -> dict[str, Any]:
    async def quality_judge(run: Any, example: Any) -> list[dict[str, Any]]:
        return await llm_quality_evaluator(
            run,
            example,
            client=judge_client,
            model=judge_model,
            max_tokens=judge_max_tokens,
            parse_retries=judge_parse_retries,
        )

    results = await aevaluate(
        async_consistency_target,
        data=examples,
        evaluators=[rule_evaluator, quality_judge],
        experiment_prefix=f"{prefix}-round-{round_number}",
        description=(
            f"Consistency validation round {round_number}; frozen reports; "
            "full deterministic rules and LLM judge."
        ),
        max_concurrency=concurrency,
        num_repetitions=1,
        metadata={
            "purpose": "evaluator-consistency",
            "evaluation_round": round_number,
            "judge_model": judge_model,
            "evaluator_version": version,
        },
        client=client,
        error_handling="log",
    )
    await results.wait()
    experiment_rows = [row async for row in results]
    evaluated_at = datetime.now(UTC).isoformat()
    flat_rows: list[dict[str, Any]] = []
    missing_by_case: dict[str, list[str]] = {}
    for experiment_row in experiment_rows:
        row, missing = build_result_row(
            experiment_row,
            round_number=round_number,
            experiment_name=results.experiment_name,
            experiment_id=str(results.experiment_id),
            experiment_url=str(results.url or ""),
            judge_model=judge_model,
            version=version,
            evaluated_at=evaluated_at,
        )
        flat_rows.append(row)
        if missing:
            missing_by_case[row["case_id"] or row["dataset_example_id"]] = missing

    flat_rows.sort(key=lambda row: str(row["case_id"]))
    if len(flat_rows) != len(examples) or missing_by_case:
        partial_path = output_dir / f"round-{round_number}-partial.csv"
        write_csv(partial_path, flat_rows)
        details = "; ".join(
            f"{case_id}: {','.join(keys)}" for case_id, keys in missing_by_case.items()
        )
        raise RuntimeError(
            f"第 {round_number} 轮不完整：结果数={len(flat_rows)}/{len(examples)}；"
            f"缺失评分={details or '无'}；诊断文件={partial_path.resolve()}"
        )

    result_path = output_dir / f"round-{round_number}.csv"
    write_csv(result_path, flat_rows)
    return {
        "status": "complete",
        "experiment_name": results.experiment_name,
        "experiment_id": str(results.experiment_id),
        "experiment_url": str(results.url or ""),
        "result_path": str(result_path.resolve()),
        "row_count": len(flat_rows),
        "completed_at": evaluated_at,
    }


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="对冻结报告运行三轮独立、完整且可恢复的一致性评测。"
    )
    parser.add_argument("dataset", help="已导入 LangSmith 的冻结报告数据集名称")
    parser.add_argument(
        "--prefix",
        default="benchmark-consistency-v1",
        help="三轮 LangSmith experiment 名称前缀",
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--expected-count", type=int, default=30)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/consistency/evaluation-v1"),
    )
    parser.add_argument("--judge-max-tokens", type=int, default=None)
    parser.add_argument("--judge-parse-retries", type=int, default=1)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只验证数据集与调用数量，不创建实验、不调用模型",
    )
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds 必须大于等于1")
    if args.concurrency < 1:
        parser.error("--concurrency 必须大于等于1")
    if args.expected_count < 1:
        parser.error("--expected-count 必须大于等于1")
    if not 0 <= args.judge_parse_retries <= 5:
        parser.error("--judge-parse-retries 必须在0到5之间")

    settings = get_settings()
    judge_max_tokens = args.judge_max_tokens or settings.evaluation_judge_max_tokens
    if not 256 <= judge_max_tokens <= 8000:
        parser.error("--judge-max-tokens 必须在256到8000之间")
    if not args.dry_run and not settings.deepseek_api_key:
        parser.error("未配置 DEEPSEEK_API_KEY，不能运行 LLM Judge。")

    client = LangSmithClient(
        api_url=settings.langsmith_endpoint,
        api_key=settings.langsmith_api_key or None,
    )
    examples = validate_examples(
        client.list_examples(dataset_name=args.dataset),
        expected_count=args.expected_count,
    )
    version = evaluator_version()
    fingerprint = dataset_fingerprint(examples)
    configuration = {
        "judge_model": settings.evaluation_judge_model,
        "judge_max_tokens": judge_max_tokens,
        "judge_parse_retries": args.judge_parse_retries,
        "temperature": 0,
        "thinking": "disabled",
        "evaluator_version": version,
    }
    configuration_hash = _configuration_hash(configuration)
    print(
        f"样本={len(examples)}，轮数={args.rounds}，每轮完整评分={len(ALL_METRICS)}项，"
        f"预计LLM Judge调用={len(examples) * args.rounds}次；不会生成研究报告。"
    )
    print(
        f"数据指纹={fingerprint[:16]}，评测器版本={version}，"
        f"模型={settings.evaluation_judge_model}。"
    )
    if args.dry_run:
        print("dry-run验证通过：未创建实验，未调用模型，未产生评测费用。")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    manifest = _read_manifest(
        manifest_path,
        dataset=args.dataset,
        prefix=args.prefix,
        fingerprint=fingerprint,
        configuration_hash=configuration_hash,
    )
    manifest["configuration"] = configuration
    manifest["expected_count"] = len(examples)
    _write_json(manifest_path, manifest)

    judge_client = AsyncOpenAI(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        timeout=settings.evaluation_judge_timeout_seconds,
        max_retries=settings.evaluation_judge_max_retries,
    )
    try:
        for round_number in range(1, args.rounds + 1):
            state = _mapping(_mapping(manifest.get("rounds")).get(str(round_number)))
            if state.get("status") == "complete" and Path(
                str(state.get("result_path") or "")
            ).exists():
                print(f"第 {round_number} 轮已完成，按检查点跳过。")
                continue
            print(f"开始第 {round_number}/{args.rounds} 轮完整评测……")
            completed = await _run_round(
                round_number=round_number,
                examples=examples,
                client=client,
                judge_client=judge_client,
                judge_model=settings.evaluation_judge_model,
                judge_max_tokens=judge_max_tokens,
                judge_parse_retries=args.judge_parse_retries,
                prefix=args.prefix,
                concurrency=args.concurrency,
                output_dir=args.output_dir,
                version=version,
            )
            manifest.setdefault("rounds", {})[str(round_number)] = completed
            _write_json(manifest_path, manifest)
            print(
                f"第 {round_number} 轮完成：{completed['experiment_name']}，"
                f"{completed['row_count']}条。"
            )
    finally:
        await judge_client.close()

    merged_path = merge_completed_rounds(
        manifest,
        rounds=args.rounds,
        output_dir=args.output_dir,
    )
    print(f"三轮合并完成：{merged_path.resolve()}")


if __name__ == "__main__":
    asyncio.run(main())
