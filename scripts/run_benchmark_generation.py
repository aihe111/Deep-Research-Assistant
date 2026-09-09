"""Generate one resumable batch of reports from a LangSmith benchmark dataset."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langsmith import Client as LangSmithClient
from langsmith import aevaluate
from langsmith.schemas import Example

from deep_research_assistant.artifact_store import ArtifactStore
from deep_research_assistant.config import Settings, get_settings
from deep_research_assistant.deep_research_graph import build_deep_research_graph

MANIFEST_VERSION = 1
COMPLETE_STATUS = "complete"


@dataclass(frozen=True)
class BenchmarkCase:
    """Normalized fields needed to generate and checkpoint one dataset example."""

    example: Example
    example_id: str
    case_id: str
    domain: str
    difficulty: str
    question: str


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _safe_name(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-._")
    return normalized or "benchmark"


def _field(inputs: dict[str, Any], metadata: dict[str, Any], name: str) -> str:
    value = metadata.get(name)
    if value in (None, ""):
        value = inputs.get(name)
    return str(value or "").strip()


def normalize_example(example: Example) -> BenchmarkCase:
    """Read benchmark labels from metadata or inputs and validate required fields."""

    inputs = dict(example.inputs or {})
    metadata = dict(example.metadata or {})
    question = str(inputs.get("question") or "").strip()
    case_id = _field(inputs, metadata, "case_id")
    domain = _field(inputs, metadata, "domain")
    difficulty = _field(inputs, metadata, "difficulty")
    missing = [
        name
        for name, value in (
            ("case_id", case_id),
            ("domain", domain),
            ("difficulty", difficulty),
            ("question", question),
        )
        if not value
    ]
    if missing:
        raise ValueError(
            f"LangSmith 样本 {example.id} 缺少字段：{', '.join(missing)}。"
            "question 应为 input；case_id/domain/difficulty 可放在 metadata 或 input。"
        )
    return BenchmarkCase(
        example=example,
        example_id=str(example.id),
        case_id=case_id,
        domain=domain,
        difficulty=difficulty,
        question=question,
    )


def normalize_examples(examples: list[Example]) -> list[BenchmarkCase]:
    cases = [normalize_example(example) for example in examples]
    duplicate_ids = sorted(
        case_id
        for case_id in {case.case_id for case in cases}
        if sum(case.case_id == case_id for case in cases) > 1
    )
    if duplicate_ids:
        raise ValueError("case_id 必须唯一，重复值：" + ", ".join(duplicate_ids))
    return sorted(cases, key=lambda case: case.case_id)


def select_labeled_examples(examples: list[Example]) -> tuple[list[Example], int]:
    """Select benchmark rows and ignore legacy rows that have no case_id."""

    selected: list[Example] = []
    for example in examples:
        inputs = dict(example.inputs or {})
        metadata = dict(example.metadata or {})
        if _field(inputs, metadata, "case_id"):
            selected.append(example)
    return selected, len(examples) - len(selected)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def _new_manifest(dataset_name: str, dataset_id: str) -> dict[str, Any]:
    now = _utc_now()
    return {
        "version": MANIFEST_VERSION,
        "dataset_name": dataset_name,
        "dataset_id": dataset_id,
        "created_at": now,
        "updated_at": now,
        "experiment_id": None,
        "experiment_name": None,
        "experiment_url": None,
        "cases": {},
    }


def load_manifest(path: Path, dataset_name: str, dataset_id: str) -> dict[str, Any]:
    if not path.exists():
        return _new_manifest(dataset_name, dataset_id)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("version") != MANIFEST_VERSION:
        raise ValueError(f"不支持的检查点版本：{manifest.get('version')!r}")
    if str(manifest.get("dataset_id")) != dataset_id:
        raise ValueError(
            f"检查点属于另一个数据集：{manifest.get('dataset_name')!r} "
            f"({manifest.get('dataset_id')})"
        )
    return manifest


def save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = _utc_now()
    _write_json(path, manifest)


def pending_cases(
    cases: list[BenchmarkCase],
    manifest: dict[str, Any],
    batch_size: int,
) -> list[BenchmarkCase]:
    states = manifest.get("cases") or {}
    pending = [
        case
        for case in cases
        if (states.get(case.case_id) or {}).get("status") != COMPLETE_STATUS
    ]
    return pending[:batch_size]


def _saved_result(result: dict[str, Any], case: BenchmarkCase) -> dict[str, Any]:
    """Keep evaluation-relevant graph outputs without serializing internal messages."""

    keys = (
        "final_report",
        "report_completeness_check",
        "research_brief",
        "research_outline",
        "research_unit_count",
        "source_count",
        "status",
        "thread_id",
        "clarification_completed",
        "outline_confirmed",
        "hy3_calls_used",
        "hy3_call_budget",
        "hy3_call_budget_exhausted",
        "hy3_calls_by_stage",
        "tools_used",
        "error",
    )
    saved = {key: result.get(key) for key in keys if key in result}
    saved.update(
        {
            "case_id": case.case_id,
            "domain": case.domain,
            "difficulty": case.difficulty,
            "question": case.question,
            "langsmith_example_id": case.example_id,
        }
    )
    return saved


def save_case_result(
    output_dir: Path,
    case: BenchmarkCase,
    result: dict[str, Any],
) -> tuple[Path, Path]:
    case_dir = output_dir / "cases" / _safe_name(case.case_id)
    report_path = case_dir / "report.md"
    result_path = case_dir / "result.json"
    case_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(str(result.get("final_report") or ""), encoding="utf-8")
    _write_json(result_path, _saved_result(result, case))
    return report_path, result_path


def configure_langsmith(settings: Settings) -> None:
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
    if settings.langsmith_api_key:
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key


def _completed_run_output(
    client: LangSmithClient,
    experiment_id: str,
    case: BenchmarkCase,
) -> tuple[dict[str, Any], str] | None:
    """Recover a completed run if interruption happened before local checkpointing."""

    runs = client.list_runs(
        project_id=experiment_id,
        reference_example_id=case.example_id,
        is_root=True,
        error=False,
        limit=10,
    )
    for run in runs:
        outputs = dict(run.outputs or {})
        if outputs.get("status") == COMPLETE_STATUS and str(outputs.get("final_report") or ""):
            return outputs, str(run.id)
    return None


def reconcile_manifest(
    client: LangSmithClient,
    experiment_id: str,
    cases: list[BenchmarkCase],
    output_dir: Path,
    manifest: dict[str, Any],
    manifest_path: Path,
) -> int:
    recovered = 0
    states = manifest.setdefault("cases", {})
    for case in cases:
        if (states.get(case.case_id) or {}).get("status") == COMPLETE_STATUS:
            continue
        found = _completed_run_output(client, experiment_id, case)
        if found is None:
            continue
        result, run_id = found
        report_path, result_path = save_case_result(output_dir, case, result)
        states[case.case_id] = {
            "status": COMPLETE_STATUS,
            "attempts": (states.get(case.case_id) or {}).get("attempts", 1),
            "completed_at": _utc_now(),
            "thread_id": result.get("thread_id"),
            "langsmith_run_id": run_id,
            "report_path": str(report_path.resolve()),
            "result_path": str(result_path.resolve()),
            "recovered_from_langsmith": True,
        }
        recovered += 1
        save_manifest(manifest_path, manifest)
    return recovered


async def generate_case(
    case: BenchmarkCase,
    *,
    graph: Any,
    client: LangSmithClient,
    settings: Settings,
    experiment: Any | None,
    experiment_prefix: str,
    output_dir: Path,
    manifest: dict[str, Any],
    manifest_path: Path,
) -> bool:
    state = manifest.setdefault("cases", {}).setdefault(case.case_id, {})
    attempt = int(state.get("attempts") or 0) + 1
    thread_id = (
        f"benchmark-{_safe_name(case.case_id)}-"
        f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    )
    state.update(
        {
            "status": "running",
            "attempts": attempt,
            "started_at": _utc_now(),
            "thread_id": thread_id,
            "example_id": case.example_id,
            "domain": case.domain,
            "difficulty": case.difficulty,
            "error": None,
        }
    )
    save_manifest(manifest_path, manifest)
    holder: dict[str, Any] = {}

    async def target(inputs: dict[str, Any]) -> dict[str, Any]:
        question = str(inputs.get("question") or "").strip()
        if question != case.question:
            raise ValueError(f"收到非预期样本：{question[:80]!r}")
        result = await graph.ainvoke(
            {
                "thread_id": thread_id,
                "mode": "research",
                "original_request": question,
                "messages": [{"role": "user", "content": question}],
                "clarification_completed": False,
            }
        )
        holder["result"] = result
        if result.get("status") != COMPLETE_STATUS:
            raise RuntimeError(str(result.get("error") or "深度研究未完成"))
        if not str(result.get("final_report") or "").strip():
            raise RuntimeError("深度研究返回 complete，但 final_report 为空")
        return result

    try:
        results = await aevaluate(
            target,
            data=[case.example],
            experiment=experiment,
            experiment_prefix=experiment_prefix if experiment is None else None,
            description="Resumable generation of frozen deep-research benchmark reports.",
            metadata={
                "models": settings.hy3_model,
                "dataset": manifest["dataset_name"],
                "workflow": "deep-research-benchmark-generation",
            },
            max_concurrency=0,
            client=client,
            error_handling="log",
        )
        if not manifest.get("experiment_id"):
            manifest["experiment_id"] = str(results.experiment_id)
            manifest["experiment_name"] = results.experiment_name
            manifest["experiment_url"] = results.url
            save_manifest(manifest_path, manifest)
        await results.wait()
        result = holder.get("result")
        if not isinstance(result, dict) or result.get("status") != COMPLETE_STATUS:
            raise RuntimeError("LangSmith 已记录失败运行，未取得完整报告")
        report_path, result_path = save_case_result(output_dir, case, result)
        state.update(
            {
                "status": COMPLETE_STATUS,
                "completed_at": _utc_now(),
                "report_path": str(report_path.resolve()),
                "result_path": str(result_path.resolve()),
                "error": None,
            }
        )
        save_manifest(manifest_path, manifest)
        return True
    except Exception as exc:
        state.update(
            {
                "status": "failed",
                "failed_at": _utc_now(),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        save_manifest(manifest_path, manifest)
        return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="从 LangSmith 数据集断点续跑生成深度研究报告；默认每次最多5篇",
    )
    parser.add_argument("dataset", help="LangSmith 数据集名称，例如 bird")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="本次最多尝试的样本数，默认5",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="检查点和本地报告目录；默认 outputs/benchmark_generation/<dataset>",
    )
    parser.add_argument(
        "--experiment-prefix",
        default=None,
        help="首次运行时创建的 LangSmith experiment 名称前缀",
    )
    parser.add_argument(
        "--case-id",
        default=None,
        help="只处理指定 case_id；默认按检查点继续下一批",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重跑 --case-id 指定的样本，即使检查点已标记为完成",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅显示下一批样本，不调用模型",
    )
    return parser


async def main() -> int:
    args = build_parser().parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size 必须大于等于1")
    if args.force and not args.case_id:
        raise SystemExit("--force 必须与 --case-id 一起使用，防止误重跑整批样本")
    settings = get_settings().model_copy(
        update={
            "allow_clarification": False,
            "require_outline_confirmation": False,
        }
    )
    configure_langsmith(settings)
    client = LangSmithClient(
        api_url=settings.langsmith_endpoint,
        api_key=settings.langsmith_api_key or None,
    )
    dataset = client.read_dataset(dataset_name=args.dataset)
    all_examples = list(client.list_examples(dataset_id=dataset.id))
    if not all_examples:
        raise SystemExit(f"数据集 {args.dataset!r} 没有样本")
    examples, ignored_count = select_labeled_examples(all_examples)
    if not examples:
        raise SystemExit(
            f"数据集 {args.dataset!r} 中没有带 case_id 的基准样本。"
            f"当前共有 {len(all_examples)} 条旧样本，均未处理。"
        )
    cases = normalize_examples(examples)
    requested_cases = cases
    if args.case_id:
        requested_cases = [case for case in cases if case.case_id == args.case_id]
        if not requested_cases:
            raise SystemExit(
                f"数据集 {args.dataset!r} 中不存在 case_id={args.case_id!r}"
            )
    output_dir = args.output_dir or (
        Path("outputs") / "benchmark_generation" / _safe_name(args.dataset)
    )
    output_dir = output_dir.resolve()
    manifest_path = output_dir / "manifest.json"
    manifest = load_manifest(manifest_path, str(dataset.name), str(dataset.id))

    experiment_id = str(manifest.get("experiment_id") or "")
    experiment = None
    if experiment_id:
        experiment = client.read_project(project_id=experiment_id)
        recovered = reconcile_manifest(
            client,
            experiment_id,
            cases,
            output_dir,
            manifest,
            manifest_path,
        )
        if recovered:
            print(f"已从 LangSmith 恢复 {recovered} 个中断前完成的样本。")

    batch = (
        requested_cases[:1]
        if args.force
        else pending_cases(requested_cases, manifest, args.batch_size)
    )
    completed_total = sum(
        (manifest.get("cases", {}).get(case.case_id) or {}).get("status") == COMPLETE_STATUS
        for case in cases
    )
    print(
        f"数据集={dataset.name}，基准样本={len(cases)}，忽略旧样本={ignored_count}，"
        f"已完成={completed_total}，本批={len(batch)}；不会运行质量评分器。"
    )
    if not batch:
        print("全部样本均已生成，无需继续。")
        if manifest.get("experiment_name"):
            print(f"LangSmith experiment：{manifest['experiment_name']}")
        return 0
    print("本批 case_id：" + ", ".join(case.case_id for case in batch))
    if args.dry_run:
        return 0

    if not settings.hy3_api_key:
        raise SystemExit("未配置 HY3_API_KEY，无法生成报告")

    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = ArtifactStore(settings.database_url)
    graph = build_deep_research_graph(settings, artifacts)
    successes = 0
    failures = 0
    prefix = args.experiment_prefix or f"{_safe_name(str(dataset.name))}-reports"
    try:
        for index, case in enumerate(batch, start=1):
            print(f"\n[{index}/{len(batch)}] 正在生成 {case.case_id}：{case.question[:80]}")
            current_experiment = experiment
            if manifest.get("experiment_id"):
                current_experiment = client.read_project(
                    project_id=str(manifest["experiment_id"])
                )
            success = await generate_case(
                case,
                graph=graph,
                client=client,
                settings=settings,
                experiment=current_experiment,
                experiment_prefix=prefix,
                output_dir=output_dir,
                manifest=manifest,
                manifest_path=manifest_path,
            )
            if success:
                successes += 1
                print(f"[完成] {case.case_id}")
            else:
                failures += 1
                error = manifest["cases"][case.case_id].get("error")
                print(f"[失败] {case.case_id}：{error}")
    finally:
        artifacts.close()

    remaining = len(cases) - sum(
        (manifest.get("cases", {}).get(case.case_id) or {}).get("status") == COMPLETE_STATUS
        for case in cases
    )
    print(
        f"\n本批结束：成功={successes}，失败={failures}，剩余={remaining}。"
        f"\n检查点：{manifest_path}"
    )
    if manifest.get("experiment_name"):
        print(f"LangSmith experiment：{manifest['experiment_name']}")
    if manifest.get("experiment_url"):
        print(f"实验地址：{manifest['experiment_url']}")
    print("检查本批报告后，重新执行同一命令即可继续下一批。")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
