"""Calibrate rule and LLM evaluators on prewritten LangSmith dataset reports."""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

from langsmith import Client as LangSmithClient
from langsmith import aevaluate
from openai import AsyncOpenAI

from deep_research_assistant.config import get_settings
from scripts.run_langsmith_evaluation import (
    citation_quality,
    llm_quality_evaluator,
    outline_structure_coverage,
    report_format,
    research_process,
    run_completion,
)

REQUIRED_INPUT_ALIASES = {
    "question": ("question", "input"),
    "outline": ("research_brief", "outline", "evaluation_outline"),
    "report": ("final_report", "report"),
}
PROCESS_FIELDS = ("research_unit_count", "source_count", "tools_used")
CALIBRATION_RULE_WEIGHTS = {
    "run_completion": 0.20,
    "citation_quality": 0.30,
    "outline_structure_coverage": 0.15,
    "report_format": 0.15,
}


def _first_present(inputs: dict[str, Any], aliases: tuple[str, ...]) -> Any:
    for alias in aliases:
        value = inputs.get(alias)
        if value is not None and value != "":
            return value
    return None


def _parse_outline(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        return {"sections": value}
    if not isinstance(value, str):
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def validate_calibration_inputs(
    inputs: Any,
    *,
    example_id: str = "unknown",
) -> dict[str, Any]:
    """Validate and normalize one calibration dataset example."""

    if not isinstance(inputs, dict):
        raise ValueError(f"样本 {example_id} 的 inputs 必须是对象")

    resolved = {
        name: _first_present(inputs, aliases)
        for name, aliases in REQUIRED_INPUT_ALIASES.items()
    }
    missing = [name for name, value in resolved.items() if value is None]
    if missing:
        raise ValueError(
            f"样本 {example_id} 缺少校准字段：{', '.join(missing)}；"
            "需要 question、research_brief/outline、final_report/report"
        )
    if not str(resolved["question"]).strip():
        raise ValueError(f"样本 {example_id} 的 question 不能为空")
    if not str(resolved["report"]).strip():
        raise ValueError(f"样本 {example_id} 的 final_report/report 不能为空")
    return inputs


def calibration_target(inputs: dict[str, Any]) -> dict[str, Any]:
    """Return a prewritten report without invoking the research application."""

    validate_calibration_inputs(inputs)
    question = _first_present(inputs, REQUIRED_INPUT_ALIASES["question"])
    outline_value = _first_present(inputs, REQUIRED_INPUT_ALIASES["outline"])
    report = _first_present(inputs, REQUIRED_INPUT_ALIASES["report"])
    parsed_outline = _parse_outline(outline_value)
    research_brief = (
        outline_value
        if isinstance(outline_value, str)
        else json.dumps(outline_value, ensure_ascii=False)
    )
    outputs: dict[str, Any] = {
        "original_request": str(question).strip(),
        "research_brief": research_brief,
        "evaluation_outline": parsed_outline,
        "final_report": str(report).strip(),
        "status": str(inputs.get("status") or "complete"),
        "error": str(inputs.get("error") or ""),
    }
    for field in PROCESS_FIELDS:
        if field in inputs:
            outputs[field] = inputs[field]
    return outputs


async def async_calibration_target(inputs: dict[str, Any]) -> dict[str, Any]:
    """Async adapter required by LangSmith's ``aevaluate`` API."""

    return calibration_target(inputs)


def calibration_rule_evaluator(run: Any, example: Any) -> list[dict[str, Any]]:
    """Run content rules and include process rules only when evidence is supplied."""

    metrics = [
        run_completion(run, example),
        citation_quality(run, example),
        outline_structure_coverage(run, example),
        report_format(run, example),
    ]
    outputs = getattr(run, "outputs", None) or {}
    has_process_evidence = all(field in outputs for field in PROCESS_FIELDS)
    weights = dict(CALIBRATION_RULE_WEIGHTS)
    if has_process_evidence:
        metrics.insert(1, research_process(run, example))
        weights["research_process"] = 0.20

    scores = {str(metric["key"]): float(metric["score"]) for metric in metrics}
    available_weight = sum(weights.values())
    overall = round(
        sum(scores[key] * weight for key, weight in weights.items())
        / available_weight,
        4,
    )
    missing_note = (
        ""
        if has_process_evidence
        else "；数据集未提供研究过程字段，research_process未评分且已从权重中排除"
    )
    metrics.append(
        {
            "key": "rule_overall",
            "score": overall,
            "comment": (
                "校准规则总分按当前可用规则权重重新归一化"
                f"{missing_note}。"
            ),
        }
    )
    return metrics


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="评测包含问题、大纲和预制最终报告的 LangSmith 校准数据集",
    )
    parser.add_argument("dataset", help="LangSmith 校准数据集名称或 UUID")
    parser.add_argument(
        "--prefix",
        default="evaluator-calibration",
        help="新建校准实验的名称前缀",
    )
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="每个样本重复评测次数；一致性验证建议设为3",
    )
    parser.add_argument(
        "--skip-llm-judge",
        action="store_true",
        help="只运行确定性规则评分器，不调用 DeepSeek",
    )
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("--concurrency 必须大于等于1")
    if args.repetitions < 1:
        parser.error("--repetitions 必须大于等于1")

    settings = get_settings()
    if not args.skip_llm_judge and not settings.deepseek_api_key:
        parser.error(
            "未配置 DEEPSEEK_API_KEY。请写入 .env，或使用 --skip-llm-judge "
            "只运行规则评分。"
        )
    langsmith_client = LangSmithClient(
        api_url=settings.langsmith_endpoint,
        api_key=settings.langsmith_api_key or None,
    )
    examples = list(
        langsmith_client.list_examples(
            dataset_name=args.dataset,
        )
    )
    if not examples:
        parser.error(f"校准数据集 {args.dataset!r} 没有样本")
    for example in examples:
        validate_calibration_inputs(
            example.inputs,
            example_id=str(example.id),
        )

    evaluators: list[Any] = [calibration_rule_evaluator]
    judge_client: AsyncOpenAI | None = None
    if not args.skip_llm_judge:
        judge_client = AsyncOpenAI(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            timeout=settings.evaluation_judge_timeout_seconds,
            max_retries=settings.evaluation_judge_max_retries,
        )

        async def quality_judge(run: Any, example: Any) -> list[dict[str, Any]]:
            assert judge_client is not None
            return await llm_quality_evaluator(
                run,
                example,
                client=judge_client,
                model=settings.evaluation_judge_model,
                max_tokens=settings.evaluation_judge_max_tokens,
            )

        evaluators.append(quality_judge)

    judge_calls = 0 if args.skip_llm_judge else len(examples) * args.repetitions
    print(
        f"校准样本数={len(examples)}，重复次数={args.repetitions}，"
        f"预计LLM Judge调用数={judge_calls}；不会生成研究报告。"
    )
    try:
        results = await aevaluate(
            async_calibration_target,
            data=examples,
            evaluators=evaluators,
            experiment_prefix=args.prefix,
            max_concurrency=args.concurrency,
            num_repetitions=args.repetitions,
            metadata={
                "purpose": "evaluator-calibration",
                "judge_model": (
                    settings.evaluation_judge_model
                    if not args.skip_llm_judge
                    else "disabled"
                ),
            },
            client=langsmith_client,
        )
        await results.wait()
    finally:
        if judge_client is not None:
            await judge_client.close()


if __name__ == "__main__":
    asyncio.run(main())
