"""Run one bounded real-provider smoke test of the deep-research graph."""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime

from deep_research_assistant.artifact_store import ArtifactStore
from deep_research_assistant.config import Settings
from deep_research_assistant.deep_research_graph import build_deep_research_graph

SAMPLE_REQUEST = """调研中国普通家庭购买洗碗机是否值得。

研究时间范围为 2022—2026 年，面向准备装修或更换家电的普通消费者。重点比较洗碗机与手洗
在用水量、用电量、清洁效果、卫生安全、时间成本、购买和维护成本方面的差异，并分别讨论
1—2 人家庭、3—4 人家庭以及经常做饭的家庭。

优先使用政府或行业标准、消费者协会测评、公开实验研究、学术论文和主流电商能够核实的价格
信息。品牌官网只能用于说明产品参数，不能单独作为“洗碗机更好”的依据。不同来源结论冲突时，
需要解释测试条件、餐具数量、清洗方式和地区水电价格造成的差异。

最终生成一篇 2600—3000 个有效正文字符的中文完整调研报告，不要使用 S1、S2 或“研究员一”
之类的内部编号。先给出核心结论，再展开证据比较，最后按不同家庭情况给出购买建议。正文的重要
数字附近必须有来源，末尾提供去重后的主要来源，并说明主要来源的资料类型。"""


async def main() -> None:
    settings = Settings(
        allow_clarification=False,
        require_outline_confirmation=False,
        max_concurrent_research_units=3,
        max_supervisor_iterations=3,
        max_researcher_iterations=8,
        max_search_tool_calls=4,
        max_hy3_calls_per_research=65,
        tavily_max_results=5,
        tavily_max_queries=3,
        tavily_max_summarized_results=5,
        researcher_tool_result_max_characters=20_000,
        researcher_context_max_characters=60_000,
        final_report_context_max_characters=120_000,
    )
    if settings.langsmith_tracing:
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
        os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
        if settings.langsmith_api_key:
            os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key

    thread_id = "smoke-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    artifacts = ArtifactStore(settings.database_url)
    graph = build_deep_research_graph(settings, artifacts)
    try:
        result = await graph.ainvoke(
            {
                "thread_id": thread_id,
                "mode": "research",
                "original_request": SAMPLE_REQUEST,
                "messages": [{"role": "user", "content": SAMPLE_REQUEST}],
                "clarification_completed": False,
            }
        )
        report = str(result.get("final_report") or "")
        summary = {
            "thread_id": thread_id,
            "status": result.get("status"),
            "research_unit_count": result.get("research_unit_count"),
            "source_count": result.get("source_count"),
            "tools_used": result.get("tools_used"),
            "hy3_calls_used": result.get("hy3_calls_used"),
            "hy3_call_budget": result.get("hy3_call_budget"),
            "hy3_call_budget_exhausted": result.get("hy3_call_budget_exhausted"),
            "hy3_calls_by_stage": result.get("hy3_calls_by_stage"),
            "notes_cleared_after_report": not result.get("notes"),
            "report_characters": len(report),
            "contains_internal_section_ids": any(
                marker in report for marker in ("S1", "S2", "S3")
            ),
            "error": result.get("error"),
        }
        print("SMOKE_SUMMARY=" + json.dumps(summary, ensure_ascii=False, indent=2))
        print("REPORT_BEGIN")
        print(report)
        print("REPORT_END")
    finally:
        artifacts.close()


if __name__ == "__main__":
    asyncio.run(main())
