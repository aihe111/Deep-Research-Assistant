"""Command-line entry point for Deep-Research-Assistant."""

import re
import time
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.json import JSON
from rich.table import Table
from sqlalchemy.engine import make_url

from deep_research_assistant.artifact_store import ArtifactStore
from deep_research_assistant.config import get_settings
from deep_research_assistant.langgraph_client import LangGraphServerClient
from deep_research_assistant.literature_workflow import LiteratureWorkflow
from deep_research_assistant.models import LiteratureSearchResult
from deep_research_assistant.report_workflow import ReportWorkflow
from deep_research_assistant.workflow import PlanningWorkflow, ResearchPlan

app = typer.Typer(help="Deep-Research-Assistant 通用深度研究助手")
console = Console()


@app.command("web")
def run_web(
    host: Annotated[str, typer.Option("--host", help="Web 服务监听地址")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", help="Web 服务端口", min=1, max=65535)] = 8000,
    reload: Annotated[bool, typer.Option("--reload", help="开发时自动重载代码")] = False,
) -> None:
    """Start the browser-based research assistant."""

    import uvicorn

    console.print(f"[green]Deep-Research-Assistant Web：http://{host}:{port}[/green]")
    uvicorn.run(
        "deep_research_assistant.web_app:app",
        host=host,
        port=port,
        reload=reload,
    )


@app.command()
def doctor() -> None:
    """Show non-sensitive Hy3 configuration before the first API call."""

    settings = get_settings()
    table = Table(title="Deep-Research-Assistant 配置检查")
    table.add_column("配置项")
    table.add_column("当前值")
    table.add_row("API Base URL", settings.hy3_base_url)
    table.add_row("Model", settings.hy3_model)
    table.add_row("API Key", "已配置" if settings.hy3_api_key else "未配置")
    table.add_row("Timeout", f"{settings.hy3_timeout_seconds:g} 秒")
    table.add_row("Hy3 transient retries", str(settings.hy3_max_retries))
    table.add_row("Reasoning effort", settings.hy3_reasoning_effort)
    table.add_row("OpenAlex API Key", "已配置" if settings.openalex_api_key else "未配置")
    table.add_row("OpenAlex per query", str(settings.openalex_per_query))
    table.add_row(
        "OpenAlex full text",
        "已启用" if settings.openalex_fetch_full_text and settings.openalex_api_key else "未启用",
    )
    table.add_row(
        "Tavily",
        (
            f"已配置（{settings.tavily_provider}）"
            if settings.tavily_active_api_key
            else "未启用"
        ),
    )
    table.add_row("Tavily raw content", f"最多 {settings.tavily_max_content_length} 字符/网页")
    table.add_row("MCP servers", str(len(settings.mcp_servers)))
    table.add_row("Parallel researchers", str(settings.max_concurrent_research_units))
    table.add_row(
        "Hy3 calls / research",
        (
            str(settings.max_hy3_calls_per_research)
            if settings.max_hy3_calls_per_research > 0
            else "不限（按阶段统计）"
        ),
    )
    table.add_row(
        "Tavily full-page summaries",
        str(settings.tavily_max_summarized_results),
    )
    table.add_row("LangSmith tracing", "已启用" if settings.langsmith_tracing else "未启用")
    database_url = make_url(settings.database_url)
    table.add_row("Database", database_url.render_as_string(hide_password=True))
    table.add_row("LangGraph Server", settings.langgraph_api_url)
    table.add_row("LangGraph assistant", settings.langgraph_assistant_id)
    console.print(table)


def _safe_thread_name(thread_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", thread_id).strip("_") or "research"


def _interrupt_payload(result: dict[str, Any]) -> dict[str, Any] | None:
    interrupts = result.get("interrupts") or result.get("__interrupt__") or ()
    if not interrupts:
        return None
    first = interrupts[0]
    value = first.get("value") if isinstance(first, dict) else getattr(first, "value", None)
    return value if isinstance(value, dict) else None


def _resume_payload(interrupt_payload: dict[str, Any]) -> dict[str, Any]:
    kind = interrupt_payload.get("kind")
    if kind == "clarification":
        questions = interrupt_payload.get("questions") or []
        current_round = interrupt_payload.get("round", 1)
        max_rounds = interrupt_payload.get("max_rounds", 3)
        console.print(
            f"\n[bold yellow]需要先补充以下调研参数（{current_round}/{max_rounds}）：[/bold yellow]"
        )
        answers: list[str] = []
        for question in questions:
            answer = typer.prompt(str(question)).strip()
            while not answer:
                console.print("[red]该项不能为空，请重新输入。[/red]")
                answer = typer.prompt(str(question)).strip()
            answers.append(answer)
        return {"answers": answers}

    raise RuntimeError(f"无法识别的人工中断类型：{kind}")


def _export_deep_research_report(
    database_url: str,
    thread_id: str,
    output_root: Path,
) -> Path:
    output_dir = output_root / _safe_thread_name(thread_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = ArtifactStore(database_url)
    try:
        (output_dir / "research_report.md").write_text(
            artifacts.get_text(thread_id, "research_report"),
            encoding="utf-8",
        )
    finally:
        artifacts.close()
    return output_dir


@app.command("agent")
def run_agent(
    request: Annotated[
        str | None,
        typer.Argument(help="自然语言调研需求；不填写时会在终端中提示输入"),
    ] = None,
    thread_id: Annotated[
        str | None,
        typer.Option("--thread-id", help="LangGraph Thread UUID；重复使用可接续中断任务"),
    ] = None,
    output_root: Annotated[
        Path,
        typer.Option("--output-root", help="各任务输出目录"),
    ] = Path("outputs/runs"),
) -> None:
    """Submit the general deep-research graph to LangGraph Server and wait for its Run."""

    settings = get_settings()
    if not settings.hy3_api_key:
        console.print("[bold red]HY3_API_KEY 未配置，无法启动 Agent。[/bold red]")
        raise typer.Exit(code=2)
    client = LangGraphServerClient(settings)
    try:
        thread = client.create_thread(thread_id)
        resolved_thread_id = str(thread["thread_id"])
        state = client.get_state(resolved_thread_id)
        pending = _interrupt_payload(state)
        run: dict[str, Any] | None = None
        if pending:
            run = client.create_run(
                resolved_thread_id,
                resume=_resume_payload(pending),
            )
        else:
            initial_request = (request or typer.prompt("请输入你的调研问题")).strip()
            if len(initial_request) < 4:
                console.print("[bold red]调研需求过短，请至少描述主题和目标。[/bold red]")
                raise typer.Exit(code=2)
            run = client.create_run(
                resolved_thread_id,
                input={
                    "thread_id": resolved_thread_id,
                    "mode": "research",
                    "original_request": initial_request,
                    "messages": [{"role": "user", "content": initial_request}],
                    "clarification_completed": False,
                },
            )

        while run is not None:
            run_id = str(run["run_id"])
            while True:
                run_status = str(client.get_run(resolved_thread_id, run_id).get("status"))
                if run_status in {"pending", "running"}:
                    time.sleep(0.5)
                    continue
                break
            if run_status == "interrupted":
                pending = _interrupt_payload(client.get_state(resolved_thread_id))
                if pending is None:
                    raise RuntimeError("Run 已中断，但 Thread 未返回中断内容")
                run = client.create_run(
                    resolved_thread_id,
                    resume=_resume_payload(pending),
                )
                continue
            if run_status != "success":
                console.print(f"[bold red]LangGraph Run 状态：{run_status}[/bold red]")
                console.print(f"Thread ID：{resolved_thread_id}")
                raise typer.Exit(code=4)
            run = None

        values = client.get_state(resolved_thread_id).get("values") or {}
        if values.get("status") != "complete":
            raise RuntimeError(str(values.get("error") or "深度研究未完成"))
        output_dir = _export_deep_research_report(
            settings.database_url,
            resolved_thread_id,
            output_root,
        )
        console.print("\n[bold green]调研报告生成完成。[/bold green]")
        console.print(f"Thread ID：{resolved_thread_id}")
        console.print(
            f"研究单元：{values.get('research_unit_count', 0)}；"
            f"来源链接：{values.get('source_count', 0)}；"
            f"工具：{', '.join(values.get('tools_used', [])) or '无'}"
        )
        console.print(f"[green]全部产物已保存至：{output_dir.resolve()}[/green]")
    finally:
        client.close()


@app.command("plan")
def create_plan(
    request: Annotated[str, typer.Argument(help="自然语言调研需求")],
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="结构化调研计划保存路径",
        ),
    ] = Path("outputs/research_plan.json"),
) -> None:
    """Analyze a request and generate an editable research outline."""

    settings = get_settings()
    if not settings.hy3_api_key:
        console.print(
            "[bold red]HY3_API_KEY 未配置。请先复制 .env.example 为 .env 并填写密钥。[/bold red]"
        )
        raise typer.Exit(code=2)

    workflow = PlanningWorkflow()
    enriched_request = request.strip()
    intent = None
    for round_number in range(1, 4):
        with console.status("正在识别调研意图..."):
            intent = workflow.analyze(enriched_request)
        if not intent.clarification_questions:
            break

        console.print(f"\n[bold yellow]还需要补充以下信息（第 {round_number} 轮）：[/bold yellow]")
        answers = []
        for question in intent.clarification_questions:
            answer = typer.prompt(question).strip()
            while not answer:
                console.print("[red]该项不能为空，请重新输入。[/red]")
                answer = typer.prompt(question).strip()
            answers.append((question, answer))
        supplement = "\n".join(f"- {question} 回答：{answer}" for question, answer in answers)
        enriched_request += "\n\n用户补充信息：\n" + supplement
    else:
        console.print("[bold red]经过 3 轮补充后意图仍不完整，请重新描述调研需求。[/bold red]")
        raise typer.Exit(code=3)

    if intent is None or intent.clarification_questions:
        raise typer.Exit(code=3)
    with console.status("意图已确认，正在生成调研大纲..."):
        plan = workflow.build(enriched_request, intent)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    console.print(JSON(plan.model_dump_json()))
    console.print(f"[green]调研计划已保存至：{output.resolve()}[/green]")


@app.command()
def status() -> None:
    """Show the current implementation milestone."""

    console.print("[bold green]通用 Deep Research 主流程已就绪。[/bold green]")
    console.print("当前节点：澄清 → 研究大纲 → Manager/并行 Researcher → 最终报告。")


@app.command("search")
def search_literature(
    plan_file: Annotated[Path, typer.Argument(help="M1 生成的 research_plan.json")],
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="文献检索结果保存路径",
        ),
    ] = Path("outputs/literature_results.json"),
) -> None:
    """Generate queries, retrieve OpenAlex works, deduplicate, and rank them."""

    if not plan_file.is_file():
        console.print(f"[bold red]调研计划文件不存在：{plan_file}[/bold red]")
        raise typer.Exit(code=2)

    settings = get_settings()
    if not settings.hy3_api_key:
        console.print("[bold red]HY3_API_KEY 未配置，无法生成检索计划。[/bold red]")
        raise typer.Exit(code=2)
    if not settings.openalex_api_key:
        console.print(
            "[yellow]OPENALEX_API_KEY 未配置，将尝试匿名请求；正式使用建议配置免费 Key。[/yellow]"
        )

    plan = ResearchPlan.model_validate_json(plan_file.read_text(encoding="utf-8"))
    with console.status("正在生成检索词并检索 OpenAlex..."):
        result = LiteratureWorkflow().run(plan)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(result.model_dump_json(indent=2), encoding="utf-8")

    table = Table(title="OpenAlex 文献检索结果")
    table.add_column("排名", justify="right")
    table.add_column("年份")
    table.add_column("分数")
    table.add_column("引用")
    table.add_column("标题", overflow="fold")
    for index, work in enumerate(result.works, start=1):
        table.add_row(
            str(index),
            str(work.publication_year or "-"),
            f"{work.relevance_score:.3f}",
            str(work.cited_by_count),
            work.title,
        )
    console.print(table)
    console.print(
        f"检索 {result.retrieved_count} 条，去重后 {result.deduplicated_count} 条，"
        f"保留前 {len(result.works)} 条。"
    )
    console.print(f"[green]结果已保存至：{output.resolve()}[/green]")


@app.command("report")
def create_report(
    literature_file: Annotated[
        Path,
        typer.Argument(help="M2 生成的 literature_results.json"),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Markdown 报告保存路径"),
    ] = Path("outputs/research_report.md"),
    evidence_output: Annotated[
        Path,
        typer.Option("--evidence-output", help="证据卡 JSON 保存路径"),
    ] = Path("outputs/evidence_cards.json"),
) -> None:
    """Build evidence cards and generate a citation-validated Markdown report."""

    if not literature_file.is_file():
        console.print(f"[bold red]文献结果文件不存在：{literature_file}[/bold red]")
        raise typer.Exit(code=2)
    if not get_settings().hy3_api_key:
        console.print("[bold red]HY3_API_KEY 未配置，无法生成证据卡和报告。[/bold red]")
        raise typer.Exit(code=2)

    search_result = LiteratureSearchResult.model_validate_json(
        literature_file.read_text(encoding="utf-8")
    )
    with console.status("正在构建证据卡并生成带引用的报告..."):
        artifacts = ReportWorkflow().run(search_result)

    evidence_output.parent.mkdir(parents=True, exist_ok=True)
    evidence_output.write_text(artifacts.evidence.model_dump_json(indent=2), encoding="utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(artifacts.markdown, encoding="utf-8")

    console.print(f"生成证据卡：{len(artifacts.evidence.cards)} 张")
    console.print(f"[green]证据卡已保存至：{evidence_output.resolve()}[/green]")
    console.print(f"[green]Markdown 报告已保存至：{output.resolve()}[/green]")


if __name__ == "__main__":
    app()
