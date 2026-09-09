"""FastAPI application for the Deep-Research-Assistant web interface."""

import asyncio
import json
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langgraph_sdk.errors import NotFoundError
from pydantic import BaseModel, Field

from deep_research_assistant.artifact_store import ArtifactNotFoundError, ArtifactStore
from deep_research_assistant.config import get_settings
from deep_research_assistant.conversation_store import ConversationStore
from deep_research_assistant.langgraph_client import LangGraphServerClient
from deep_research_assistant.research_execution import record_run_failure, record_run_state

STATIC_DIRECTORY = Path(__file__).with_name("web_static")


class MessageRequest(BaseModel):
    content: str = Field(min_length=1, max_length=8000)


class ResumeRequest(BaseModel):
    kind: Literal["clarification", "outline_confirmation"]
    answers: list[str] = Field(default_factory=list, max_length=8)
    action: Literal["approve", "revise"] | None = None
    feedback: str = Field(default="", max_length=2000)


def _open_store():
    store = ConversationStore(get_settings().database_url)
    try:
        yield store
    finally:
        store.close()


StoreDependency = Annotated[ConversationStore, Depends(_open_store)]


def _open_langgraph_client():
    client = LangGraphServerClient(get_settings())
    try:
        yield client
    finally:
        client.close()


LangGraphDependency = Annotated[LangGraphServerClient, Depends(_open_langgraph_client)]


def _conversation_payload(store: ConversationStore, conversation_id: str) -> dict[str, Any]:
    conversation = store.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return {
        "conversation": asdict(conversation),
        "messages": [asdict(message) for message in store.list_messages(conversation_id)],
        "events": [asdict(event) for event in store.list_events(conversation_id)],
    }


def _sse_message(event: str, data: dict[str, Any], *, event_id: int | None = None) -> str:
    lines = []
    if event_id is not None:
        lines.append(f"id: {event_id}")
    lines.append(f"event: {event}")
    lines.append("data: " + json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    return "\n".join(lines) + "\n\n"


def _sync_active_run(
    store: ConversationStore,
    conversation_id: str,
    client: LangGraphServerClient,
) -> None:
    conversation = store.get_conversation(conversation_id)
    if conversation is None or not conversation.active_run_id:
        return
    run_id = conversation.active_run_id
    run = client.get_run(conversation_id, run_id)
    run_status = str(run.get("status") or "error")
    if run_status == "pending":
        return
    if run_status == "running":
        if conversation.status != "responding":
            store.update_conversation(conversation_id, status="running")
        return
    if run_status in {"success", "interrupted"}:
        record_run_state(store, conversation_id, run_id, client.get_state(conversation_id))
        return
    record_run_failure(store, conversation_id, run_id, run_status)


def _submit_run(
    store: ConversationStore,
    client: LangGraphServerClient,
    conversation_id: str,
    *,
    input: dict[str, Any] | None = None,
    resume: dict[str, Any] | None = None,
    conversation_status: str = "queued",
) -> dict[str, Any]:
    settings = get_settings()
    if not settings.hy3_api_key:
        raise HTTPException(status_code=503, detail="HY3_API_KEY 未配置")
    try:
        run = client.create_run(conversation_id, input=input, resume=resume)
    except Exception as exc:
        error = f"LangGraph Server 无法创建 Run：{exc}"
        failure_status = "complete" if conversation_status == "responding" else "failed"
        store.update_conversation(conversation_id, status=failure_status, clear_active_run=True)
        store.add_message(
            conversation_id,
            "assistant",
            error,
            kind="error",
        )
        raise HTTPException(status_code=503, detail=error) from exc
    run_id = str(run["run_id"])
    store.update_conversation(
        conversation_id,
        status=conversation_status,
        active_run_id=run_id,
    )
    return {
        "conversation_id": conversation_id,
        "run_id": run_id,
        "status": conversation_status,
    }


def create_app() -> FastAPI:
    app = FastAPI(title="Deep-Research-Assistant", version="0.3.0")
    app.mount("/assets", StaticFiles(directory=STATIC_DIRECTORY), name="assets")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIRECTORY / "index.html")

    @app.get("/api/health")
    def health() -> dict[str, str]:
        settings = get_settings()
        return {
            "status": "ok",
            "hy3": "configured" if settings.hy3_api_key else "missing",
            "openalex": "enabled" if settings.enable_openalex else "disabled",
            "openalex_full_text": (
                "configured"
                if settings.openalex_fetch_full_text and settings.openalex_api_key
                else "disabled"
            ),
            "tavily": "configured" if settings.tavily_active_api_key else "disabled",
            "mcp_servers": str(len(settings.mcp_servers)),
            "langsmith": "enabled" if settings.langsmith_tracing else "disabled",
            "hy3_call_budget": (
                str(settings.max_hy3_calls_per_research)
                if settings.max_hy3_calls_per_research > 0
                else "unlimited"
            ),
            "tavily_result_policy": "all_unique",
            "tavily_fulltext_summaries": str(
                settings.tavily_max_summarized_results
            ),
        }

    @app.post("/api/conversations", status_code=201)
    def create_conversation(
        store: StoreDependency,
        langgraph: LangGraphDependency,
    ) -> dict[str, Any]:
        conversation_id: str | None = None
        try:
            thread = langgraph.create_thread()
            conversation_id = str(thread["thread_id"])
            store.create_conversation(conversation_id)
        except Exception as exc:
            if conversation_id is not None:
                try:
                    langgraph.delete_thread(conversation_id)
                except Exception:
                    pass
            raise HTTPException(
                status_code=503,
                detail=f"LangGraph Server 无法创建 Thread：{exc}",
            ) from exc
        return _conversation_payload(store, conversation_id)

    @app.get("/api/conversations")
    def list_conversations(store: StoreDependency) -> list[dict[str, Any]]:
        return [asdict(item) for item in store.list_conversations()]

    @app.get("/api/conversations/{conversation_id}")
    def get_conversation(
        conversation_id: str,
        store: StoreDependency,
        langgraph: LangGraphDependency,
    ) -> dict[str, Any]:
        try:
            _sync_active_run(store, conversation_id, langgraph)
        except NotFoundError:
            conversation = store.get_conversation(conversation_id)
            if conversation is not None and conversation.active_run_id:
                record_run_failure(
                    store,
                    conversation_id,
                    conversation.active_run_id,
                    "thread_not_found",
                )
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"无法读取 LangGraph Run 状态：{exc}",
            ) from exc
        return _conversation_payload(store, conversation_id)

    @app.delete("/api/conversations/{conversation_id}", status_code=204)
    def delete_conversation(
        conversation_id: str,
        store: StoreDependency,
        langgraph: LangGraphDependency,
    ) -> Response:
        conversation = store.get_conversation(conversation_id)
        if conversation is None:
            raise HTTPException(status_code=404, detail="会话不存在")
        if conversation.status in {"queued", "running", "responding"}:
            raise HTTPException(status_code=409, detail="Agent 正在处理该会话，请完成后再删除")
        try:
            langgraph.delete_thread(conversation_id)
        except NotFoundError:
            # Local in-memory LangGraph development may have been restarted.
            pass
        store.delete_conversation(conversation_id)
        return Response(status_code=204)

    @app.post("/api/conversations/{conversation_id}/messages", status_code=202)
    def send_message(
        conversation_id: str,
        request: MessageRequest,
        store: StoreDependency,
        langgraph: LangGraphDependency,
    ) -> dict[str, Any]:
        conversation = store.get_conversation(conversation_id)
        if conversation is None:
            raise HTTPException(status_code=404, detail="会话不存在")
        if conversation.status == "complete":
            content = request.content.strip()
            store.add_message(conversation_id, "user", content)
            return _submit_run(
                store,
                langgraph,
                conversation_id,
                input={
                    "thread_id": conversation_id,
                    "mode": "followup",
                },
                conversation_status="responding",
            )
        if conversation.status != "new":
            raise HTTPException(status_code=409, detail="当前会话已开始，请按页面提示继续")

        content = request.content.strip()
        if len(content) < 4:
            raise HTTPException(status_code=422, detail="请至少描述调研主题和目标")
        title = content.splitlines()[0][:28]
        if len(content.splitlines()[0]) > 28:
            title += "…"
        store.update_conversation(conversation_id, title=title)
        store.add_message(conversation_id, "user", content)
        initial_state = {
            "thread_id": conversation_id,
            "mode": "research",
            "original_request": content,
            "messages": [{"role": "user", "content": content}],
            "clarification_completed": False,
        }
        return _submit_run(
            store,
            langgraph,
            conversation_id,
            input=initial_state,
        )

    @app.post("/api/conversations/{conversation_id}/resume", status_code=202)
    def resume_conversation(
        conversation_id: str,
        request: ResumeRequest,
        store: StoreDependency,
        langgraph: LangGraphDependency,
    ) -> dict[str, Any]:
        conversation = store.get_conversation(conversation_id)
        if conversation is None:
            raise HTTPException(status_code=404, detail="会话不存在")

        if request.kind == "clarification":
            if conversation.status != "waiting_for_clarification":
                raise HTTPException(status_code=409, detail="当前会话不在参数补充阶段")
            answers = [answer.strip() for answer in request.answers]
            if not answers or any(not answer for answer in answers):
                raise HTTPException(status_code=422, detail="请回答全部问题")
            display = "\n".join(f"{index}. {answer}" for index, answer in enumerate(answers, 1))
            store.add_message(
                conversation_id,
                "user",
                display,
                kind="clarification_answers",
                payload={"answers": answers},
            )
            resume_value = {"answers": answers}
        else:
            if conversation.status != "waiting_for_outline_confirmation":
                raise HTTPException(status_code=409, detail="当前会话不在大纲确认阶段")
            if request.action not in {"approve", "revise"}:
                raise HTTPException(status_code=422, detail="请选择确认或修改大纲")
            if request.action == "revise" and not request.feedback.strip():
                raise HTTPException(status_code=422, detail="请输入大纲修改意见")
            content = "确认大纲" if request.action == "approve" else request.feedback.strip()
            store.add_message(
                conversation_id,
                "user",
                content,
                kind="outline_action",
                payload={"action": request.action},
            )
            resume_value = {
                "action": request.action,
                "feedback": request.feedback.strip(),
            }
        return _submit_run(
            store,
            langgraph,
            conversation_id,
            resume=resume_value,
        )

    @app.get("/api/conversations/{conversation_id}/events")
    async def stream_conversation_events(
        conversation_id: str,
        request: Request,
        store: StoreDependency,
        langgraph: LangGraphDependency,
        after: int = 0,
    ) -> StreamingResponse:
        if store.get_conversation(conversation_id) is None:
            raise HTTPException(status_code=404, detail="会话不存在")

        async def event_stream():
            cursor = max(after, 0)
            sent_draft: str | None = None
            heartbeat_ticks = 0
            sync_ticks = 0
            artifacts = ArtifactStore(get_settings().database_url)
            try:
                while not await request.is_disconnected():
                    if sync_ticks == 0:
                        try:
                            await asyncio.to_thread(
                                _sync_active_run,
                                store,
                                conversation_id,
                                langgraph,
                            )
                        except Exception as exc:
                            yield _sse_message(
                                "stream_error",
                                {"message": f"暂时无法同步后台 Run：{exc}"},
                            )
                    sync_ticks = (sync_ticks + 1) % 4

                    for progress in store.list_events(
                        conversation_id,
                        after_id=cursor,
                    ):
                        cursor = progress.id
                        yield _sse_message(
                            "progress",
                            asdict(progress),
                            event_id=progress.id,
                        )

                    try:
                        draft = artifacts.get_text(
                            conversation_id,
                            "research_report_draft",
                        )
                    except ArtifactNotFoundError:
                        draft = ""
                    if sent_draft is None:
                        sent_draft = draft
                        if draft:
                            yield _sse_message(
                                "report_snapshot",
                                {"markdown": draft},
                            )
                    elif draft != sent_draft:
                        if draft.startswith(sent_draft):
                            payload = {"delta": draft[len(sent_draft) :]}
                            event_name = "report_delta"
                        else:
                            payload = {"markdown": draft}
                            event_name = "report_snapshot"
                        sent_draft = draft
                        yield _sse_message(event_name, payload)

                    conversation = store.get_conversation(conversation_id)
                    status = conversation.status if conversation is not None else "deleted"
                    if status not in {"queued", "running", "responding"}:
                        yield _sse_message("state", {"status": status})
                        break

                    heartbeat_ticks += 1
                    if heartbeat_ticks >= 30:
                        heartbeat_ticks = 0
                        yield ": keep-alive\n\n"
                    await asyncio.sleep(0.4)
            finally:
                artifacts.close()

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    @app.get("/api/conversations/{conversation_id}/report")
    def get_report(conversation_id: str, store: StoreDependency) -> dict[str, str]:
        conversation = store.get_conversation(conversation_id)
        if conversation is None:
            raise HTTPException(status_code=404, detail="会话不存在")
        if conversation.status != "complete":
            raise HTTPException(status_code=409, detail="报告尚未生成")
        artifacts = ArtifactStore(get_settings().database_url)
        try:
            markdown = artifacts.get_text(conversation_id, "research_report")
        except ArtifactNotFoundError as exc:
            raise HTTPException(status_code=404, detail="报告产物不存在") from exc
        finally:
            artifacts.close()
        return {"markdown": markdown}

    return app


app = create_app()
