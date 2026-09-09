"""Bridge LangGraph Run state into application-owned conversation history."""

from __future__ import annotations

from typing import Any

from deep_research_assistant.artifact_store import ArtifactNotFoundError, ArtifactStore
from deep_research_assistant.config import Settings
from deep_research_assistant.conversation_store import ConversationStore
from deep_research_assistant.followup_assistant import FollowUpAssistant
from deep_research_assistant.hy3_client import Hy3Client


def _interrupt_payload(thread_state: dict[str, Any]) -> dict[str, Any] | None:
    interrupts = thread_state.get("interrupts") or ()
    if not interrupts:
        return None
    first = interrupts[0]
    value = first.get("value") if isinstance(first, dict) else getattr(first, "value", None)
    return value if isinstance(value, dict) else None


def _already_recorded(
    store: ConversationStore,
    conversation_id: str,
    run_id: str,
) -> bool:
    return any(
        message.payload.get("langgraph_run_id") == run_id
        for message in store.list_messages(conversation_id)
    )


def record_run_state(
    store: ConversationStore,
    conversation_id: str,
    run_id: str,
    thread_state: dict[str, Any],
) -> None:
    """Persist one terminal LangGraph Run outcome exactly once under normal polling."""

    if _already_recorded(store, conversation_id, run_id):
        store.update_conversation(conversation_id, clear_active_run=True)
        return

    pending = _interrupt_payload(thread_state)
    if pending:
        kind = pending.get("kind")
        if kind == "clarification":
            questions = pending.get("questions") or []
            round_number = int(pending.get("round") or 1)
            max_rounds = int(pending.get("max_rounds") or 3)
            heading = f"为了准确规划调研，请先补充以下信息（{round_number}/{max_rounds}）：\n"
            question_list = "\n".join(
                f"{index}. {question}" for index, question in enumerate(questions, start=1)
            )
            content = heading + question_list
            status = "waiting_for_clarification"
        elif kind == "outline_confirmation":
            revision_count = int(pending.get("revision_count") or 0)
            max_revisions = int(pending.get("max_revisions") or 5)
            content = (
                "调研大纲已经生成。请确认后开始检索，或提出修改意见。"
                f"（已修改 {revision_count}/{max_revisions} 次）"
            )
            status = "waiting_for_outline_confirmation"
        else:
            raise RuntimeError(f"未知的中断类型：{kind}")
        store.add_message(
            conversation_id,
            "assistant",
            content,
            kind=str(kind),
            payload={**pending, "langgraph_run_id": run_id},
        )
        store.update_conversation(
            conversation_id,
            status=status,
            clear_active_run=True,
        )
        return

    values = thread_state.get("values") or {}
    if not isinstance(values, dict):
        values = {}

    if values.get("mode") == "followup" and values.get("followup_answer"):
        metadata = dict(values.get("followup_metadata") or {})
        metadata["langgraph_run_id"] = run_id
        store.add_message(
            conversation_id,
            "assistant",
            str(values["followup_answer"]),
            kind="follow_up",
            payload=metadata,
        )
        store.update_conversation(
            conversation_id,
            status="complete",
            clear_active_run=True,
        )
        return

    if values.get("status") == "complete":
        metrics = {
            "research_unit_count": values.get("research_unit_count", 0),
            "source_count": values.get("source_count", 0),
            "tools_used": values.get("tools_used", []),
            "hy3_calls_used": values.get("hy3_calls_used", 0),
            "hy3_call_budget": values.get("hy3_call_budget", 0),
            "hy3_call_budget_exhausted": values.get("hy3_call_budget_exhausted", False),
            "hy3_calls_by_stage": values.get("hy3_calls_by_stage", {}),
            # Backward-compatible fields for conversations completed by the
            # pre-0.3 literature graph before an in-place deployment upgrade.
            "retrieved_count": values.get("retrieved_count", 0),
            "selected_count": values.get("selected_count", 0),
            "langgraph_run_id": run_id,
        }
        store.add_message(
            conversation_id,
            "assistant",
            "深度研究已经完成，报告和可追溯来源已生成。",
            kind="report_ready",
            payload=metrics,
        )
        store.update_conversation(
            conversation_id,
            status="complete",
            clear_active_run=True,
        )
        return

    error = str(values.get("error") or "调研流程未能完成")
    store.add_message(
        conversation_id,
        "assistant",
        error,
        kind="error",
        payload={"langgraph_run_id": run_id},
    )
    store.update_conversation(
        conversation_id,
        status="failed",
        clear_active_run=True,
    )


def record_run_failure(
    store: ConversationStore,
    conversation_id: str,
    run_id: str,
    run_status: str,
) -> None:
    """Record a LangGraph Server error/timeout while preserving completed reports."""

    if _already_recorded(store, conversation_id, run_id):
        store.update_conversation(conversation_id, clear_active_run=True)
        return
    conversation = store.get_conversation(conversation_id)
    was_followup = conversation is not None and conversation.status == "responding"
    error = f"LangGraph Run {run_status}：请检查 LangGraph Server 日志。"
    store.add_message(
        conversation_id,
        "assistant",
        error,
        kind="error",
        payload={"langgraph_run_id": run_id, "run_status": run_status},
    )
    store.update_conversation(
        conversation_id,
        status="complete" if was_followup else "failed",
        clear_active_run=True,
    )


def answer_follow_up(settings: Settings, conversation_id: str) -> dict[str, Any]:
    """Build a report-grounded answer inside a dedicated LangGraph Run."""

    if not settings.hy3_api_key:
        raise RuntimeError("HY3_API_KEY 未配置")
    store = ConversationStore(settings.database_url)
    artifacts = ArtifactStore(settings.database_url)
    try:
        try:
            report = artifacts.get_text(conversation_id, "research_report")
        except ArtifactNotFoundError as exc:
            raise RuntimeError("当前会话的报告产物不存在") from exc

        assistant = FollowUpAssistant(Hy3Client(settings))
        memory = store.get_memory(conversation_id)
        context = assistant.prepare_context(
            store.list_messages(conversation_id),
            previous_summary=memory.summary,
            summarized_through_message_id=memory.summarized_through_message_id,
        )
        if context.summary_updated:
            store.upsert_memory(
                conversation_id,
                context.conversation_summary,
                context.summarized_through_message_id,
            )
        answer = assistant.respond(report, context)
        return {
            "answer": answer,
            "metadata": {
                "summary_updated": context.summary_updated,
                "summarized_through_message_id": context.summarized_through_message_id,
                "recent_message_count": len(context.recent_messages),
                "relevant_memory_count": len(context.relevant_old_memories),
            },
        }
    finally:
        artifacts.close()
        store.close()
