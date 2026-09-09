from deep_research_assistant.conversation_store import ConversationStore
from deep_research_assistant.research_execution import record_run_failure, record_run_state


def test_terminal_run_state_is_recorded_once(tmp_path) -> None:
    store = ConversationStore(tmp_path / "research.db")
    store.create_conversation("web-1")
    store.update_conversation("web-1", status="running", active_run_id="run-1")
    state = {
        "interrupts": [],
        "values": {
            "status": "complete",
            "retrieved_count": 10,
            "selected_count": 5,
        },
    }
    try:
        record_run_state(store, "web-1", "run-1", state)
        record_run_state(store, "web-1", "run-1", state)

        conversation = store.get_conversation("web-1")
        messages = store.list_messages("web-1")
        assert conversation is not None
        assert conversation.status == "complete"
        assert conversation.active_run_id is None
        assert len(messages) == 1
        assert messages[0].payload["langgraph_run_id"] == "run-1"
    finally:
        store.close()


def test_failed_followup_run_preserves_completed_report_status(tmp_path) -> None:
    store = ConversationStore(tmp_path / "research.db")
    store.create_conversation("web-1")
    store.update_conversation("web-1", status="responding", active_run_id="run-1")
    try:
        record_run_failure(store, "web-1", "run-1", "timeout")
        conversation = store.get_conversation("web-1")
        assert conversation is not None
        assert conversation.status == "complete"
        assert conversation.active_run_id is None
        assert store.list_messages("web-1")[-1].kind == "error"
    finally:
        store.close()
