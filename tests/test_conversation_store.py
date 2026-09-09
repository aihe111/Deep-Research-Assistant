import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver

from deep_research_assistant.artifact_store import ArtifactStore
from deep_research_assistant.conversation_store import ConversationStore


def test_store_creates_missing_sqlite_parent_directory(tmp_path) -> None:
    database = tmp_path / "nested" / "data" / "research.db"

    store = ConversationStore(database)
    try:
        store.create_conversation("fresh-clone")
    finally:
        store.close()

    assert database.is_file()


def test_conversation_history_survives_store_reopen(tmp_path) -> None:
    database = tmp_path / "research.db"
    store = ConversationStore(database)
    store.create_conversation("web-1")
    store.update_conversation("web-1", title="RAG 评测调研", status="running")
    store.add_message("web-1", "user", "调研 RAG 评测")
    store.add_message(
        "web-1",
        "assistant",
        "请补充报告字数",
        kind="clarification",
        payload={"questions": ["报告字数是多少？"]},
    )
    store.upsert_memory("web-1", "用户希望生成3000字报告。", 1)
    store.close()

    reopened = ConversationStore(database)
    try:
        conversation = reopened.get_conversation("web-1")
        messages = reopened.list_messages("web-1")

        assert conversation is not None
        assert conversation.title == "RAG 评测调研"
        assert [message.role for message in messages] == ["user", "assistant"]
        assert messages[1].payload == {"questions": ["报告字数是多少？"]}
        assert reopened.get_memory("web-1").summary == "用户希望生成3000字报告。"
    finally:
        reopened.close()


def test_newest_conversation_is_listed_first(tmp_path) -> None:
    store = ConversationStore(tmp_path / "research.db")
    try:
        store.create_conversation("web-1", "第一项调研")
        store.create_conversation("web-2", "第二项调研")

        assert [item.id for item in store.list_conversations()] == ["web-2", "web-1"]
    finally:
        store.close()


def test_delete_conversation_removes_application_data_after_checkpoint_cleanup(tmp_path) -> None:
    database = tmp_path / "research.db"
    store = ConversationStore(database)
    store.create_conversation("web-1")
    store.add_message("web-1", "user", "调研 RAG")
    store.add_event(
        "web-1",
        "tool_completed",
        "research",
        "网页检索已完成",
        detail="消费者购买决策",
        payload={"tool": "tavily_search"},
    )
    store.upsert_memory("web-1", "滚动摘要", 1)
    artifacts = ArtifactStore(database)
    artifacts.put_text("web-1", "research_report", "# Report")
    artifacts.close()
    checkpoint_connection = sqlite3.connect(database)
    saver = SqliteSaver(checkpoint_connection)
    saver.setup()
    checkpoint_connection.execute(
        """
        INSERT INTO checkpoints
            (thread_id, checkpoint_ns, checkpoint_id, type, checkpoint, metadata)
        VALUES (?, '', 'cp-1', 'json', ?, ?)
        """,
        ("web-1", b"{}", b"{}"),
    )
    checkpoint_connection.execute(
        """
        INSERT INTO writes
            (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, value)
        VALUES (?, '', 'cp-1', 'task-1', 0, 'state', 'json', ?)
        """,
        ("web-1", b"{}"),
    )
    checkpoint_connection.commit()

    try:
        saver.delete_thread("web-1")
        assert store.delete_conversation("web-1") is True
        assert store.get_conversation("web-1") is None
        assert store.list_messages("web-1") == []
        assert checkpoint_connection.execute(
            "SELECT COUNT(*) FROM research_artifacts WHERE thread_id = 'web-1'"
        ).fetchone()[0] == 0
        assert checkpoint_connection.execute(
            "SELECT COUNT(*) FROM conversation_memories WHERE conversation_id = 'web-1'"
        ).fetchone()[0] == 0
        assert checkpoint_connection.execute(
            "SELECT COUNT(*) FROM research_events WHERE conversation_id = 'web-1'"
        ).fetchone()[0] == 0
        assert checkpoint_connection.execute(
            "SELECT COUNT(*) FROM checkpoints WHERE thread_id = 'web-1'"
        ).fetchone()[0] == 0
        assert checkpoint_connection.execute(
            "SELECT COUNT(*) FROM writes WHERE thread_id = 'web-1'"
        ).fetchone()[0] == 0
    finally:
        checkpoint_connection.close()
        store.close()


def test_research_events_are_ordered_and_resumable(tmp_path) -> None:
    store = ConversationStore(tmp_path / "research.db")
    try:
        store.create_conversation("web-events")
        first = store.add_event(
            "web-events",
            "researcher_started",
            "research",
            "Researcher 已启动",
            detail="线下商超选址",
        )
        second = store.add_event(
            "web-events",
            "tool_completed",
            "research",
            "Tavily 网页检索已完成",
            payload={"tool": "tavily_search"},
        )

        assert [event.id for event in store.list_events("web-events")] == [
            first.id,
            second.id,
        ]
        assert store.list_events("web-events", after_id=first.id) == [second]
    finally:
        store.close()
