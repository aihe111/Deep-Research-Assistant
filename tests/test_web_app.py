import json
import uuid
from typing import Any

from fastapi.testclient import TestClient

from deep_research_assistant import research_execution, web_app
from deep_research_assistant.artifact_store import ArtifactStore
from deep_research_assistant.config import Settings
from deep_research_assistant.conversation_store import ConversationStore


class FakeLangGraphClient:
    def __init__(self) -> None:
        self.threads: set[str] = set()
        self.runs: dict[str, dict[str, Any]] = {}
        self.states: dict[str, dict[str, Any]] = {}
        self.created_runs: list[dict[str, Any]] = []

    def close(self) -> None:
        return None

    def create_thread(self, thread_id: str | None = None) -> dict[str, Any]:
        thread_id = thread_id or str(uuid.uuid4())
        self.threads.add(thread_id)
        return {"thread_id": thread_id}

    def delete_thread(self, thread_id: str) -> None:
        self.threads.discard(thread_id)

    def create_run(
        self,
        thread_id: str,
        *,
        input: dict[str, Any] | None = None,
        resume: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        run_id = f"run-{len(self.runs) + 1}"
        run = {"run_id": run_id, "thread_id": thread_id, "status": "pending"}
        self.runs[run_id] = run
        self.created_runs.append(
            {"thread_id": thread_id, "run_id": run_id, "input": input, "resume": resume}
        )
        return run

    def get_run(self, thread_id: str, run_id: str) -> dict[str, Any]:
        run = self.runs[run_id]
        assert run["thread_id"] == thread_id
        return run

    def get_state(self, thread_id: str) -> dict[str, Any]:
        return self.states[thread_id]


def _client(tmp_path, monkeypatch, *, hy3_api_key: str = "test-key"):
    settings = Settings(
        _env_file=None,
        database_url=str(tmp_path / "research.db"),
        hy3_api_key=hy3_api_key,
    )
    fake = FakeLangGraphClient()
    monkeypatch.setattr(web_app, "get_settings", lambda: settings)
    monkeypatch.setattr(web_app, "LangGraphServerClient", lambda _settings: fake)
    return TestClient(web_app.create_app()), fake, settings


def test_web_app_creates_and_restores_conversation_history(tmp_path, monkeypatch) -> None:
    client, langgraph, _ = _client(tmp_path, monkeypatch)

    created = client.post("/api/conversations")
    conversation_id = created.json()["conversation"]["id"]
    listed = client.get("/api/conversations")
    restored = client.get(f"/api/conversations/{conversation_id}")

    assert created.status_code == 201
    assert conversation_id in langgraph.threads
    assert listed.status_code == 200
    assert listed.json()[0]["id"] == conversation_id
    assert restored.json()["conversation"]["status"] == "new"
    assert restored.json()["messages"] == []
    assert restored.json()["events"] == []


def test_web_app_serves_interface_assets(tmp_path, monkeypatch) -> None:
    client, _, _ = _client(tmp_path, monkeypatch, hy3_api_key="")

    page = client.get("/")
    script = client.get("/assets/app.js")

    assert page.status_code == 200
    assert "Deep-Research-Assistant" in page.text
    assert "新建调研" in page.text
    assert "你想研究什么？" in page.text
    assert 'id="menu-button"' in page.text
    assert 'id="new-chat"' not in page.text
    assert 'id="new-chat-panel"' in page.text
    assert '<div class="rail-brand" aria-hidden="true">D</div>' in page.text
    assert "suggestion-grid" not in page.text
    assert script.status_code == 200
    assert "LangGraph Run" in script.text
    assert "new EventSource" in script.text
    assert "研究过程" in script.text
    assert "elements.input.disabled = composerDisabled" in script.text
    assert "elements.send.disabled = composerDisabled" in script.text
    process_renderer = script.text.split("function renderProcessPanel", 1)[1].split(
        "function renderMessage", 1
    )[0]
    assert "created_at" not in process_renderer


def test_event_stream_replays_progress_and_report_snapshot(tmp_path, monkeypatch) -> None:
    client, _, settings = _client(tmp_path, monkeypatch)
    conversation_id = client.post("/api/conversations").json()["conversation"]["id"]
    store = ConversationStore(settings.database_url)
    try:
        event = store.add_event(
            conversation_id,
            "tool_completed",
            "research",
            "Tavily 网页检索已完成",
            detail="社区养老服务",
        )
        store.update_conversation(conversation_id, status="complete")
    finally:
        store.close()
    artifacts = ArtifactStore(settings.database_url)
    artifacts.put_text(
        conversation_id,
        "research_report_draft",
        "# 社区养老服务调研",
    )
    artifacts.close()

    response = client.get(f"/api/conversations/{conversation_id}/events?after=0")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert f"id: {event.id}" in response.text
    assert "event: progress" in response.text
    assert "Tavily 网页检索已完成" in response.text
    assert "event: report_snapshot" in response.text
    assert "# 社区养老服务调研" in response.text
    assert "event: state" in response.text


def test_event_stream_after_cursor_skips_old_progress(tmp_path, monkeypatch) -> None:
    client, _, settings = _client(tmp_path, monkeypatch)
    conversation_id = client.post("/api/conversations").json()["conversation"]["id"]
    store = ConversationStore(settings.database_url)
    try:
        first = store.add_event(
            conversation_id,
            "outline_started",
            "outline",
            "正在生成研究大纲",
        )
        second = store.add_event(
            conversation_id,
            "outline_completed",
            "outline",
            "研究大纲已生成",
        )
        store.update_conversation(conversation_id, status="complete")
    finally:
        store.close()

    response = client.get(
        f"/api/conversations/{conversation_id}/events?after={first.id}"
    )

    assert f"id: {first.id}\n" not in response.text
    assert f"id: {second.id}\n" in response.text
    assert "正在生成研究大纲" not in response.text
    assert "研究大纲已生成" in response.text


def test_web_app_deletes_application_conversation_and_langgraph_thread(
    tmp_path, monkeypatch
) -> None:
    client, langgraph, _ = _client(tmp_path, monkeypatch)
    conversation_id = client.post("/api/conversations").json()["conversation"]["id"]

    deleted = client.delete(f"/api/conversations/{conversation_id}")

    assert deleted.status_code == 204
    assert conversation_id not in langgraph.threads
    assert client.get(f"/api/conversations/{conversation_id}").status_code == 404


def test_web_message_creates_run_and_resume_uses_command(tmp_path, monkeypatch) -> None:
    client, langgraph, _ = _client(tmp_path, monkeypatch)
    conversation_id = client.post("/api/conversations").json()["conversation"]["id"]

    started = client.post(
        f"/api/conversations/{conversation_id}/messages",
        json={"content": "调研 RAG 评测方法"},
    )
    first_run = langgraph.created_runs[-1]
    assert first_run["input"]["mode"] == "research"
    assert first_run["input"]["original_request"] == "调研 RAG 评测方法"

    langgraph.runs[first_run["run_id"]]["status"] = "interrupted"
    langgraph.states[conversation_id] = {
        "values": first_run["input"],
        "interrupts": [
            {
                "value": {
                    "kind": "clarification",
                    "questions": ["报告字数是多少？"],
                    "round": 1,
                    "max_rounds": 3,
                }
            }
        ],
    }
    paused = client.get(f"/api/conversations/{conversation_id}").json()
    assert paused["conversation"]["status"] == "waiting_for_clarification"

    resumed = client.post(
        f"/api/conversations/{conversation_id}/resume",
        json={"kind": "clarification", "answers": ["3000字"]},
    )

    assert started.status_code == 202
    assert resumed.status_code == 202
    assert resumed.json()["run_id"] == langgraph.created_runs[-1]["run_id"]
    assert langgraph.created_runs[-1]["resume"] == {"answers": ["3000字"]}
    restored = client.get(f"/api/conversations/{conversation_id}").json()
    assert restored["conversation"]["status"] == "queued"
    assert [message["kind"] for message in restored["messages"]] == [
        "text",
        "clarification",
        "clarification_answers",
    ]


def test_completed_conversation_follow_up_is_a_langgraph_run(tmp_path, monkeypatch) -> None:
    client, langgraph, settings = _client(tmp_path, monkeypatch)
    conversation_id = client.post("/api/conversations").json()["conversation"]["id"]
    store = ConversationStore(settings.database_url)
    store.update_conversation(conversation_id, status="complete")
    store.close()

    response = client.post(
        f"/api/conversations/{conversation_id}/messages",
        json={"content": "这个结论有什么局限？"},
    )
    run = langgraph.created_runs[-1]
    assert response.status_code == 202
    assert run["input"] == {
        "thread_id": conversation_id,
        "mode": "followup",
    }
    store = ConversationStore(settings.database_url)
    try:
        assert store.list_messages(conversation_id)[-1].content == "这个结论有什么局限？"
    finally:
        store.close()

    langgraph.runs[run["run_id"]]["status"] = "success"
    langgraph.states[conversation_id] = {
        "interrupts": [],
        "values": {
            "mode": "followup",
            "status": "complete",
            "followup_answer": "现有证据只覆盖论文摘要。",
            "followup_metadata": {"summary_updated": False},
        },
    }
    completed = client.get(f"/api/conversations/{conversation_id}").json()
    assert completed["conversation"]["status"] == "complete"
    assert completed["messages"][-1]["kind"] == "follow_up"
    assert completed["messages"][-1]["content"] == "现有证据只覆盖论文摘要。"


def test_follow_up_graph_node_persists_rolling_memory(tmp_path, monkeypatch) -> None:
    database = tmp_path / "research.db"
    settings = Settings(
        _env_file=None,
        database_url=str(database),
        hy3_api_key="test-key",
    )

    class FakeHy3Client:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def chat(self, messages, **_kwargs):
            if "记忆压缩节点" in messages[0]["content"]:
                return "- 用户最初要求报告控制在3000字。"
            payload = json.loads(messages[1]["content"])
            assert set(payload) == {
                "current_report",
                "conversation_summary",
                "recent_messages",
                "relevant_old_memories",
            }
            return "最初要求是3000字。"

    monkeypatch.setattr(research_execution, "Hy3Client", FakeHy3Client)
    store = ConversationStore(database)
    store.create_conversation("web-memory")
    store.update_conversation("web-memory", status="responding")
    store.add_message("web-memory", "user", "报告控制在3000字")
    for index in range(2, 20):
        store.add_message("web-memory", "assistant", f"中间消息{index}")
    store.add_message("web-memory", "user", "我最初要求多少字？")
    artifacts = ArtifactStore(database)
    artifacts.put_text("web-memory", "research_report", "# 测试报告")
    artifacts.close()
    store.close()

    result = research_execution.answer_follow_up(settings, "web-memory")

    reopened = ConversationStore(database)
    try:
        memory = reopened.get_memory("web-memory")
        assert result["answer"] == "最初要求是3000字。"
        assert result["metadata"]["summary_updated"] is True
        assert memory.summarized_through_message_id == 4
        assert "3000" in memory.summary
    finally:
        reopened.close()
