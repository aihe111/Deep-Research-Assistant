"""Small synchronous client for the LangGraph Server Thread/Run API."""

from __future__ import annotations

from typing import Any

from langgraph_sdk import get_sync_client

from deep_research_assistant.config import Settings


class LangGraphServerClient:
    """Submit and inspect background Runs owned by LangGraph Server."""

    def __init__(self, settings: Settings) -> None:
        self.assistant_id = settings.langgraph_assistant_id
        self._client = get_sync_client(
            url=settings.langgraph_api_url,
            api_key=settings.langgraph_api_key or None,
            timeout=settings.langgraph_request_timeout_seconds,
        )

    def close(self) -> None:
        self._client.close()

    def create_thread(self, thread_id: str | None = None) -> dict[str, Any]:
        return dict(
            self._client.threads.create(
                thread_id=thread_id,
                if_exists="do_nothing",
                metadata={"application": "deep-research-assistant"},
            )
        )

    def delete_thread(self, thread_id: str) -> None:
        self._client.threads.delete(thread_id)

    def create_run(
        self,
        thread_id: str,
        *,
        input: dict[str, Any] | None = None,
        resume: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        command = {"resume": resume} if resume is not None else None
        return dict(
            self._client.runs.create(
                thread_id,
                self.assistant_id,
                input=input,
                command=command,
                multitask_strategy="reject",
                stream_mode="values",
            )
        )

    def get_run(self, thread_id: str, run_id: str) -> dict[str, Any]:
        return dict(self._client.runs.get(thread_id, run_id))

    def get_state(self, thread_id: str) -> dict[str, Any]:
        return dict(self._client.threads.get_state(thread_id))
