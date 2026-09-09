"""Persistent runtime wiring for the LangGraph research workflow."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from sqlalchemy.engine import make_url

from deep_research_assistant.artifact_store import ArtifactStore
from deep_research_assistant.config import Settings, get_settings
from deep_research_assistant.database import normalize_database_url
from deep_research_assistant.hy3_client import Hy3Client
from deep_research_assistant.openalex_client import OpenAlexClient
from deep_research_assistant.research_graph import build_research_graph


def delete_thread_checkpoints(settings: Settings, thread_id: str) -> None:
    """Delete a thread without constructing model clients or the research graph."""

    parsed_url = make_url(normalize_database_url(settings.database_url))
    if parsed_url.get_backend_name() == "sqlite":
        connection = sqlite3.connect(parsed_url.database or ":memory:", check_same_thread=False)
        try:
            checkpointer = SqliteSaver(connection)
            checkpointer.setup()
            checkpointer.delete_thread(thread_id)
        finally:
            connection.close()
        return

    from langgraph.checkpoint.postgres import PostgresSaver

    postgres_url = parsed_url.set(drivername="postgresql").render_as_string(hide_password=False)
    with PostgresSaver.from_conn_string(postgres_url) as checkpointer:
        checkpointer.setup()
        checkpointer.delete_thread(thread_id)


class ResearchGraphRuntime:
    """Own the graph checkpointer and artifact connections."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        database_url = normalize_database_url(self.settings.database_url)
        parsed_url = make_url(database_url)
        self._checkpoint_connection: sqlite3.Connection | None = None
        self._checkpoint_context: Any | None = None

        if parsed_url.get_backend_name() == "sqlite":
            database_path = parsed_url.database or ":memory:"
            if database_path != ":memory:":
                Path(database_path).parent.mkdir(parents=True, exist_ok=True)
            self._checkpoint_connection = sqlite3.connect(
                database_path,
                check_same_thread=False,
            )
            self.checkpointer = SqliteSaver(self._checkpoint_connection)
        else:
            from langgraph.checkpoint.postgres import PostgresSaver

            postgres_url = parsed_url.set(drivername="postgresql").render_as_string(
                hide_password=False
            )
            self._checkpoint_context = PostgresSaver.from_conn_string(postgres_url)
            self.checkpointer = self._checkpoint_context.__enter__()

        self.checkpointer.setup()
        self.artifacts = ArtifactStore(database_url)
        self.graph = build_research_graph(
            hy3_client=Hy3Client(self.settings),
            artifact_store=self.artifacts,
            openalex_factory=lambda: OpenAlexClient(self.settings),
            checkpointer=self.checkpointer,
        )

    def delete_thread(self, thread_id: str) -> None:
        self.checkpointer.delete_thread(thread_id)

    def close(self) -> None:
        self.artifacts.close()
        if self._checkpoint_context is not None:
            self._checkpoint_context.__exit__(None, None, None)
        elif self._checkpoint_connection is not None:
            self._checkpoint_connection.close()

    def __enter__(self) -> ResearchGraphRuntime:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
