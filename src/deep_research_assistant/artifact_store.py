"""Database-backed storage for large workflow artifacts."""

import json
from typing import Any

from sqlalchemy import func, insert, select, update

from deep_research_assistant.database import (
    create_database_engine,
    release_database_engine,
    research_artifacts,
)


class ArtifactNotFoundError(KeyError):
    """Raised when a graph state points at a missing artifact."""


class ArtifactStore:
    """Persist JSON and text artifacts with idempotent upserts."""

    def __init__(self, database_url: str) -> None:
        self._engine = create_database_engine(str(database_url))

    def close(self) -> None:
        release_database_engine(self._engine)

    def put_text(
        self,
        thread_id: str,
        artifact_key: str,
        content: str,
        *,
        content_type: str = "text/plain",
    ) -> str:
        with self._engine.begin() as connection:
            existing = connection.execute(
                select(research_artifacts.c.thread_id).where(
                    research_artifacts.c.thread_id == thread_id,
                    research_artifacts.c.artifact_key == artifact_key,
                )
            ).scalar_one_or_none()
            values = {
                "content_type": content_type,
                "content": content,
                "updated_at": func.now(),
            }
            if existing is None:
                connection.execute(
                    insert(research_artifacts).values(
                        thread_id=thread_id,
                        artifact_key=artifact_key,
                        **values,
                    )
                )
            else:
                connection.execute(
                    update(research_artifacts)
                    .where(
                        research_artifacts.c.thread_id == thread_id,
                        research_artifacts.c.artifact_key == artifact_key,
                    )
                    .values(**values)
                )
        return artifact_key

    def get_text(self, thread_id: str, artifact_key: str) -> str:
        with self._engine.connect() as connection:
            value = connection.execute(
                select(research_artifacts.c.content).where(
                    research_artifacts.c.thread_id == thread_id,
                    research_artifacts.c.artifact_key == artifact_key,
                )
            ).scalar_one_or_none()
        if value is None:
            raise ArtifactNotFoundError(f"找不到产物：{thread_id}/{artifact_key}")
        return str(value)

    def put_json(self, thread_id: str, artifact_key: str, value: Any) -> str:
        return self.put_text(
            thread_id,
            artifact_key,
            json.dumps(value, ensure_ascii=False),
            content_type="application/json",
        )

    def get_json(self, thread_id: str, artifact_key: str) -> Any:
        return json.loads(self.get_text(thread_id, artifact_key))
