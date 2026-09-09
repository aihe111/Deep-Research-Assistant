"""Shared SQLAlchemy schema for application-owned persistent data."""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    event,
    func,
    inspect,
    text,
)
from sqlalchemy.engine import Engine, make_url

metadata = MetaData()

conversations = Table(
    "conversations",
    metadata,
    Column("id", String(255), primary_key=True),
    Column("title", String(255), nullable=False),
    Column("status", String(255), nullable=False),
    Column("active_run_id", String(255), nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("updated_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
)

conversation_messages = Table(
    "conversation_messages",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column(
        "conversation_id",
        String(255),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("role", String(255), nullable=False),
    Column("content", Text, nullable=False),
    Column("kind", String(255), nullable=False, server_default="text"),
    Column("payload", JSON, nullable=False, server_default="{}"),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
)

conversation_memories = Table(
    "conversation_memories",
    metadata,
    Column(
        "conversation_id",
        String(255),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column("summary", Text, nullable=False, server_default=""),
    Column("summarized_through_message_id", Integer, nullable=False, server_default="0"),
    Column("updated_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
)

research_artifacts = Table(
    "research_artifacts",
    metadata,
    Column("thread_id", String(255), primary_key=True),
    Column("artifact_key", String(255), primary_key=True),
    Column("content_type", String(255), nullable=False),
    Column("content", Text, nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
)

research_events = Table(
    "research_events",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column(
        "conversation_id",
        String(255),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("event_type", String(100), nullable=False),
    Column("stage", String(100), nullable=False),
    Column("title", String(500), nullable=False),
    Column("detail", Text, nullable=False, server_default=""),
    Column("payload", JSON, nullable=False, server_default="{}"),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
)

Index(
    "idx_conversation_messages_order",
    conversation_messages.c.conversation_id,
    conversation_messages.c.id,
)
Index("idx_conversations_updated", conversations.c.updated_at.desc())
Index(
    "idx_research_events_order",
    research_events.c.conversation_id,
    research_events.c.id,
)


def normalize_database_url(value: str) -> str:
    """Accept legacy filesystem paths while preferring explicit database URLs."""

    if "://" in value:
        return value
    normalized = value.replace("\\", "/")
    if normalized.startswith("/"):
        return f"sqlite:///{normalized}"
    return f"sqlite:///{normalized}"


def _build_database_engine(url: str) -> Engine:
    parsed_url = make_url(url)
    if parsed_url.get_backend_name() == "sqlite":
        database_path = parsed_url.database
        if database_path and database_path != ":memory:":
            Path(database_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(url, pool_pre_ping=True)
    if engine.dialect.name == "sqlite":

        @event.listens_for(engine, "connect")
        def _enable_sqlite_foreign_keys(dbapi_connection: Any, _: Any) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA busy_timeout=5000")
            cursor.close()

    metadata.create_all(engine)
    columns = {column["name"] for column in inspect(engine).get_columns("conversations")}
    if "active_run_id" not in columns:
        with engine.begin() as connection:
            connection.execute(
                text("ALTER TABLE conversations ADD COLUMN active_run_id VARCHAR(255)")
            )
            if "active_job_id" in columns:
                connection.execute(
                    text(
                        "UPDATE conversations SET active_run_id = active_job_id "
                        "WHERE active_run_id IS NULL"
                    )
                )
    return engine


@lru_cache(maxsize=8)
def _create_postgres_engine(url: str) -> Engine:
    """Reuse one connection pool per PostgreSQL URL inside each process."""

    return _build_database_engine(url)


def create_database_engine(database_url: str) -> Engine:
    url = normalize_database_url(database_url)
    if make_url(url).get_backend_name() == "sqlite":
        return _build_database_engine(url)
    return _create_postgres_engine(url)


def release_database_engine(engine: Engine) -> None:
    """Close test/compatibility SQLite engines; PostgreSQL pools are process-scoped."""

    if engine.dialect.name == "sqlite":
        engine.dispose()


def timestamp_text(value: datetime | str | None) -> str | None:
    if value is None:
        return None
    return value.isoformat(sep=" ") if isinstance(value, datetime) else str(value)
