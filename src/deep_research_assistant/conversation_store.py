"""Persistent conversation and background-job state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import delete, func, insert, select, update
from sqlalchemy.engine import RowMapping

from deep_research_assistant.database import (
    conversation_memories,
    conversation_messages,
    conversations,
    create_database_engine,
    release_database_engine,
    research_artifacts,
    research_events,
    timestamp_text,
)


@dataclass(frozen=True)
class Conversation:
    id: str
    title: str
    status: str
    active_run_id: str | None
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class ConversationMessage:
    id: int
    conversation_id: str
    role: str
    content: str
    kind: str
    payload: dict[str, Any]
    created_at: str


@dataclass(frozen=True)
class ConversationMemory:
    conversation_id: str
    summary: str
    summarized_through_message_id: int
    updated_at: str | None


@dataclass(frozen=True)
class ResearchEvent:
    id: int
    conversation_id: str
    event_type: str
    stage: str
    title: str
    detail: str
    payload: dict[str, Any]
    created_at: str


class ConversationStore:
    """Store application conversations while LangGraph Server owns Runs."""

    def __init__(self, database_url: str) -> None:
        self._engine = create_database_engine(str(database_url))

    def close(self) -> None:
        release_database_engine(self._engine)

    def create_conversation(self, conversation_id: str, title: str = "新调研") -> Conversation:
        with self._engine.begin() as connection:
            connection.execute(
                insert(conversations).values(
                    id=conversation_id,
                    title=title,
                    status="new",
                )
            )
        conversation = self.get_conversation(conversation_id)
        if conversation is None:
            raise RuntimeError("创建会话失败")
        return conversation

    def list_conversations(self, limit: int = 100) -> list[Conversation]:
        statement = (
            select(conversations)
            .order_by(
                conversations.c.updated_at.desc(),
                conversations.c.created_at.desc(),
                conversations.c.id.desc(),
            )
            .limit(limit)
        )
        with self._engine.connect() as connection:
            rows = connection.execute(statement).mappings().all()
        return [self._conversation_from_row(row) for row in rows]

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        with self._engine.connect() as connection:
            row = (
                connection.execute(
                    select(conversations).where(conversations.c.id == conversation_id)
                )
                .mappings()
                .one_or_none()
            )
        return self._conversation_from_row(row) if row else None

    def update_conversation(
        self,
        conversation_id: str,
        *,
        title: str | None = None,
        status: str | None = None,
        active_run_id: str | None = None,
        clear_active_run: bool = False,
    ) -> None:
        values: dict[str, Any] = {"updated_at": func.now()}
        if title is not None:
            values["title"] = title
        if status is not None:
            values["status"] = status
        if active_run_id is not None:
            values["active_run_id"] = active_run_id
        elif clear_active_run:
            values["active_run_id"] = None
        with self._engine.begin() as connection:
            connection.execute(
                update(conversations)
                .where(conversations.c.id == conversation_id)
                .values(**values)
            )

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        *,
        kind: str = "text",
        payload: dict[str, Any] | None = None,
    ) -> ConversationMessage:
        with self._engine.begin() as connection:
            message_id = connection.execute(
                insert(conversation_messages)
                .values(
                    conversation_id=conversation_id,
                    role=role,
                    content=content,
                    kind=kind,
                    payload=payload or {},
                )
                .returning(conversation_messages.c.id)
            ).scalar_one()
            connection.execute(
                update(conversations)
                .where(conversations.c.id == conversation_id)
                .values(updated_at=func.now())
            )
        message = self.get_message(int(message_id))
        if message is None:
            raise RuntimeError("保存消息失败")
        return message

    def get_message(self, message_id: int) -> ConversationMessage | None:
        with self._engine.connect() as connection:
            row = (
                connection.execute(
                    select(conversation_messages).where(conversation_messages.c.id == message_id)
                )
                .mappings()
                .one_or_none()
            )
        return self._message_from_row(row) if row else None

    def list_messages(self, conversation_id: str) -> list[ConversationMessage]:
        statement = (
            select(conversation_messages)
            .where(conversation_messages.c.conversation_id == conversation_id)
            .order_by(conversation_messages.c.id.asc())
        )
        with self._engine.connect() as connection:
            rows = connection.execute(statement).mappings().all()
        return [self._message_from_row(row) for row in rows]

    def add_event(
        self,
        conversation_id: str,
        event_type: str,
        stage: str,
        title: str,
        *,
        detail: str = "",
        payload: dict[str, Any] | None = None,
    ) -> ResearchEvent:
        """Persist one sanitized user-facing progress event."""

        with self._engine.begin() as connection:
            event_id = connection.execute(
                insert(research_events)
                .values(
                    conversation_id=conversation_id,
                    event_type=event_type,
                    stage=stage,
                    title=title,
                    detail=detail,
                    payload=payload or {},
                )
                .returning(research_events.c.id)
            ).scalar_one()
        event = self.get_event(int(event_id))
        if event is None:
            raise RuntimeError("保存研究过程事件失败")
        return event

    def get_event(self, event_id: int) -> ResearchEvent | None:
        with self._engine.connect() as connection:
            row = (
                connection.execute(
                    select(research_events).where(research_events.c.id == event_id)
                )
                .mappings()
                .one_or_none()
            )
        return self._event_from_row(row) if row else None

    def list_events(
        self,
        conversation_id: str,
        *,
        after_id: int = 0,
        limit: int = 500,
    ) -> list[ResearchEvent]:
        statement = (
            select(research_events)
            .where(
                research_events.c.conversation_id == conversation_id,
                research_events.c.id > max(after_id, 0),
            )
            .order_by(research_events.c.id.asc())
            .limit(limit)
        )
        with self._engine.connect() as connection:
            rows = connection.execute(statement).mappings().all()
        return [self._event_from_row(row) for row in rows]

    def get_memory(self, conversation_id: str) -> ConversationMemory:
        with self._engine.connect() as connection:
            row = (
                connection.execute(
                    select(conversation_memories).where(
                        conversation_memories.c.conversation_id == conversation_id
                    )
                )
                .mappings()
                .one_or_none()
            )
        if row is None:
            return ConversationMemory(conversation_id, "", 0, None)
        return ConversationMemory(
            conversation_id=str(row["conversation_id"]),
            summary=str(row["summary"]),
            summarized_through_message_id=int(row["summarized_through_message_id"]),
            updated_at=timestamp_text(row["updated_at"]),
        )

    def upsert_memory(
        self,
        conversation_id: str,
        summary: str,
        summarized_through_message_id: int,
    ) -> ConversationMemory:
        with self._engine.begin() as connection:
            existing = connection.execute(
                select(conversation_memories.c.conversation_id).where(
                    conversation_memories.c.conversation_id == conversation_id
                )
            ).scalar_one_or_none()
            values = {
                "summary": summary,
                "summarized_through_message_id": summarized_through_message_id,
                "updated_at": func.now(),
            }
            if existing is None:
                connection.execute(
                    insert(conversation_memories).values(
                        conversation_id=conversation_id,
                        **values,
                    )
                )
            else:
                connection.execute(
                    update(conversation_memories)
                    .where(conversation_memories.c.conversation_id == conversation_id)
                    .values(**values)
                )
        return self.get_memory(conversation_id)

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete application-owned history and artifacts for a conversation."""

        with self._engine.begin() as connection:
            exists = connection.execute(
                select(conversations.c.id).where(conversations.c.id == conversation_id)
            ).scalar_one_or_none()
            if exists is None:
                return False
            connection.execute(
                delete(research_artifacts).where(research_artifacts.c.thread_id == conversation_id)
            )
            connection.execute(delete(conversations).where(conversations.c.id == conversation_id))
        return True

    @staticmethod
    def _conversation_from_row(row: RowMapping) -> Conversation:
        return Conversation(
            id=str(row["id"]),
            title=str(row["title"]),
            status=str(row["status"]),
            active_run_id=str(row["active_run_id"]) if row["active_run_id"] else None,
            created_at=str(timestamp_text(row["created_at"])),
            updated_at=str(timestamp_text(row["updated_at"])),
        )

    @staticmethod
    def _message_from_row(row: RowMapping) -> ConversationMessage:
        return ConversationMessage(
            id=int(row["id"]),
            conversation_id=str(row["conversation_id"]),
            role=str(row["role"]),
            content=str(row["content"]),
            kind=str(row["kind"]),
            payload=dict(row["payload"] or {}),
            created_at=str(timestamp_text(row["created_at"])),
        )

    @staticmethod
    def _event_from_row(row: RowMapping) -> ResearchEvent:
        return ResearchEvent(
            id=int(row["id"]),
            conversation_id=str(row["conversation_id"]),
            event_type=str(row["event_type"]),
            stage=str(row["stage"]),
            title=str(row["title"]),
            detail=str(row["detail"] or ""),
            payload=dict(row["payload"] or {}),
            created_at=str(timestamp_text(row["created_at"])),
        )
