"""ORM-модели лиги.

Типы — переносимые (`Uuid`, `JSON`, enum как VARCHAR): одна схема работает и
в Postgres (прод, CI), и в SQLite (быстрые локальные тесты).
"""

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import JSON, CheckConstraint, DateTime, Enum, ForeignKey, Index, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from rl_arena.domain import AgentKind, AgentPool, JobKind, JobStatus, MatchPurpose, MatchResult


def utcnow() -> datetime:
    """Вернуть текущее время в UTC (timezone-aware)."""
    return datetime.now(UTC)


def str_enum(enum_cls: type[StrEnum]) -> Enum:
    """Хранить StrEnum строкой без native-типа БД: проще миграции, переносимо."""
    return Enum(enum_cls, native_enum=False, length=32, validate_strings=True)


class Base(DeclarativeBase):
    """Базовый класс моделей."""

    type_annotation_map = {dict[str, Any]: JSON}  # noqa: RUF012


class Agent(Base):
    """Агент лиги: чекпоинт + роль + пул."""

    __tablename__ = "agents"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(128), unique=True)
    kind: Mapped[AgentKind] = mapped_column(str_enum(AgentKind))
    pool: Mapped[AgentPool] = mapped_column(str_enum(AgentPool), index=True)
    checkpoint_uri: Mapped[str] = mapped_column(String(1024))
    parent_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("agents.id", ondelete="SET NULL")
    )
    active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Match(Base):
    """Сыгранный матч; исход записан с точки зрения `agent_id`."""

    __tablename__ = "matches"
    __table_args__ = (
        CheckConstraint("agent_id <> opponent_id", name="ck_matches_distinct_players"),
        Index("ix_matches_agent_purpose", "agent_id", "purpose"),
        Index("ix_matches_opponent_purpose", "opponent_id", "purpose"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    agent_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("agents.id", ondelete="CASCADE"))
    opponent_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("agents.id", ondelete="CASCADE"))
    result: Mapped[MatchResult] = mapped_column(str_enum(MatchResult))
    purpose: Mapped[MatchPurpose] = mapped_column(str_enum(MatchPurpose))
    replay_uri: Mapped[str | None] = mapped_column(String(1024))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Job(Base):
    """Задача для GPU-воркера; сама задача уходит в Redis Stream, здесь — её учёт."""

    __tablename__ = "jobs"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    kind: Mapped[JobKind] = mapped_column(str_enum(JobKind))
    status: Mapped[JobStatus] = mapped_column(str_enum(JobStatus), default=JobStatus.queued)
    agent_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("agents.id", ondelete="SET NULL"))
    params: Mapped[dict[str, Any]] = mapped_column(default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
