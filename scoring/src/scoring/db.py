"""Таблица задач и запись результата."""

from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID

from sqlalchemy import DateTime, Enum, Float, String
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class TaskStatus(StrEnum):
    pending = "pending"
    done = "done"


class Base(DeclarativeBase):
    pass


class TaskRow(Base):
    __tablename__ = "tasks"

    id: Mapped[UUID] = mapped_column(primary_key=True)
    status: Mapped[TaskStatus] = mapped_column(
        Enum(TaskStatus, native_enum=False, length=16), default=TaskStatus.pending
    )
    win_prob: Mapped[float | None] = mapped_column(Float)
    model_version: Mapped[str | None] = mapped_column(String(32))
    enqueued_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


def make_engine(url: str) -> AsyncEngine:
    return create_async_engine(url, pool_pre_ping=True)


def make_sessionmaker(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


async def create_schema(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def save_result(
    session: AsyncSession,
    task_id: UUID,
    win_prob: float,
    model_version: str,
    enqueued_at: datetime,
) -> None:
    """Повторная доставка той же задачи перезаписывает строку, а не добавляет новую."""
    values = {
        "status": TaskStatus.done,
        "win_prob": win_prob,
        "model_version": model_version,
        "finished_at": datetime.now(UTC),
    }
    stmt = insert(TaskRow).values(id=task_id, enqueued_at=enqueued_at, **values)
    await session.execute(stmt.on_conflict_do_update(index_elements=[TaskRow.id], set_=values))
    await session.commit()
