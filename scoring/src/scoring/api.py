"""HTTP-вход в конвейер: задача ставится в очередь или сразу отклоняется с 503.

Запуск: `uv run uvicorn --factory scoring.api:create_app --port 8000`
"""

import logging
import math
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from aio_pika.exceptions import AMQPError, DeliveryError
from fastapi import APIRouter, FastAPI, HTTPException, Request, Response, status
from faststream.rabbit import RabbitBroker
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from scoring.config import Settings, get_settings
from scoring.db import TaskRow, TaskStatus, create_schema, make_engine, make_sessionmaker
from scoring.topology import Task, declare, tasks_queue

log = logging.getLogger(__name__)

RETRY_AFTER_MAX_S = 60


class Observation(BaseModel):
    """Observation Kaggriculture в формате kaggle_environments (лишние поля допустимы)."""

    model_config = ConfigDict(extra="allow")

    day: int = Field(ge=0, le=30)
    hour: int = Field(default=0, ge=0, le=23)
    farms: list[dict[str, Any]] = Field(min_length=2, max_length=2)
    market: dict[str, Any] = Field(default_factory=dict)
    town: dict[str, Any] = Field(default_factory=dict)


class TaskRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    status: TaskStatus
    win_prob: float | None
    model_version: str | None
    enqueued_at: datetime
    finished_at: datetime | None


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    logging.basicConfig(
        level=settings.log_level, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )
    broker = RabbitBroker(settings.rabbit_url)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        engine = make_engine(settings.database_url)
        await create_schema(engine)
        app.state.sessions = make_sessionmaker(engine)
        async with broker:
            await declare(broker, settings)
            yield
        await engine.dispose()

    def sessions(request: Request) -> async_sessionmaker[AsyncSession]:
        maker: async_sessionmaker[AsyncSession] = request.app.state.sessions
        return maker

    async def retry_after() -> int:
        """Сколько секунд воркерам нужно, чтобы разобрать текущую очередь."""
        queue = await broker.declare_queue(tasks_queue(settings))
        state = await queue.declare()
        if not state.consumer_count:
            return RETRY_AFTER_MAX_S
        capacity = state.consumer_count * settings.prefetch_count / settings.inference_delay_s
        return min(max(math.ceil((state.message_count or 0) / capacity), 1), RETRY_AFTER_MAX_S)

    api = APIRouter(prefix="/api/v1")

    @api.post(
        "/tasks",
        status_code=status.HTTP_202_ACCEPTED,
        responses={503: {"description": "Очередь заполнена, повторить через Retry-After"}},
    )
    async def create_task(body: Observation, request: Request, response: Response) -> TaskRead:
        task = Task(id=uuid4(), observation=body.model_dump(), enqueued_at=datetime.now(UTC))
        async with sessions(request)() as session:
            row = TaskRow(id=task.id, status=TaskStatus.pending, enqueued_at=task.enqueued_at)
            session.add(row)
            await session.flush()
            try:
                await broker.publish(task, queue=settings.queue_name, persist=True, timeout=5)
            except DeliveryError:
                await session.rollback()
                detail, delay = "queue is full", await retry_after()
            except (AMQPError, TimeoutError):
                log.exception("broker unavailable")
                await session.rollback()
                detail, delay = "broker unavailable", RETRY_AFTER_MAX_S
            else:
                await session.commit()
                response.headers["Location"] = f"{api.prefix}/tasks/{task.id}"
                return TaskRead.model_validate(row)
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=detail,
            headers={"Retry-After": str(delay)},
        )

    @api.get("/tasks/{task_id}")
    async def get_task(task_id: UUID, request: Request) -> TaskRead:
        async with sessions(request)() as session:
            row = await session.get(TaskRow, task_id)
        if row is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail="task not found")
        return TaskRead.model_validate(row)

    app = FastAPI(title="scoring", lifespan=lifespan)
    app.state.broker = broker

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(api)
    return app
