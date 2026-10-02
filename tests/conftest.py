"""Общие фикстуры.

По умолчанию тесты герметичны и быстры: SQLite во временном файле + fakeredis.
В CI тот же набор тестов дополнительно гоняется против настоящих Postgres и
Redis — для этого задаются `TEST_DATABASE_URL` и `TEST_REDIS_URL`.
"""

import os
from collections.abc import AsyncIterator, Callable, Coroutine
from pathlib import Path
from typing import Any

import fakeredis
import pytest
from asgi_lifespan import LifespanManager
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from rl_arena.core.config import Settings
from rl_arena.db.models import Base
from rl_arena.main import create_app

REAL_REDIS_URL = os.environ.get("TEST_REDIS_URL")
UNREACHABLE_REDIS_URL = "redis://127.0.0.1:1/0"


class FakeRedis(fakeredis.FakeAsyncRedis):
    """fakeredis без команды INFO — отвечаем как настоящий Redis на `INFO server`."""

    async def info(self, section: str | None = None, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"redis_version": "7.4.0-fake"}


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    database_url = os.environ.get("TEST_DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path}/test.db")
    return Settings(
        environment="test",
        database_url=database_url,
        redis_url=REAL_REDIS_URL or UNREACHABLE_REDIS_URL,
        health_timeout_s=1.0,
        exploiter_resample_every=5,
        exploiters_per_league=2,
    )


@pytest.fixture
async def app(settings: Settings) -> AsyncIterator[FastAPI]:
    application = create_app(settings)
    async with LifespanManager(application):
        async with application.state.engine.begin() as connection:
            await connection.run_sync(Base.metadata.drop_all)
            await connection.run_sync(Base.metadata.create_all)
        if REAL_REDIS_URL:
            await application.state.redis.flushdb()
        else:
            await application.state.redis.aclose()
            application.state.redis = FakeRedis()
        yield application


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as http:
        yield http


@pytest.fixture
async def session(app: FastAPI) -> AsyncIterator[AsyncSession]:
    async with app.state.sessionmaker() as db_session:
        yield db_session


type AgentFactory = Callable[..., Coroutine[Any, Any, dict[str, Any]]]


@pytest.fixture
def make_agent(client: AsyncClient) -> AgentFactory:
    """Зарегистрировать агента через API и вернуть JSON ответа."""
    counter = 0

    async def factory(kind: str = "main", pool: str = "training", **extra: Any) -> dict[str, Any]:
        nonlocal counter
        counter += 1
        payload = {
            "name": extra.pop("name", f"{kind}-{pool}-{counter}"),
            "kind": kind,
            "pool": pool,
            "checkpoint_uri": f"s3://checkpoints/{kind}-{counter}.pt",
            **extra,
        }
        response = await client.post("/api/v1/agents", json=payload)
        assert response.status_code == 201, response.text
        return response.json()

    return factory


type MatchRecorder = Callable[..., Coroutine[Any, Any, None]]


@pytest.fixture
def record_matches(client: AsyncClient) -> MatchRecorder:
    """Записать `count` одинаковых матчей."""

    async def recorder(
        agent: dict[str, Any], opponent: dict[str, Any], result: str, purpose: str, count: int = 1
    ) -> None:
        for _ in range(count):
            response = await client.post(
                "/api/v1/matches",
                json={
                    "agent_id": agent["id"],
                    "opponent_id": opponent["id"],
                    "result": result,
                    "purpose": purpose,
                },
            )
            assert response.status_code == 201, response.text

    return recorder
