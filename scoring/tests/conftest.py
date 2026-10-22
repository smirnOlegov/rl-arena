"""Общие фикстуры: Postgres и RabbitMQ в Testcontainers, фейковая модель, фабрика задач."""

import gzip
import json
from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from testcontainers.community.postgres import PostgresContainer
from testcontainers.community.rabbitmq import RabbitMqContainer

from farmvalue.service.model import LoadedModel, ModelInfo
from scoring.config import Settings
from scoring.db import TaskRow, create_schema, make_engine, make_sessionmaker
from scoring.topology import Task


class ConstantPredictor:
    def predict(self, data: pd.DataFrame) -> Any:
        return np.tile([0.25, 0.75], (len(data), 1))


def observation(day: int = 1) -> dict[str, Any]:
    farm = {"money": 100, "tiles": [[{"kind": "EMPTY"}]]}
    return {"day": day, "hour": 0, "farms": [farm, dict(farm)], "market": {}, "town": {}}


def make_task() -> Task:
    return Task(id=uuid4(), observation=observation(), enqueued_at=datetime.now(UTC))


@pytest.fixture(scope="session")
def postgres_url() -> Iterator[str]:
    with PostgresContainer("postgres:17-alpine", driver="asyncpg") as pg:
        yield pg.get_connection_url()


@pytest.fixture(scope="session")
def rabbit_url() -> Iterator[str]:
    with RabbitMqContainer("rabbitmq:4.1-management-alpine") as rmq:
        host, port = rmq.get_container_host_ip(), rmq.get_exposed_port(5672)
        yield f"amqp://guest:guest@{host}:{port}/"


@pytest.fixture
def model() -> LoadedModel:
    info = ModelInfo(name="test", alias="champion", version="7", run_id="run", model_id=None)
    return LoadedModel(info=info, predictor=ConstantPredictor())


@pytest.fixture
def settings(postgres_url: str, request: pytest.FixtureRequest) -> Settings:
    return Settings(
        database_url=postgres_url,
        queue_name=f"tasks-{request.node.name}",
        inference_delay_s=0.0,
    )


@pytest.fixture
async def sessions(settings: Settings) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = make_engine(settings.database_url)
    await create_schema(engine)
    maker = make_sessionmaker(engine)
    async with maker() as session:
        await session.execute(delete(TaskRow))
        await session.commit()
    yield maker
    await engine.dispose()


@pytest.fixture
def replay(tmp_path: Path) -> Path:
    """Реплей из 30 шагов в формате kaggle_environments."""
    steps = [[{"observation": observation(day=step // 24)}] for step in range(30)]
    path = tmp_path / "episode.json.gz"
    with gzip.open(path, "wt") as fh:
        json.dump({"info": {"EpisodeId": str(uuid4())}, "steps": steps}, fh)
    return path
