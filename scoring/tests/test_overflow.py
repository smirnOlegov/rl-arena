"""Переполнение очереди на настоящем RabbitMQ: отказ брокера, 503, ожидание источника, DLQ."""

import asyncio
from collections.abc import Awaitable, Callable

import pytest
from aio_pika.exceptions import DeliveryError
from asgi_lifespan import LifespanManager
from faststream import TestApp
from faststream.rabbit import RabbitBroker
from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select

from scoring import api, source, worker
from scoring.config import Settings
from scoring.db import TaskRow
from scoring.topology import dead_letter_queue, declare, tasks_queue
from tests.conftest import make_task, observation


@pytest.fixture
def bounded(settings: Settings, rabbit_url: str) -> Settings:
    return settings.model_copy(update={"rabbit_url": rabbit_url, "queue_max_length": 5})


async def fill(broker: RabbitBroker, settings: Settings) -> int:
    accepted = 0
    while True:
        try:
            await broker.publish(make_task(), queue=settings.queue_name, persist=True)
        except DeliveryError:
            return accepted
        accepted += 1


async def depth(broker: RabbitBroker, settings: Settings) -> int:
    queue = await broker.declare_queue(tasks_queue(settings))
    return (await queue.declare()).message_count or 0


async def eventually(check: Callable[[], Awaitable[bool]]) -> None:
    async with asyncio.timeout(20):
        while not await check():  # noqa: ASYNC110
            await asyncio.sleep(0.1)


def count_rows(sessions) -> Callable[[], Awaitable[int]]:
    async def count() -> int:
        async with sessions() as session:
            return await session.scalar(select(func.count(func.distinct(TaskRow.id)))) or 0

    return count


async def test_full_queue_rejects_publish(bounded):
    async with RabbitBroker(bounded.rabbit_url) as broker:
        await declare(broker, bounded)
        accepted = await fill(broker, bounded)
        assert bounded.queue_max_length <= accepted <= bounded.queue_max_length + 1
        assert await depth(broker, bounded) == accepted


async def test_api_sheds_load_without_pending_task(bounded, sessions):
    app = api.create_app(bounded)
    async with (
        LifespanManager(app),
        AsyncClient(transport=ASGITransport(app), base_url="http://test") as client,
    ):
        await fill(app.state.broker, bounded)
        response = await client.post("/api/v1/tasks", json=observation())

    assert response.status_code == 503
    assert response.headers["retry-after"] == str(api.RETRY_AFTER_MAX_S)
    assert await count_rows(sessions)() == 0


async def test_source_waits_for_worker_without_losing_tasks(bounded, sessions, model, replay):
    publishing = asyncio.create_task(source.run(replay, None, bounded))
    await asyncio.sleep(1.5)
    assert not publishing.done()

    async with RabbitBroker(bounded.rabbit_url) as probe:
        assert await depth(probe, bounded) <= bounded.queue_max_length + 1

    async def all_saved() -> bool:
        return await count_rows(sessions)() == 30

    async with TestApp(worker.create_app(bounded, model)):
        await asyncio.wait_for(publishing, timeout=30)
        await eventually(all_saved)


async def test_poison_message_goes_to_dlq(bounded, sessions, model):
    app = worker.create_app(bounded, model)
    async with TestApp(app):
        await app.broker.publish({"id": "garbage"}, queue=bounded.queue_name, persist=True)
        dlq = await app.broker.declare_queue(dead_letter_queue(bounded))

        async def dead_lettered() -> bool:
            return (await dlq.declare()).message_count == 1

        await eventually(dead_lettered)
        message = await dlq.get()
    assert b"garbage" in message.body
    assert await count_rows(sessions)() == 0


async def test_graceful_shutdown_finishes_taken_tasks(bounded, sessions, model):
    bounded.inference_delay_s = 0.5
    app = worker.create_app(bounded, model)
    async with TestApp(app):
        for _ in range(8):
            await app.broker.publish(make_task(), queue=bounded.queue_name, persist=True)
        await asyncio.sleep(0.2)

    assert await count_rows(sessions)() == bounded.prefetch_count
    async with RabbitBroker(bounded.rabbit_url) as probe:
        assert await depth(probe, bounded) == 8 - bounded.prefetch_count
