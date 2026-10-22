"""Обработчик воркера на TestRabbitBroker: запись результата, идемпотентность, event loop."""

import asyncio
import time

import pytest
from faststream.rabbit import TestRabbitBroker
from sqlalchemy import func, select

from scoring.db import TaskRow, TaskStatus
from scoring.worker import create_app
from tests.conftest import make_task


async def test_result_saved(settings, sessions, model):
    app = create_app(settings, model)
    task = make_task()
    async with TestRabbitBroker(app.broker) as broker:
        await broker.publish(task, queue=settings.queue_name)

    async with sessions() as session:
        row = await session.get(TaskRow, task.id)
    assert row is not None
    assert row.status is TaskStatus.done
    assert row.win_prob == pytest.approx(0.75)
    assert row.model_version == "7"


async def test_redelivery_does_not_duplicate(settings, sessions, model):
    app = create_app(settings, model)
    task = make_task()
    async with TestRabbitBroker(app.broker) as broker:
        await broker.publish(task, queue=settings.queue_name)
        await broker.publish(task, queue=settings.queue_name)

    async with sessions() as session:
        count = await session.scalar(select(func.count()).select_from(TaskRow))
    assert count == 1


async def test_invalid_message_fails_without_result(settings, sessions, model):
    app = create_app(settings, model)
    async with TestRabbitBroker(app.broker) as broker:
        with pytest.raises(Exception, match="validation error"):
            await broker.publish({"id": "not-a-uuid"}, queue=settings.queue_name)

    async with sessions() as session:
        assert await session.scalar(select(func.count()).select_from(TaskRow)) == 0


async def test_inference_does_not_block_event_loop(settings, sessions, model):
    settings.inference_delay_s = 0.5
    app = create_app(settings, model)
    gaps: list[float] = []

    async def ticker() -> None:
        last = time.monotonic()
        while True:
            await asyncio.sleep(0.01)
            now = time.monotonic()
            gaps.append(now - last)
            last = now

    async with TestRabbitBroker(app.broker) as broker:
        tick = asyncio.create_task(ticker())
        await broker.publish(make_task(), queue=settings.queue_name)
        tick.cancel()

    assert len(gaps) > 20
    assert max(gaps) < 0.1
