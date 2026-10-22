"""Формат задачи и очереди RabbitMQ: ограниченная очередь задач и DLQ."""

from datetime import datetime
from typing import Any
from uuid import UUID

from faststream.rabbit import ExchangeType, QueueType, RabbitBroker, RabbitExchange, RabbitQueue
from faststream.rabbit.schemas.queue import QuorumQueueArgs
from pydantic import BaseModel

from scoring.config import Settings


class Task(BaseModel):
    id: UUID
    observation: dict[str, Any]
    enqueued_at: datetime


def dead_letter_exchange(settings: Settings) -> RabbitExchange:
    return RabbitExchange(f"{settings.queue_name}.dlx", type=ExchangeType.FANOUT, durable=True)


def dead_letter_queue(settings: Settings) -> RabbitQueue:
    return RabbitQueue(f"{settings.queue_name}.dlq", queue_type=QueueType.QUORUM, durable=True)


def tasks_queue(settings: Settings) -> RabbitQueue:
    arguments: QuorumQueueArgs = {
        "x-delivery-limit": settings.delivery_limit,
        "x-dead-letter-exchange": dead_letter_exchange(settings).name,
    }
    if settings.queue_max_length > 0:
        arguments["x-max-length"] = settings.queue_max_length
        arguments["x-overflow"] = "reject-publish"
        arguments["x-dead-letter-strategy"] = "at-least-once"
    return RabbitQueue(
        settings.queue_name, queue_type=QueueType.QUORUM, durable=True, arguments=arguments
    )


async def declare(broker: RabbitBroker, settings: Settings) -> None:
    """Объявить топологию до первой публикации: без DLX брокер молча выбросит отказы."""
    dlx = await broker.declare_exchange(dead_letter_exchange(settings))
    dlq = await broker.declare_queue(dead_letter_queue(settings))
    await dlq.bind(dlx)
    await broker.declare_queue(tasks_queue(settings))
