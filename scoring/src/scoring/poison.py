"""Заведомо некорректная задача: воркер не разберёт её DELIVERY_LIMIT + 1 раз, и она уйдёт в DLQ.

Публикуется через тот же брокер с OpenTelemetry, поэтому у всех попыток общие task_id и трейс.
Запуск: `uv run python -m scoring.poison` или `make poison`
"""

import asyncio
import logging
from uuid import uuid4

from faststream.rabbit import RabbitBroker
from faststream.rabbit.opentelemetry import RabbitTelemetryMiddleware
from opentelemetry import trace

from scoring.config import Settings, get_settings
from scoring.logs import configure_logging, current_task_id
from scoring.telemetry import setup_telemetry, shutdown_telemetry

log = logging.getLogger("scoring.poison")
tracer = trace.get_tracer(__name__)


async def send(settings: Settings) -> None:
    task_id = str(uuid4())
    current_task_id.set(task_id)
    broker = RabbitBroker(
        settings.rabbit_url,
        middlewares=[RabbitTelemetryMiddleware()],
        logger=logging.getLogger("faststream"),
    )
    async with broker:
        with tracer.start_as_current_span("send poison", attributes={"task.id": task_id}):
            # Нет observation и enqueued_at: схема Task не пропустит
            await broker.publish(
                {"id": task_id}, queue=settings.queue_name, persist=True, correlation_id=task_id
            )
            log.warning("poison task sent", extra={"queue": settings.queue_name})


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level, settings.otel_service_name)
    setup_telemetry(settings)
    try:
        asyncio.run(send(settings))
    finally:
        shutdown_telemetry()


if __name__ == "__main__":
    main()
