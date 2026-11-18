"""Источник: пакетный скоринг снимков из реплеев с подстройкой под пропускную способность.

Запуск: `uv run python -m scoring.source [PATH] [--limit N]`, по умолчанию PATH из REPLAYS
"""

import argparse
import asyncio
import gzip
import json
import logging
import random
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from aio_pika.exceptions import DeliveryError
from faststream.rabbit import RabbitBroker
from faststream.rabbit.opentelemetry import RabbitTelemetryMiddleware
from opentelemetry import metrics, trace

from scoring.config import Settings, get_settings
from scoring.logs import configure_logging, current_task_id
from scoring.telemetry import setup_telemetry, shutdown_telemetry
from scoring.topology import Task, declare

# Под `python -m` __name__ равен "__main__", а в логах нужно имя модуля
log = logging.getLogger("scoring.source")
tracer = trace.get_tracer(__name__)
meter = metrics.get_meter(__name__)

PROGRESS_INTERVAL_S = 5

published_total = meter.create_counter(
    "scoring.source.published", unit="{task}", description="Задачи, принятые брокером"
)
rejects_total = meter.create_counter(
    "scoring.source.rejects", unit="{publish}", description="Публикации, отклонённые брокером"
)
backoff_seconds_total = meter.create_counter(
    "scoring.source.backoff", unit="s", description="Время в паузах после отказа брокера"
)

PUBLIC_KEYS = ("day", "hour", "farms", "market", "town")


def observations(path: Path) -> Iterator[tuple[str, dict[str, Any]]]:
    """Снимки seat 0 по шагам; в памяти не больше одной партии."""
    files = sorted(path.rglob("*.json.gz")) if path.is_dir() else [path]
    for file in files:
        with gzip.open(file, "rt") as fh:
            episode = json.load(fh)
        episode_id = (episode.get("info") or {}).get("EpisodeId") or file.name
        for step, agents in enumerate(episode["steps"]):
            obs = agents[0]["observation"]
            yield f"{episode_id}/{step}", {key: obs.get(key) for key in PUBLIC_KEYS}


async def publish(broker: RabbitBroker, settings: Settings, task: Task) -> int:
    """Опубликовать с подтверждением; при отказе ждать и повторять. Возвращает число отказов."""
    delay = settings.backoff_min_s
    rejects = 0
    while True:
        try:
            accepted = task.model_copy(update={"enqueued_at": datetime.now(UTC)})
            await broker.publish(
                accepted, queue=settings.queue_name, persist=True, correlation_id=str(task.id)
            )
            return rejects
        except DeliveryError:
            rejects += 1
            rejects_total.add(1)
            pause = delay * random.uniform(0.5, 1.0)  # noqa: S311
            log.debug(
                "publish rejected, queue is full", extra={"attempt": rejects, "pause_s": pause}
            )
            with tracer.start_as_current_span(
                "backoff", attributes={"source.rejects": rejects, "source.pause_s": pause}
            ):
                await asyncio.sleep(pause)
            backoff_seconds_total.add(pause)
            delay = min(delay * 2, settings.backoff_max_s)


async def enqueue(broker: RabbitBroker, settings: Settings, key: str, task: Task) -> int:
    """Одна задача источника — один трейс; пакет из тысяч задач в один трейс не собирается."""
    with tracer.start_as_current_span(
        "enqueue", attributes={"task.id": str(task.id), "source.key": key}
    ) as span:
        token = current_task_id.set(str(task.id))
        try:
            rejects = await publish(broker, settings, task)
        finally:
            current_task_id.reset(token)
        span.set_attribute("source.rejects", rejects)
    published_total.add(1)
    return rejects


async def run(path: Path, limit: int | None, settings: Settings) -> None:
    broker = RabbitBroker(
        settings.rabbit_url,
        middlewares=[RabbitTelemetryMiddleware()],
        logger=logging.getLogger("faststream"),
    )
    async with broker:
        await declare(broker, settings)
        log.info("source started", extra={"path": str(path), "limit": limit})
        started = window = time.monotonic()
        sent = rejects = window_sent = window_rejects = 0
        for key, observation in observations(path):
            if limit is not None and sent >= limit:
                break
            task = Task(
                id=uuid5(NAMESPACE_URL, key), observation=observation, enqueued_at=datetime.now(UTC)
            )
            task_rejects = await enqueue(broker, settings, key, task)
            rejects += task_rejects
            window_rejects += task_rejects
            sent += 1
            if (now := time.monotonic()) - window >= PROGRESS_INTERVAL_S:
                progress = {
                    "sent": sent,
                    "rate": round((sent - window_sent) / (now - window), 1),
                    "rejected": rejects,
                    "window_rejected": window_rejects,
                }
                if window_rejects:
                    log.warning("queue is full, source slowed down", extra=progress)
                else:
                    log.info("progress", extra=progress)
                window, window_sent, window_rejects = now, sent, 0
        log.info(
            "source finished",
            extra={
                "sent": sent,
                "rejected": rejects,
                "duration_s": round(time.monotonic() - started, 1),
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, nargs="?")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    settings = get_settings()
    configure_logging(settings.log_level, settings.otel_service_name)
    setup_telemetry(settings)
    try:
        asyncio.run(run(args.path or settings.replays, args.limit, settings))
    finally:
        shutdown_telemetry()


if __name__ == "__main__":
    main()
