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

from scoring.config import Settings, get_settings
from scoring.topology import Task, declare

log = logging.getLogger(__name__)

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
            await broker.publish(accepted, queue=settings.queue_name, persist=True)
            return rejects
        except DeliveryError:
            rejects += 1
            await asyncio.sleep(delay * random.uniform(0.5, 1.0))  # noqa: S311
            delay = min(delay * 2, settings.backoff_max_s)


async def run(path: Path, limit: int | None, settings: Settings) -> None:
    async with RabbitBroker(settings.rabbit_url) as broker:
        await declare(broker, settings)
        started = window = time.monotonic()
        sent = rejects = window_sent = 0
        for key, observation in observations(path):
            if limit is not None and sent >= limit:
                break
            task = Task(
                id=uuid5(NAMESPACE_URL, key), observation=observation, enqueued_at=datetime.now(UTC)
            )
            rejects += await publish(broker, settings, task)
            sent += 1
            if (now := time.monotonic()) - window >= 5:
                rate = (sent - window_sent) / (now - window)
                log.info("sent %d, %.1f/s, rejected %d", sent, rate, rejects)
                window, window_sent = now, sent
        log.info("done: %d tasks in %.1fs, rejected %d", sent, time.monotonic() - started, rejects)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, nargs="?")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )
    asyncio.run(run(args.path or settings.replays, args.limit, settings))


if __name__ == "__main__":
    main()
