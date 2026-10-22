"""Воркер: скоринг задач из RabbitMQ моделью из MLflow Model Registry.

Запуск: `uv run uvicorn --factory scoring.worker:create_app --port 8001`
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from faststream import AckPolicy
from faststream.asgi import AsgiFastStream, make_ping_asgi
from faststream.rabbit import Channel, RabbitBroker

from farmvalue.service.model import LoadedModel, load_from_registry
from scoring.config import Settings, get_settings
from scoring.db import create_schema, make_engine, make_sessionmaker, save_result
from scoring.topology import Task, declare, tasks_queue

log = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None, model: LoadedModel | None = None
) -> AsgiFastStream:
    settings = settings or get_settings()
    logging.basicConfig(
        level=settings.log_level, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )
    broker = RabbitBroker(
        settings.rabbit_url,
        default_channel=Channel(prefetch_count=settings.prefetch_count),
        graceful_timeout=settings.graceful_timeout_s,
    )
    engine = make_engine(settings.database_url)
    sessions = make_sessionmaker(engine)
    pool = ThreadPoolExecutor(settings.worker_threads, thread_name_prefix="inference")
    loaded = model or load_from_registry(
        settings.mlflow_tracking_uri, settings.model_name, settings.model_alias
    )

    def predict(observation: dict[str, Any]) -> float:
        time.sleep(settings.inference_delay_s)
        return loaded.win_probability([observation])[0]

    @broker.subscriber(tasks_queue(settings), ack_policy=AckPolicy.NACK_ON_ERROR)
    async def score(task: Task) -> None:
        win_prob = await asyncio.get_running_loop().run_in_executor(pool, predict, task.observation)
        async with sessions() as session:
            await save_result(session, task.id, win_prob, loaded.info.version, task.enqueued_at)

    async def startup() -> None:
        await create_schema(engine)
        await broker.connect()
        await declare(broker, settings)

    async def shutdown() -> None:
        pool.shutdown(wait=True)
        await engine.dispose()
        log.info("worker stopped")

    return AsgiFastStream(
        broker,
        asgi_routes=[("/healthz", make_ping_asgi(broker, timeout=1.0))],
        on_startup=[startup],
        after_shutdown=[shutdown],
    )
