"""Воркер: скоринг задач из RabbitMQ моделью из MLflow Model Registry.

Запуск: `uv run uvicorn --factory scoring.worker:create_app --port 8001 --no-access-log`
"""

import asyncio
import contextvars
import functools
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Any

from faststream import AckPolicy, BaseMiddleware
from faststream.asgi import AsgiFastStream, make_ping_asgi
from faststream.message import StreamMessage
from faststream.rabbit import Channel, RabbitBroker
from faststream.rabbit.opentelemetry import RabbitTelemetryMiddleware
from opentelemetry import metrics, trace

from farmvalue.service.model import LoadedModel, load_from_registry
from scoring.config import Settings, get_settings
from scoring.db import create_schema, make_engine, make_sessionmaker, save_result
from scoring.logs import configure_logging, current_task_id
from scoring.telemetry import setup_telemetry, shutdown_telemetry
from scoring.topology import Task, declare, tasks_queue

log = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)
meter = metrics.get_meter(__name__)

# Время задачи от публикации до записи результата: при полной очереди из 100 задач
# и 15-30 задачах/с это 3-7 с, без лимита очереди — минуты
TASK_DURATION_BUCKETS_S = (0.5, 1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 60, 120, 300)
# Inference — это INFERENCE_DELAY_S плюс 40-100 мс на модель, когда заняты все 4 потока пула.
# При 0.2 с границы идут через 10 мс на 0.24-0.28 с, где лежит большая часть замеров
INFERENCE_BUCKET_FACTORS = (1.0, 1.1, 1.2, 1.25, 1.3, 1.35, 1.4, 1.5, 1.75, 2, 3, 5)

tasks_total = meter.create_counter(
    "scoring.worker.tasks", unit="{task}", description="Обработанные доставки по результату"
)
redeliveries_total = meter.create_counter(
    "scoring.worker.redeliveries", unit="{delivery}", description="Повторные доставки задач"
)
tasks_in_progress = meter.create_up_down_counter(
    "scoring.worker.tasks_in_progress", unit="{task}", description="Задачи в обработке"
)
task_duration = meter.create_histogram(
    "scoring.task.duration",
    unit="s",
    description="От публикации задачи до записи результата",
    explicit_bucket_boundaries_advisory=TASK_DURATION_BUCKETS_S,
)


def inference_buckets(delay_s: float) -> list[float]:
    # Без задержки (тесты) остаётся только время самой модели, от миллисекунд
    base = delay_s or 0.01
    return [round(base * factor, 4) for factor in INFERENCE_BUCKET_FACTORS]


def create_app(
    settings: Settings | None = None, model: LoadedModel | None = None
) -> AsgiFastStream:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.otel_service_name)
    setup_telemetry(settings)
    inference_duration = meter.create_histogram(
        "scoring.inference.duration",
        unit="s",
        description="Время inference в пуле потоков",
        explicit_bucket_boundaries_advisory=inference_buckets(settings.inference_delay_s),
    )
    delivery_limit = settings.delivery_limit
    in_flight: set[str] = set()

    class TaskTracking(BaseMiddleware):
        """task_id в логах, номер попытки в спане, учёт ошибок и перевода в DLQ."""

        async def consume_scope(self, call_next: Any, msg: StreamMessage[Any]) -> Any:
            task_id = msg.correlation_id
            # quorum-очередь считает возвраты сама; первая доставка приходит без заголовка
            attempt = int(msg.headers.get("x-delivery-count", 0)) + 1
            span = trace.get_current_span()
            span.set_attributes({"task.id": task_id, "messaging.delivery.attempt": attempt})
            token = current_task_id.set(task_id)
            in_flight.add(task_id)
            tasks_in_progress.add(1)
            if attempt > 1:
                redeliveries_total.add(1)
            try:
                result = await call_next(msg)
            except Exception:
                # после delivery_limit возвратов брокер отправит этот nack в DLX
                dead = attempt > delivery_limit
                log.exception(
                    "task failed", extra={"attempt": attempt, "delivery_limit": delivery_limit}
                )
                if dead:
                    log.error("task moved to DLQ", extra={"attempt": attempt})
                    span.add_event("moved to DLQ", {"messaging.delivery.attempt": attempt})
                tasks_total.add(1, {"result": "dlq" if dead else "error"})
                raise
            else:
                tasks_total.add(1, {"result": "ok"})
                return result
            finally:
                tasks_in_progress.add(-1)
                in_flight.discard(task_id)
                current_task_id.reset(token)

    broker = RabbitBroker(
        settings.rabbit_url,
        default_channel=Channel(prefetch_count=settings.prefetch_count),
        graceful_timeout=settings.graceful_timeout_s,
        middlewares=[RabbitTelemetryMiddleware(), TaskTracking],
        # Received/Processed и повтор ошибки FastStream пишет вне спана обработки,
        # без task_id, trace_id и номера попытки. Записи о задаче делает TaskTracking
        logger=None,
    )
    engine = make_engine(settings.database_url)
    sessions = make_sessionmaker(engine)
    pool = ThreadPoolExecutor(settings.worker_threads, thread_name_prefix="inference")
    loaded = model or load_from_registry(
        settings.mlflow_tracking_uri, settings.model_name, settings.model_alias
    )
    model_attributes = {
        "model.name": loaded.info.name,
        "model.alias": loaded.info.alias,
        "model.version": loaded.info.version,
    }

    def predict(observation: dict[str, Any]) -> float:
        with tracer.start_as_current_span("inference", attributes=model_attributes):
            started = time.perf_counter()
            time.sleep(settings.inference_delay_s)
            win_prob = loaded.win_probability([observation])[0]
            inference_duration.record(time.perf_counter() - started)
            return win_prob

    @broker.subscriber(tasks_queue(settings), ack_policy=AckPolicy.NACK_ON_ERROR)
    async def score(task: Task) -> None:
        queue_wait = datetime.now(UTC) - task.enqueued_at
        trace.get_current_span().set_attribute("task.queue_wait_s", queue_wait.total_seconds())
        # run_in_executor не переносит contextvars в поток пула: без копии контекста
        # спан inference стал бы корневым, а логи потеряли бы task_id
        context = contextvars.copy_context()
        loop = asyncio.get_running_loop()
        win_prob = await loop.run_in_executor(
            pool, functools.partial(context.run, predict, task.observation)
        )
        with tracer.start_as_current_span("save result"):
            async with sessions() as session:
                await save_result(session, task.id, win_prob, loaded.info.version, task.enqueued_at)
        duration = datetime.now(UTC) - task.enqueued_at
        task_duration.record(duration.total_seconds())
        log.info(
            "task scored",
            extra={
                "win_prob": round(win_prob, 4),
                "model_version": loaded.info.version,
                "queue_wait_s": round(queue_wait.total_seconds(), 3),
                "duration_s": round(duration.total_seconds(), 3),
            },
        )

    async def startup() -> None:
        await create_schema(engine)
        await broker.connect()
        await declare(broker, settings)

    async def begin_shutdown() -> None:
        log.info(
            "graceful shutdown started",
            extra={"in_flight": sorted(in_flight), "timeout_s": settings.graceful_timeout_s},
        )

    async def shutdown() -> None:
        pool.shutdown(wait=True)
        await engine.dispose()
        log.info("graceful shutdown finished", extra={"in_flight": sorted(in_flight)})
        shutdown_telemetry()

    return AsgiFastStream(
        broker,
        asgi_routes=[("/healthz", make_ping_asgi(broker, timeout=1.0))],
        logger=logging.getLogger("faststream"),
        on_startup=[startup],
        on_shutdown=[begin_shutdown],
        after_shutdown=[shutdown],
    )
