"""End-to-end health-check внешних компонентов.

Каждый компонент проверяется реальным запросом (а не «пул создан»), который
заодно возвращает версию компонента. Проверки идут параллельно и каждая
ограничена таймаутом — один зависший компонент не вешает весь отчёт.
"""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

from redis.asyncio import Redis
from redis.exceptions import RedisError
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine

from rl_arena.schemas.system import ComponentHealth, ComponentStatus, HealthReport, ReportStatus

log = logging.getLogger(__name__)

# Запрос версии зависит от диалекта: в проде Postgres, в быстрых тестах SQLite.
_DB_VERSION_QUERIES = {
    "postgresql": "SHOW server_version",
    "sqlite": "SELECT sqlite_version()",
}

type Probe = Callable[[], Awaitable[str]]


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


async def database_version(engine: AsyncEngine) -> str:
    """Выполнить запрос к БД и вернуть `<диалект> <версия>`."""
    query = _DB_VERSION_QUERIES[engine.dialect.name]
    async with engine.connect() as connection:
        server_version = await connection.scalar(text(query))
    return f"{engine.dialect.name} {server_version}"


async def redis_version(redis: Redis) -> str:
    """Запросить у Redis `INFO server` и вернуть версию сервера."""
    info = await redis.info("server")
    return str(info["redis_version"])


async def check_component(name: str, probe: Probe, timeout_s: float) -> ComponentHealth:
    """Выполнить проверку с таймаутом и замером времени; ошибку превратить в статус `down`."""
    start = time.perf_counter()
    try:
        async with asyncio.timeout(timeout_s):
            version = await probe()
    except (OSError, SQLAlchemyError, RedisError, TimeoutError) as exc:
        log.warning("Health-check %s failed: %r", name, exc)
        return ComponentHealth(
            status=ComponentStatus.down, latency_ms=_elapsed_ms(start), error=type(exc).__name__
        )
    return ComponentHealth(
        status=ComponentStatus.up, version=version, latency_ms=_elapsed_ms(start)
    )


async def build_report(
    engine: AsyncEngine, redis: Redis, app_version: str, timeout_s: float
) -> HealthReport:
    """Параллельно проверить все компоненты и собрать сводный отчёт."""
    start = time.perf_counter()
    probes: dict[str, Probe] = {
        "database": lambda: database_version(engine),
        "redis": lambda: redis_version(redis),
    }
    results = await asyncio.gather(
        *(check_component(name, probe, timeout_s) for name, probe in probes.items())
    )
    components = dict(zip(probes, results, strict=True))
    healthy = all(c.status is ComponentStatus.up for c in components.values())
    return HealthReport(
        status=ReportStatus.ok if healthy else ReportStatus.degraded,
        version=app_version,
        checked_at=datetime.now(UTC),
        latency_ms=_elapsed_ms(start),
        components=components,
    )
