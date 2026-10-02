"""Точка сборки приложения.

Запуск: `uv run uvicorn rl_arena.main:app --reload`
"""

import logging
import re
import time
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI, Request, Response
from redis.asyncio import Redis

from rl_arena.api import agents, jobs, league, matches, system
from rl_arena.api.errors import register_error_handlers
from rl_arena.core.config import Settings, get_settings
from rl_arena.core.logging import REQUEST_ID, setup_logging
from rl_arena.core.version import get_build_info
from rl_arena.db.session import create_engine, create_sessionmaker

log = logging.getLogger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"
_VALID_REQUEST_ID = re.compile(r"^[A-Za-z0-9-]{1,64}$")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Создать подключения к БД и Redis при старте, закрыть при остановке."""
    settings: Settings = app.state.settings
    app.state.engine = create_engine(settings.database_url)
    app.state.sessionmaker = create_sessionmaker(app.state.engine)
    app.state.redis = Redis.from_url(settings.redis_url, socket_timeout=settings.health_timeout_s)
    build = get_build_info()
    log.info(
        "%s %s started (commit=%s, env=%s)",
        build.name,
        build.version,
        build.commit,
        settings.environment,
    )
    try:
        yield
    finally:
        await app.state.redis.aclose()
        await app.state.engine.dispose()
        log.info("%s stopped", build.name)


async def log_requests(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Связать все логи запроса одним request_id и залогировать итог запроса.

    Входящий `X-Request-ID` (от балансировщика или другого сервиса) переиспользуется,
    чтобы трассировать запрос между сервисами; мусорный — заменяется своим.
    """
    incoming = request.headers.get(REQUEST_ID_HEADER, "")
    request_id = incoming if _VALID_REQUEST_ID.match(incoming) else uuid.uuid4().hex[:12]
    token = REQUEST_ID.set(request_id)
    start = time.perf_counter()
    try:
        response = await call_next(request)
        log.info(
            "%s %s -> %d (%.1f ms)",
            request.method,
            request.url.path,
            response.status_code,
            (time.perf_counter() - start) * 1000,
        )
    except Exception:
        log.exception("%s %s -> unhandled error", request.method, request.url.path)
        raise
    finally:
        REQUEST_ID.reset(token)
    response.headers[REQUEST_ID_HEADER] = request_id
    return response


def create_app(settings: Settings | None = None) -> FastAPI:
    """Собрать приложение FastAPI."""
    settings = settings or get_settings()
    setup_logging(settings.log_level, settings.log_format)
    app = FastAPI(title=settings.app_name, version=get_build_info().version, lifespan=lifespan)
    app.state.settings = settings
    app.middleware("http")(log_requests)
    register_error_handlers(app)

    api_v1 = APIRouter(prefix="/api/v1")
    for module in (system, agents, matches, league, jobs):
        api_v1.include_router(module.router)
    app.include_router(system.probes_router)
    app.include_router(api_v1)
    return app


app = create_app()
