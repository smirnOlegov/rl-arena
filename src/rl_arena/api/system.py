"""Служебные эндпоинты: liveness, версия, end-to-end health.

`/healthz` намеренно живёт вне `/api/v1`: это контракт с инфраструктурой
(Docker HEALTHCHECK, k8s livenessProbe, балансировщик), а не часть
версионируемого бизнес-API. Он не должен меняться при выходе `/api/v2`, и
инфраструктура не должна знать о версиях API. Он же не ходит во внешние
зависимости: упавшая БД — не повод перезапускать живой процесс.
"""

import logging

from fastapi import APIRouter, Response, status

from rl_arena.api.deps import EngineDep, RedisDep, SettingsDep
from rl_arena.core.version import BuildInfo, get_build_info
from rl_arena.schemas.system import HealthReport, LivenessResponse, ReportStatus
from rl_arena.services import health as health_service

log = logging.getLogger(__name__)

probes_router = APIRouter(tags=["probes"])
router = APIRouter(tags=["system"])


@probes_router.get("/healthz")
async def liveness() -> LivenessResponse:
    """Процесс жив и обслуживает запросы."""
    return LivenessResponse()


@router.get("/version")
async def app_version() -> BuildInfo:
    """Вернуть версию приложения (из метаданных пакета), коммит и дату сборки."""
    return get_build_info()


@router.get(
    "/health",
    responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"model": HealthReport}},
)
async def health(
    engine: EngineDep, redis: RedisDep, settings: SettingsDep, response: Response
) -> HealthReport:
    """Проверить все внешние компоненты: статус, версия, время ответа каждого."""
    report = await health_service.build_report(
        engine, redis, get_build_info().version, settings.health_timeout_s
    )
    if report.status is not ReportStatus.ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        log.warning("Health degraded: %s", report.model_dump_json(include={"components"}))
    return report
