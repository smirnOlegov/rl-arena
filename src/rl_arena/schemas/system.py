"""Контракты служебных эндпоинтов: liveness и end-to-end health."""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class LivenessResponse(BaseModel):
    """Ответ `/healthz`."""

    status: str = "ok"


class ComponentStatus(StrEnum):
    """Статус внешнего компонента."""

    up = "up"
    down = "down"


class ReportStatus(StrEnum):
    """Сводный статус сервиса."""

    ok = "ok"
    degraded = "degraded"


class ComponentHealth(BaseModel):
    """Результат проверки одного компонента."""

    status: ComponentStatus
    version: str | None = Field(default=None, description="Версия, которую сообщил сам компонент.")
    latency_ms: float = Field(description="Время полного round-trip запроса к компоненту.")
    error: str | None = Field(default=None, description="Класс ошибки; детали — только в логах.")


class HealthReport(BaseModel):
    """End-to-end health-отчёт `/api/v1/health`."""

    status: ReportStatus
    version: str
    checked_at: datetime
    latency_ms: float = Field(
        description="Время всей проверки (компоненты опрашиваются параллельно)."
    )
    components: dict[str, ComponentHealth]
