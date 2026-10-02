"""Health-сервис: таймауты, ошибки, недоступная БД."""

import asyncio

from sqlalchemy.ext.asyncio import create_async_engine

from rl_arena.schemas.system import ComponentStatus, ReportStatus
from rl_arena.services import health as health_service
from tests.conftest import FakeRedis


async def test_check_component_up_returns_version_and_latency() -> None:
    async def probe() -> str:
        return "1.2.3"

    result = await health_service.check_component("x", probe, timeout_s=1)
    assert result.status is ComponentStatus.up
    assert result.version == "1.2.3"
    assert result.latency_ms >= 0


async def test_check_component_times_out() -> None:
    async def hanging_probe() -> str:
        await asyncio.sleep(10)
        return "never"

    result = await health_service.check_component("slow", hanging_probe, timeout_s=0.05)
    assert result.status is ComponentStatus.down
    assert result.error == "TimeoutError"
    assert result.latency_ms < 1000


async def test_check_component_reports_error_class_only() -> None:
    async def broken_probe() -> str:
        raise ConnectionRefusedError("password=hunter2 host=10.0.0.1")

    result = await health_service.check_component("db", broken_probe, timeout_s=1)
    assert result.status is ComponentStatus.down
    assert result.error == "ConnectionRefusedError"
    assert "hunter2" not in result.model_dump_json()


async def test_report_degraded_when_database_unreachable() -> None:
    engine = create_async_engine("postgresql+asyncpg://u:p@127.0.0.1:1/nope")
    try:
        report = await health_service.build_report(engine, FakeRedis(), "0.0.0", timeout_s=2)
    finally:
        await engine.dispose()
    assert report.status is ReportStatus.degraded
    assert report.components["database"].status is ComponentStatus.down
    assert report.components["redis"].status is ComponentStatus.up
