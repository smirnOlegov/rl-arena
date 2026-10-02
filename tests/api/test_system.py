"""Служебные эндпоинты: /healthz, /api/v1/version, /api/v1/health."""

import tomllib
from importlib.metadata import version
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient
from redis.asyncio import Redis

from rl_arena.core import version as version_module
from tests.conftest import REAL_REDIS_URL, UNREACHABLE_REDIS_URL

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


async def test_liveness_is_outside_versioned_api(client) -> None:
    response = await client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert (await client.get("/api/v1/healthz")).status_code == 404


async def test_version_matches_pyproject(client) -> None:
    """Версия не захардкожена: она та же, что в pyproject.toml и в метаданных пакета."""
    expected = tomllib.loads(PYPROJECT.read_text())["project"]["version"]
    body = (await client.get("/api/v1/version")).json()
    assert body["version"] == expected == version("rl-arena")
    assert body["name"] == "rl-arena"
    assert body["api_version"] == "v1"


async def test_version_reports_build_metadata_from_env(client, monkeypatch) -> None:
    monkeypatch.setenv("GIT_SHA", "abc1234")
    monkeypatch.setenv("BUILD_DATE", "2026-10-02T12:00:00Z")
    version_module.get_build_info.cache_clear()
    try:
        body = (await client.get("/api/v1/version")).json()
    finally:
        version_module.get_build_info.cache_clear()
    assert body["commit"] == "abc1234"
    assert body["build_date"] == "2026-10-02T12:00:00Z"


async def test_health_ok_reports_versions_and_latency(client) -> None:
    response = await client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"] == version("rl-arena")
    assert set(body["components"]) == {"database", "redis"}
    for component in body["components"].values():
        assert component["status"] == "up"
        assert component["version"]
        assert component["latency_ms"] >= 0
        assert component["error"] is None
    assert body["latency_ms"] >= 0


@pytest.mark.skipif(REAL_REDIS_URL is None, reason="needs real Redis")
async def test_health_reports_real_redis_version(client) -> None:
    body = (await client.get("/api/v1/health")).json()
    assert body["components"]["redis"]["version"][0].isdigit()


async def test_health_degraded_when_redis_down(app, client) -> None:
    await app.state.redis.aclose()
    app.state.redis = Redis.from_url(UNREACHABLE_REDIS_URL)
    response = await client.get("/api/v1/health")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["components"]["redis"]["status"] == "down"
    assert body["components"]["redis"]["error"]
    assert body["components"]["database"]["status"] == "up"


async def test_request_id_generated_when_absent(client) -> None:
    response = await client.get("/healthz")
    assert len(response.headers["X-Request-ID"]) == 12


async def test_request_id_propagated_from_caller(client) -> None:
    response = await client.get("/healthz", headers={"X-Request-ID": "trace-42"})
    assert response.headers["X-Request-ID"] == "trace-42"


async def test_request_id_rejects_garbage(client) -> None:
    response = await client.get("/healthz", headers={"X-Request-ID": "bad id\n"})
    assert response.headers["X-Request-ID"] != "bad id\n"


async def test_openapi_lists_all_endpoints(client) -> None:
    paths = (await client.get("/openapi.json")).json()["paths"]
    assert {"/healthz", "/api/v1/version", "/api/v1/health"} <= set(paths)


async def test_unhandled_error_is_logged_with_request_id(app, caplog) -> None:
    async def explode() -> None:
        raise RuntimeError("boom")

    app.add_api_route("/explode", explode)
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as http:
        response = await http.get("/explode", headers={"X-Request-ID": "crash-1"})
    assert response.status_code == 500
    [record] = [r for r in caplog.records if "unhandled error" in r.getMessage()]
    assert record.request_id == "crash-1"
    assert record.exc_info is not None
