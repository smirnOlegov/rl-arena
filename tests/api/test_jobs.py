"""Задачи для GPU-воркеров."""

import uuid

from redis.asyncio import Redis
from sqlalchemy import func, select

from rl_arena.db.models import Job
from tests.conftest import UNREACHABLE_REDIS_URL


async def test_submit_sft_job_publishes_to_stream(app, client, make_agent) -> None:
    sft = await make_agent(kind="sft")
    response = await client.post(
        "/api/v1/jobs",
        json={"kind": "sft", "agent_id": sft["id"], "params": {"replays_since": "2026-10-01"}},
    )
    assert response.status_code == 202
    job = response.json()
    assert job["status"] == "queued"

    entries = await app.state.redis.xrange(app.state.settings.jobs_stream)
    assert len(entries) == 1
    _, fields = entries[0]
    assert fields[b"job_id"].decode() == job["id"]
    assert fields[b"kind"] == b"sft"

    fetched = (await client.get(f"/api/v1/jobs/{job['id']}")).json()
    assert fetched["params"] == {"replays_since": "2026-10-01"}


async def test_submit_job_for_unknown_agent_is_404(client) -> None:
    response = await client.post(
        "/api/v1/jobs", json={"kind": "self_play", "agent_id": str(uuid.uuid4())}
    )
    assert response.status_code == 404


async def test_get_unknown_job_is_404(client) -> None:
    assert (await client.get(f"/api/v1/jobs/{uuid.uuid4()}")).status_code == 404


async def test_submit_job_when_queue_down_is_503_but_job_is_kept(app, client, session) -> None:
    await app.state.redis.aclose()
    app.state.redis = Redis.from_url(UNREACHABLE_REDIS_URL)
    response = await client.post("/api/v1/jobs", json={"kind": "evaluation"})
    assert response.status_code == 503
    assert response.json()["error"] == "queue_unavailable"
    assert await session.scalar(select(func.count()).select_from(Job)) == 1
