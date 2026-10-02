"""Задачи для GPU-воркеров: учёт в Postgres, доставка через Redis Stream.

Веб-сервис — control plane: он решает, что и когда учить (SFT по свежим
реплеям, self-play, эксплоитеры, эвал), а сами вычисления делают внешние
воркеры, читающие stream через consumer group.
"""

import logging
import uuid
from typing import Any

from redis.asyncio import Redis
from redis.exceptions import RedisError
from sqlalchemy.ext.asyncio import AsyncSession

from rl_arena.db.models import Job
from rl_arena.domain import JobKind
from rl_arena.exceptions import NotFoundError, QueueUnavailableError
from rl_arena.services.agents import get_agent

log = logging.getLogger(__name__)


async def publish_job(redis: Redis, stream: str, job: Job) -> None:
    """Отправить задачу в Redis Stream."""
    fields: dict[Any, Any] = {"job_id": str(job.id), "kind": job.kind.value}
    try:
        await redis.xadd(stream, fields)
    except RedisError as exc:
        # Задача уже закоммичена в БД со статусом queued — её можно переотправить.
        raise QueueUnavailableError(f"job {job.id} saved but not published") from exc


async def create_job(
    session: AsyncSession,
    redis: Redis,
    stream: str,
    kind: JobKind,
    agent_id: uuid.UUID | None,
    params: dict[str, Any],
) -> Job:
    """Сохранить задачу и отправить её воркерам."""
    if agent_id is not None:
        await get_agent(session, agent_id)
    job = Job(kind=kind, agent_id=agent_id, params=params)
    session.add(job)
    await session.commit()
    await publish_job(redis, stream, job)
    log.info("Job queued: %s %s (agent=%s)", job.kind, job.id, agent_id)
    return job


async def get_job(session: AsyncSession, job_id: uuid.UUID) -> Job:
    """Вернуть задачу или бросить `NotFoundError`."""
    job = await session.get(Job, job_id)
    if job is None:
        raise NotFoundError(f"job {job_id} not found")
    return job
