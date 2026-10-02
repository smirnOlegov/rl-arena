"""Задачи для GPU-воркеров: SFT по реплеям, self-play, эксплоитеры, эвал."""

import uuid

from fastapi import APIRouter, status

from rl_arena.api.deps import RedisDep, SessionDep, SettingsDep
from rl_arena.schemas.league import JobCreate, JobRead
from rl_arena.services import jobs as jobs_service

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.post("", status_code=status.HTTP_202_ACCEPTED)
async def submit_job(
    payload: JobCreate, session: SessionDep, redis: RedisDep, settings: SettingsDep
) -> JobRead:
    """Поставить задачу в очередь; выполнение асинхронное, статус — через GET."""
    job = await jobs_service.create_job(
        session, redis, settings.jobs_stream, payload.kind, payload.agent_id, payload.params
    )
    return JobRead.model_validate(job)


@router.get("/{job_id}")
async def get_job(job_id: uuid.UUID, session: SessionDep) -> JobRead:
    """Статус задачи."""
    return JobRead.model_validate(await jobs_service.get_job(session, job_id))
