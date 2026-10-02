"""Управление лигой: матчмейкинг, лидерборд, тик эпохи для эксплоитеров."""

import random
import uuid

from fastapi import APIRouter, Query

from rl_arena.api.deps import RedisDep, SessionDep, SettingsDep
from rl_arena.league.pfsp import PfspWeighting
from rl_arena.schemas.league import (
    ExploiterResampleReport,
    ExploiterResampleRequest,
    LeaderboardEntry,
    OpponentPick,
)
from rl_arena.services import league as league_service

router = APIRouter(prefix="/league", tags=["league"])


@router.get("/opponent")
async def pick_opponent(
    session: SessionDep,
    agent_id: uuid.UUID,
    weighting: PfspWeighting = PfspWeighting.hard,
    seed: int | None = Query(default=None, description="Для воспроизводимого выбора."),
) -> OpponentPick:
    """Выбрать тренировочного оппонента по PFSP (эвал-пул исключён)."""
    rng = random.Random(seed)  # noqa: S311 — не криптография, нужен воспроизводимый сэмплинг
    return await league_service.pick_training_opponent(session, agent_id, weighting, rng)


@router.get("/leaderboard")
async def leaderboard(
    session: SessionDep, top: int = Query(default=10, ge=1, le=100)
) -> list[LeaderboardEntry]:
    """Топ агентов по нижней границе CI винрейта на held-out эвале."""
    return await league_service.leaderboard(session, top)


@router.post("/exploiters/resample")
async def resample_exploiters(
    payload: ExploiterResampleRequest,
    session: SessionDep,
    redis: RedisDep,
    settings: SettingsDep,
) -> ExploiterResampleReport:
    """Тик эпохи: каждые N эпох пересоздать эксплоитеров от SFT-чекпоинта."""
    return await league_service.resample_exploiters(
        session, redis, settings, payload.epoch, payload.sft_agent_id
    )
