"""Приём результатов матчей от воркеров."""

from fastapi import APIRouter, status

from rl_arena.api.deps import SessionDep
from rl_arena.schemas.league import MatchCreate, MatchRead
from rl_arena.services import matches as matches_service

router = APIRouter(prefix="/matches", tags=["matches"])


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    responses={status.HTTP_409_CONFLICT: {"description": "Матч смешивает обучение и эвал"}},
)
async def record_match(payload: MatchCreate, session: SessionDep) -> MatchRead:
    """Записать результат матча; матчи, ведущие к утечке эвала, отклоняются с 409."""
    match = await matches_service.record_match(session, payload)
    return MatchRead.model_validate(match)
