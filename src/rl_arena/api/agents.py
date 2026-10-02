"""Агенты: регистрация (в т.ч. «залить и посмотреть винрейт») и их эвал."""

import uuid

from fastapi import APIRouter, status

from rl_arena.api.deps import SessionDep
from rl_arena.domain import AgentKind, AgentPool
from rl_arena.schemas.league import AgentCreate, AgentRead, EvaluationReport
from rl_arena.services import agents as agents_service
from rl_arena.services import league as league_service

router = APIRouter(prefix="/agents", tags=["agents"])


@router.post("", status_code=status.HTTP_201_CREATED)
async def register_agent(payload: AgentCreate, session: SessionDep) -> AgentRead:
    """Зарегистрировать агента в тренировочном или held-out эвал-пуле."""
    agent = await agents_service.create_agent(session, payload)
    return AgentRead.model_validate(agent)


@router.get("")
async def list_agents(
    session: SessionDep,
    pool: AgentPool | None = None,
    kind: AgentKind | None = None,
    active: bool | None = None,
) -> list[AgentRead]:
    """Список агентов с фильтрами по пулу, роли и активности."""
    agents = await agents_service.list_agents(session, pool, kind, active)
    return [AgentRead.model_validate(agent) for agent in agents]


@router.get("/{agent_id}")
async def get_agent(agent_id: uuid.UUID, session: SessionDep) -> AgentRead:
    """Карточка агента."""
    return AgentRead.model_validate(await agents_service.get_agent(session, agent_id))


@router.get("/{agent_id}/evaluation")
async def evaluate_agent(agent_id: uuid.UUID, session: SessionDep) -> EvaluationReport:
    """Винрейт агента против held-out пула с 95% доверительным интервалом."""
    return await league_service.evaluate_agent(session, agent_id)
