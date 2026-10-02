"""Регистрация и выборка агентов."""

import logging
import uuid

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from rl_arena.db.models import Agent
from rl_arena.domain import AgentKind, AgentPool
from rl_arena.exceptions import ConflictError, NotFoundError
from rl_arena.schemas.league import AgentCreate

log = logging.getLogger(__name__)


async def get_agent(session: AsyncSession, agent_id: uuid.UUID) -> Agent:
    """Вернуть агента или бросить `NotFoundError`."""
    agent = await session.get(Agent, agent_id)
    if agent is None:
        raise NotFoundError(f"agent {agent_id} not found")
    return agent


async def create_agent(session: AsyncSession, payload: AgentCreate) -> Agent:
    """Зарегистрировать агента; имя уникально, родитель должен существовать."""
    if payload.parent_id is not None:
        await get_agent(session, payload.parent_id)
    agent = Agent(**payload.model_dump())
    session.add(agent)
    try:
        await session.commit()
    except IntegrityError as exc:
        await session.rollback()
        raise ConflictError(f"agent named {payload.name!r} already exists") from exc
    log.info("Agent registered: %s (%s, pool=%s)", agent.name, agent.kind, agent.pool)
    return agent


async def list_agents(
    session: AsyncSession, pool: AgentPool | None, kind: AgentKind | None, active: bool | None
) -> list[Agent]:
    """Вернуть агентов с опциональной фильтрацией, от новых к старым."""
    query = select(Agent).order_by(Agent.created_at.desc())
    if pool is not None:
        query = query.where(Agent.pool == pool)
    if kind is not None:
        query = query.where(Agent.kind == kind)
    if active is not None:
        query = query.where(Agent.active == active)
    return list(await session.scalars(query))
