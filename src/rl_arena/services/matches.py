"""Запись матчей и агрегирование статистики.

Здесь живёт главная защита от утечки эвала: эвал-агенты (held-out пул) не
могут участвовать в тренировочных матчах, а эвал-матч обязан идти против
эвал-пула. Значит, ни один тренировочный сигнал не содержит эвал-оппонентов,
и винрейт против них — честная оценка обобщения.
"""

import logging
import uuid
from collections import defaultdict

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from rl_arena.db.models import Agent, Match
from rl_arena.domain import AgentPool, MatchPurpose
from rl_arena.exceptions import EvalLeakageError
from rl_arena.league.stats import Record
from rl_arena.schemas.league import MatchCreate
from rl_arena.services.agents import get_agent

log = logging.getLogger(__name__)


def ensure_no_leakage(agent: Agent, opponent: Agent, purpose: MatchPurpose) -> None:
    """Проверить, что матч не смешивает обучение с held-out эвалом."""
    match purpose:
        case MatchPurpose.training:
            for player in (agent, opponent):
                if player.pool is AgentPool.evaluation:
                    raise EvalLeakageError(
                        f"evaluation-pool agent {player.name!r} cannot play training matches"
                    )
        case MatchPurpose.evaluation:
            if agent.pool is not AgentPool.training:
                raise EvalLeakageError(f"agent {agent.name!r} is a held-out evaluator itself")
            if opponent.pool is not AgentPool.evaluation:
                raise EvalLeakageError(
                    f"evaluation match requires a held-out opponent, {opponent.name!r} is not"
                )


async def record_match(session: AsyncSession, payload: MatchCreate) -> Match:
    """Записать матч после проверки на утечку эвала."""
    agent = await get_agent(session, payload.agent_id)
    opponent = await get_agent(session, payload.opponent_id)
    ensure_no_leakage(agent, opponent, payload.purpose)
    match = Match(**payload.model_dump())
    session.add(match)
    await session.commit()
    log.info(
        "Match recorded: %s %s vs %s (%s)", agent.name, match.result, opponent.name, match.purpose
    )
    return match


async def records_by_opponent(
    session: AsyncSession, agent_id: uuid.UUID, purpose: MatchPurpose
) -> dict[uuid.UUID, Record]:
    """Вернуть сводку агента против каждого оппонента, учитывая матчи в обе стороны."""
    as_agent = (
        select(Match.opponent_id, Match.result, func.count())
        .where(Match.agent_id == agent_id, Match.purpose == purpose)
        .group_by(Match.opponent_id, Match.result)
    )
    as_opponent = (
        select(Match.agent_id, Match.result, func.count())
        .where(Match.opponent_id == agent_id, Match.purpose == purpose)
        .group_by(Match.agent_id, Match.result)
    )
    records: defaultdict[uuid.UUID, Record] = defaultdict(Record)
    for opponent_id, result, count in await session.execute(as_agent):
        records[opponent_id] = records[opponent_id].add(result, count)
    for opponent_id, result, count in await session.execute(as_opponent):
        records[opponent_id] = records[opponent_id] + Record().add(result, count).flipped()
    return dict(records)


async def evaluation_records(session: AsyncSession) -> dict[uuid.UUID, Record]:
    """Вернуть суммарную эвал-сводку каждого агента (эвал всегда пишется от его лица)."""
    query = (
        select(Match.agent_id, Match.result, func.count())
        .where(Match.purpose == MatchPurpose.evaluation)
        .group_by(Match.agent_id, Match.result)
    )
    records: defaultdict[uuid.UUID, Record] = defaultdict(Record)
    for agent_id, result, count in await session.execute(query):
        records[agent_id] = records[agent_id].add(result, count)
    return dict(records)
