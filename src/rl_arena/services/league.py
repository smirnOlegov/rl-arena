"""Логика лиги: PFSP-матчмейкинг, held-out эвал, лидерборд, ресемплинг эксплоитеров."""

import logging
import random
import uuid

from redis.asyncio import Redis
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from rl_arena.core.config import Settings
from rl_arena.db.models import Agent, Job
from rl_arena.domain import AgentKind, AgentPool, JobKind, MatchPurpose
from rl_arena.exceptions import ConflictError, EvalLeakageError, NoOpponentsError
from rl_arena.league.exploiters import exploiter_name, is_resample_epoch
from rl_arena.league.pfsp import PfspWeighting, sample_opponent
from rl_arena.league.stats import Record, wilson_interval
from rl_arena.schemas.league import (
    AgentRead,
    EvaluationReport,
    ExploiterResampleReport,
    LeaderboardEntry,
    OpponentPick,
    OpponentStats,
    RecordStats,
)
from rl_arena.services.agents import get_agent
from rl_arena.services.jobs import publish_job
from rl_arena.services.matches import evaluation_records, records_by_opponent

log = logging.getLogger(__name__)


def to_stats(record: Record) -> RecordStats:
    """Превратить сводку в контракт API с доверительным интервалом."""
    ci_low, ci_high = wilson_interval(record.score or 0.0, record.games)
    return RecordStats(
        games=record.games,
        wins=record.wins,
        draws=record.draws,
        losses=record.losses,
        winrate=record.score,
        ci_low=ci_low,
        ci_high=ci_high,
    )


async def pick_training_opponent(
    session: AsyncSession, agent_id: uuid.UUID, weighting: PfspWeighting, rng: random.Random
) -> OpponentPick:
    """Выбрать тренировочного оппонента по PFSP.

    Кандидаты — только активные агенты тренировочного пула: held-out эвал-агенты
    в выборку не попадают в принципе.
    """
    agent = await get_agent(session, agent_id)
    if agent.pool is AgentPool.evaluation:
        raise EvalLeakageError(f"agent {agent.name!r} is a held-out evaluator and does not train")
    candidates = list(
        await session.scalars(
            select(Agent).where(
                Agent.pool == AgentPool.training, Agent.active.is_(True), Agent.id != agent.id
            )
        )
    )
    if not candidates:
        raise NoOpponentsError("training pool has no other active agents")
    records = await records_by_opponent(session, agent.id, MatchPurpose.training)
    winrates = {
        opponent_id: score
        for opponent_id, record in records.items()
        if (score := record.score) is not None
    }
    by_id = {candidate.id: candidate for candidate in candidates}
    opponent_id = sample_opponent(list(by_id), winrates, weighting, rng)
    return OpponentPick(
        agent_id=agent.id,
        opponent=AgentRead.model_validate(by_id[opponent_id]),
        weighting=weighting.value,
        winrate_vs_opponent=winrates.get(opponent_id),
    )


async def evaluate_agent(session: AsyncSession, agent_id: uuid.UUID) -> EvaluationReport:
    """Посчитать винрейт агента против held-out пула с разбивкой по оппонентам."""
    agent = await get_agent(session, agent_id)
    records = await records_by_opponent(session, agent.id, MatchPurpose.evaluation)
    opponents = {a.id: a for a in await session.scalars(select(Agent).where(Agent.id.in_(records)))}
    overall = sum(records.values(), Record())
    per_opponent = [
        OpponentStats(
            opponent_id=opponent_id,
            opponent_name=opponents[opponent_id].name,
            **to_stats(record).model_dump(),
        )
        for opponent_id, record in sorted(records.items(), key=lambda item: opponents[item[0]].name)
    ]
    return EvaluationReport(
        agent=AgentRead.model_validate(agent), overall=to_stats(overall), opponents=per_opponent
    )


async def leaderboard(session: AsyncSession, top: int) -> list[LeaderboardEntry]:
    """Вернуть топ агентов по нижней границе CI эвал-винрейта.

    Сортировка по нижней границе, а не по среднему: агент с 3/3 победами не
    должен обгонять агента с 80% на тысяче игр. Так отбираются «наиболее
    релевантные подходы» для дальнейшего self-play.
    """
    records = await evaluation_records(session)
    agents = await session.scalars(select(Agent).where(Agent.id.in_(records)))
    ranked = sorted(
        ((agent, to_stats(records[agent.id])) for agent in agents),
        key=lambda pair: (pair[1].ci_low, pair[1].games),
        reverse=True,
    )
    return [
        LeaderboardEntry(rank=rank, agent=AgentRead.model_validate(agent), stats=stats)
        for rank, (agent, stats) in enumerate(ranked[:top], start=1)
    ]


async def resample_exploiters(
    session: AsyncSession,
    redis: Redis,
    settings: Settings,
    epoch: int,
    sft_agent_id: uuid.UUID,
) -> ExploiterResampleReport:
    """На эпохе ресемплинга: списать текущих эксплоитеров и завести новых от SFT.

    Повторный вызов для той же эпохи даёт `ConflictError` (имена новых
    эксплоитеров детерминированы эпохой) — тик эпохи идемпотентен.
    """
    sft = await get_agent(session, sft_agent_id)
    if sft.kind is not AgentKind.sft or sft.pool is not AgentPool.training:
        raise ConflictError(f"agent {sft.name!r} is not a training-pool SFT agent")
    if not is_resample_epoch(epoch, settings.exploiter_resample_every):
        return ExploiterResampleReport(
            epoch=epoch, resampled=False, retired=[], created=[], jobs=[]
        )

    retired = list(
        await session.scalars(
            select(Agent).where(Agent.kind == AgentKind.exploiter, Agent.active.is_(True))
        )
    )
    for exploiter in retired:
        exploiter.active = False
    created = [
        Agent(
            id=uuid.uuid4(),
            name=exploiter_name(epoch, index),
            kind=AgentKind.exploiter,
            pool=AgentPool.training,
            checkpoint_uri=sft.checkpoint_uri,
            parent_id=sft.id,
        )
        for index in range(settings.exploiters_per_league)
    ]
    jobs = [
        Job(
            id=uuid.uuid4(),
            kind=JobKind.exploiter_training,
            agent_id=exploiter.id,
            params={"epoch": epoch, "init_checkpoint": sft.checkpoint_uri},
        )
        for exploiter in created
    ]
    session.add_all(created)
    try:
        await session.flush()
    except IntegrityError as exc:
        await session.rollback()
        raise ConflictError(f"exploiters for epoch {epoch} already resampled") from exc
    session.add_all(jobs)
    await session.commit()
    for job in jobs:
        await publish_job(redis, settings.jobs_stream, job)
    log.info("Epoch %d: retired %d exploiters, created %d", epoch, len(retired), len(created))
    return ExploiterResampleReport(
        epoch=epoch,
        resampled=True,
        retired=[exploiter.id for exploiter in retired],
        created=[AgentRead.model_validate(exploiter) for exploiter in created],
        jobs=[job.id for job in jobs],
    )
