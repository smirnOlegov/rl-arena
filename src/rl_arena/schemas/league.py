"""Контракты доменных эндпоинтов: агенты, матчи, эвал, лига, задачи."""

import uuid
from datetime import datetime
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rl_arena.domain import (
    AgentKind,
    AgentPool,
    JobKind,
    JobStatus,
    MatchPurpose,
    MatchResult,
)


class AgentCreate(BaseModel):
    """Регистрация агента в лиге."""

    name: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._-]+$")
    kind: AgentKind
    pool: AgentPool = AgentPool.training
    checkpoint_uri: str = Field(min_length=1, max_length=1024)
    parent_id: uuid.UUID | None = None


class AgentRead(BaseModel):
    """Агент лиги."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    kind: AgentKind
    pool: AgentPool
    checkpoint_uri: str
    parent_id: uuid.UUID | None
    active: bool
    created_at: datetime


class MatchCreate(BaseModel):
    """Результат матча с точки зрения `agent_id`."""

    agent_id: uuid.UUID
    opponent_id: uuid.UUID
    result: MatchResult
    purpose: MatchPurpose
    replay_uri: str | None = Field(default=None, max_length=1024)

    @model_validator(mode="after")
    def _distinct_players(self) -> Self:
        if self.agent_id == self.opponent_id:
            raise ValueError("agent cannot play against itself in a recorded match")
        return self


class MatchRead(MatchCreate):
    """Записанный матч."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    created_at: datetime


class RecordStats(BaseModel):
    """Сводка игр и винрейт с 95% доверительным интервалом Уилсона."""

    games: int
    wins: int
    draws: int
    losses: int
    winrate: float | None
    ci_low: float
    ci_high: float


class OpponentStats(RecordStats):
    """Сводка против конкретного оппонента."""

    opponent_id: uuid.UUID
    opponent_name: str


class EvaluationReport(BaseModel):
    """Эвал агента против held-out пула (только матчи с purpose=evaluation)."""

    agent: AgentRead
    overall: RecordStats
    opponents: list[OpponentStats]


class LeaderboardEntry(BaseModel):
    """Строка лидерборда: ранжирование по нижней границе CI — честнее среднего."""

    rank: int
    agent: AgentRead
    stats: RecordStats


class OpponentPick(BaseModel):
    """Выбранный PFSP тренировочный оппонент."""

    agent_id: uuid.UUID
    opponent: AgentRead
    weighting: str
    winrate_vs_opponent: float | None


class ExploiterResampleRequest(BaseModel):
    """Тик эпохи лиги: при необходимости пересоздать эксплоитеров."""

    epoch: int = Field(ge=0)
    sft_agent_id: uuid.UUID


class ExploiterResampleReport(BaseModel):
    """Итог ресемплинга эксплоитеров."""

    epoch: int
    resampled: bool
    retired: list[uuid.UUID]
    created: list[AgentRead]
    jobs: list[uuid.UUID]


class JobCreate(BaseModel):
    """Поставить задачу GPU-воркерам."""

    kind: JobKind
    agent_id: uuid.UUID | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class JobRead(BaseModel):
    """Задача и её статус."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    kind: JobKind
    status: JobStatus
    agent_id: uuid.UUID | None
    params: dict[str, Any]
    created_at: datetime
