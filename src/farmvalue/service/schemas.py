"""Контракты API."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Observation(BaseModel):
    """Observation Kaggriculture в формате kaggle_environments (лишние поля допустимы)."""

    model_config = ConfigDict(extra="allow")

    day: int = Field(ge=0, le=30)
    hour: int = Field(default=0, ge=0, le=23)
    farms: list[dict[str, Any]] = Field(min_length=2, max_length=2)
    market: dict[str, Any] = Field(default_factory=dict)
    town: dict[str, Any] = Field(default_factory=dict)
    player: int | None = Field(default=None, ge=0, le=1)


class ProcessRequest(BaseModel):
    observations: list[Observation] = Field(min_length=1, max_length=1024)


class Prediction(BaseModel):
    day: int
    hour: int
    win_prob_seat0: float
    win_prob_player: float | None = Field(
        description="Вероятность победы игрока из observation.player, если он указан."
    )


class ModelMeta(BaseModel):
    name: str
    alias: str
    version: str
    run_id: str
    model_id: str | None


class ProcessResponse(BaseModel):
    model: ModelMeta
    predictions: list[Prediction]
