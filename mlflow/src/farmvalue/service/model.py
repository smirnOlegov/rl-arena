"""Загрузка модели из Model Registry и предсказание."""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import pandas as pd
from mlflow import MlflowClient
from mlflow.pyfunc import load_model

from farmvalue.features import FEATURES, Observation, snapshot_features

log = logging.getLogger(__name__)


class Predictor(Protocol):
    def predict(self, data: pd.DataFrame) -> Any: ...


@dataclass(frozen=True)
class ModelInfo:
    name: str
    alias: str
    version: str
    run_id: str
    model_id: str | None


@dataclass(frozen=True)
class LoadedModel:
    info: ModelInfo
    predictor: Predictor

    def win_probability(self, observations: Sequence[Observation]) -> list[float]:
        """Вероятность победы seat 0 для каждого снимка."""
        rows = pd.DataFrame([snapshot_features(obs) for obs in observations], columns=FEATURES)
        proba = self.predictor.predict(rows.astype("float64"))
        return [float(p) for p in proba[:, 1]]


def logged_model_id(source: str | None) -> str | None:
    """Версия, зарегистрированная из LoggedModel, ссылается на него как models:/m-<id>."""
    if source and source.startswith("models:/m-"):
        return source.removeprefix("models:/")
    return None


def load_from_registry(tracking_uri: str, name: str, alias: str) -> LoadedModel:
    """Разрешить alias в конкретную версию и загрузить именно её.

    Грузим `models:/name/version`, а не `models:/name@alias`: если alias
    переставят между двумя вызовами, метаданные и веса всё равно совпадут.
    """
    client = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
    version = client.get_model_version_by_alias(name, alias)
    info = ModelInfo(
        name=name,
        alias=alias,
        version=str(version.version),
        run_id=str(version.run_id),
        model_id=logged_model_id(version.source),
    )
    predictor = load_model(f"models:/{name}/{info.version}")
    log.info("loaded %s v%s (alias %s, run %s)", name, info.version, alias, info.run_id)
    return LoadedModel(info=info, predictor=predictor)
