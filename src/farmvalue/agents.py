"""Агенты Kaggriculture как MLflow-модели (pyfunc) в реестре kaggriculture-agent.

Модель агента это его исходники (main.py и соседние файлы) как артефакт плюс
обёртка `KaggleAgent`: predict(список observation) -> список action. Агенты
держат состояние между ходами в глобалах модуля, поэтому один загруженный
экземпляр модели соответствует одному игроку в одной партии.
"""

import runpy
import shutil
import sys
import tarfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlflow.pyfunc.model import PythonModel

from farmvalue.config import PipelineSettings

type Action = dict[str, Any]
type AgentFn = Callable[..., Action]

STARTER_SOURCE = (
    "from kaggle_environments.envs.kaggriculture.kaggriculture import starter_agent\n\n\n"
    "def agent(obs):\n    return starter_agent(obs)\n"
)


@dataclass(frozen=True)
class AgentSpec:
    name: str
    origin: str
    kind: str
    path: str
    notebook: str = ""

    def source(self, settings: PipelineSettings) -> Path | None:
        return None if self.path == "starter" else settings.kaggriculture_dir / self.path


AGENTS = (
    AgentSpec("g2a", "own", "сабмит G2A: G плюс гейт окна сброса", "subs/G2A/submission.tar.gz"),
    AgentSpec("g_mkt", "own", "сабмит G: рыночная ветка", "subs/G_mkt/main.py"),
    AgentSpec(
        "multi_route",
        "public",
        "выбор маршрута развития фермы",
        "sub/pub_flexonafft_multi_route_farming_agent/main.py",
        "kaggriculture-multi-route-farming-agent",
    ),
    AgentSpec(
        "market_smart",
        "public",
        "тайминг продаж по рынку",
        "sub/pub_tetsutani_market_smart_farming_kaggric/main.py",
        "market-smart-farming-kaggriculture",
    ),
    AgentSpec(
        "shop_router",
        "public",
        "реактивный роутер по магазинам города",
        "sub/pub_aurax7_shop_router_reactive_v7/main.py",
        "kaggriculture-shop-router-reactive-v7",
    ),
    AgentSpec(
        "replay_routing",
        "public",
        "проигрывание записанных партий с маршрутизацией",
        "sub/pub_thomastschinkel_replay_routing/main.py",
    ),
    AgentSpec(
        "fieldcraft_2900",
        "public",
        "компактная эвристика",
        "sub/pub_salemali7_2900/main.py",
        "kaggriculture-2900",
    ),
    AgentSpec("starter", "builtin", "стартовый агент из kaggle_environments", "starter"),
)
AGENTS_BY_NAME = {spec.name: spec for spec in AGENTS}


class Struct(dict[str, Any]):
    """dict с доступом через атрибуты, как observation в kaggle_environments."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e


def structify(value: Any) -> Any:
    if isinstance(value, dict):
        return Struct({k: structify(v) for k, v in value.items()})
    if isinstance(value, list):
        return [structify(v) for v in value]
    return value


def load_agent(directory: Path) -> AgentFn:
    """Выполнить main.py и взять последний callable, как это делает kaggle_environments."""
    sys.path.insert(0, str(directory))
    namespace = runpy.run_path(str(directory / "main.py"), run_name="agent_main")
    functions = [v for v in namespace.values() if callable(v)]
    if not functions:
        raise ValueError(f"{directory}/main.py defines no callable")
    fn: AgentFn = functions[-1]
    return fn


class KaggleAgent(PythonModel):
    def load_context(self, context: Any) -> None:
        self._agent = load_agent(Path(context.artifacts["agent"]))

    def act(self, obs: Any, config: Any = None) -> Action:
        args = [structify(obs), structify(config or {})]
        code = getattr(self._agent, "__code__", None)
        return self._agent(*args[: code.co_argcount if code else 1])

    # без аннотаций: по type hints MLflow строит схему входа, а observation не таблица
    def predict(self, context, model_input, params=None):  # type: ignore[no-untyped-def]  # noqa: ANN001, ANN201
        return [self.act(obs) for obs in model_input]


def package(spec: AgentSpec, settings: PipelineSettings, target: Path) -> Path:
    """Разложить исходники агента в каталог target с main.py в корне."""
    target.mkdir(parents=True, exist_ok=True)
    source = spec.source(settings)
    if source is None:
        (target / "main.py").write_text(STARTER_SOURCE, encoding="utf-8")
    elif source.name.endswith(".tar.gz"):
        with tarfile.open(source) as tar:
            tar.extractall(target, filter="data")
    else:
        for path in source.parent.iterdir():
            if path.is_file() and path.suffix in {".py", ".json"}:
                shutil.copy2(path, target / path.name)
    if not (target / "main.py").exists():
        raise FileNotFoundError(f"{spec.name}: main.py not found in {source}")
    return target
