"""Эксперимент kaggriculture-league: агенты как версии модели и их round-robin.

    uv run python -m farmvalue.league register     # агенты -> kaggriculture-agent
    uv run python -m farmvalue.league play --seeds 6
    uv run python -m farmvalue.league history      # старая лига из league/games*.jsonl

Партии играют модели, скачанные из реестра (`models:/kaggriculture-agent/<v>`).
Реплеи пишутся в data/league/replays и потом входят в датасет v2.
"""

import argparse
import gzip
import hashlib
import itertools
import json
import logging
import math
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.data.pandas_dataset import from_pandas

import farmvalue
from farmvalue import plots
from farmvalue.agents import AGENTS, AGENTS_BY_NAME, KaggleAgent, package
from farmvalue.config import PipelineSettings, get_settings
from farmvalue.dataset import physical_uri
from farmvalue.tracking import git_commit, use_experiment

log = logging.getLogger(__name__)

SEED0 = 30000
CHAMPION = "champion"


@dataclass(frozen=True)
class Game:
    a0: str
    a1: str
    seed: int
    reward0: float
    reward1: float
    ok: bool
    bank0: list[float]
    bank1: list[float]

    @property
    def score0(self) -> float:
        if self.reward0 == self.reward1:
            return 0.5
        return 1.0 if self.reward0 > self.reward1 else 0.0


def wilson(score: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    centre = (score + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(score * (1 - score) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def bradley_terry(
    results: Iterable[tuple[str, str, float]], agents: list[str], iters: int = 500
) -> dict[str, float]:
    """MM-итерации Bradley-Terry, ничья как пол-победы; шкала Эло со средним 1500."""
    wins: dict[tuple[str, str], float] = defaultdict(float)
    games: dict[tuple[str, str], int] = defaultdict(int)
    for a, b, score in results:
        wins[a, b] += score
        wins[b, a] += 1 - score
        games[a, b] += 1
        games[b, a] += 1
    strength = dict.fromkeys(agents, 1.0)
    for _ in range(iters):
        new = {}
        for a in agents:
            total = sum(wins[a, b] for b in agents if b != a)
            den = sum(
                games[a, b] / (strength[a] + strength[b]) for b in agents if b != a and games[a, b]
            )
            new[a] = max(total / den, 1e-9) if den else strength[a]
        strength = new
    mean = float(np.mean([math.log(s) for s in strength.values()]))
    return {a: 1500 + 400 / math.log(10) * (math.log(s) - mean) for a, s in strength.items()}


def leaderboard(
    results: list[tuple[str, str, float]], extra: pd.DataFrame | None = None
) -> pd.DataFrame:
    agents = sorted({a for a, _, _ in results} | {b for _, b, _ in results})
    ratings = bradley_terry(results, agents)
    rows = []
    for agent in agents:
        scores = [s for a, _, s in results if a == agent] + [
            1 - s for _, b, s in results if b == agent
        ]
        score = float(np.mean(scores))
        low, high = wilson(score, len(scores))
        rows.append(
            {
                "agent": agent,
                "games": len(scores),
                "score": score,
                "ci_low": low,
                "ci_high": high,
                "bt_rating": ratings[agent],
            }
        )
    board = pd.DataFrame(rows)
    if extra is not None:
        board = board.merge(extra, on="agent", how="left")
    return board.sort_values("bt_rating", ascending=False, ignore_index=True)


def source_sha256(directory: Path) -> str:
    h = hashlib.sha256()
    for path in sorted(directory.iterdir()):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def agent_code(target: Path) -> Path:
    """Минимальный пакет farmvalue для модели агента: обёртка и её зависимость."""
    package_dir = target / "farmvalue"
    package_dir.mkdir(parents=True)
    root = Path(farmvalue.__file__).parent
    for name in ("__init__.py", "agents.py", "config.py"):
        shutil.copy2(root / name, package_dir / name)
    return package_dir


def register_agents(settings: PipelineSettings) -> None:
    """Залогировать каждого агента в свой run и зарегистрировать версией модели."""
    use_experiment(settings, settings.league_experiment)
    client = MlflowClient()
    name = settings.agent_model
    if not client.search_registered_models(f"name = '{name}'"):
        client.create_registered_model(
            name, description="Агенты Kaggriculture: pyfunc, predict(observations) -> actions."
        )
    for spec in AGENTS:
        with tempfile.TemporaryDirectory() as tmp:
            directory = package(spec, settings, Path(tmp) / "agent")
            sha = source_sha256(directory)
            existing = client.search_model_versions(
                f"name = '{name}' and tags.agent = '{spec.name}' and tags.source_sha256 = '{sha}'"
            )
            if existing:
                log.info("%s already registered as v%s", spec.name, existing[0].version)
                continue
            with mlflow.start_run(run_name=f"agent-{spec.name}"):
                mlflow.set_tags(
                    {
                        "stage": "agent",
                        "agent": spec.name,
                        "origin": spec.origin,
                        "mlflow.note.content": spec.kind,
                    }
                )
                mlflow.log_params(
                    {
                        "source": spec.path,
                        "notebook": spec.notebook or "-",
                        "source_sha256": sha[:16],
                        "code_commit": git_commit(),
                    }
                )
                info = mlflow.pyfunc.log_model(
                    name="agent",
                    python_model=KaggleAgent(),
                    artifacts={"agent": str(directory)},
                    code_paths=[str(agent_code(Path(tmp) / "code"))],
                    pip_requirements=[
                        f"mlflow=={mlflow.__version__}",
                        "kaggle-environments==1.32.7",
                    ],
                )
                version = mlflow.register_model(info.model_uri, name)
        tags = {"agent": spec.name, "origin": spec.origin, "source_sha256": sha}
        if spec.notebook:
            tags["notebook"] = f"https://www.kaggle.com/code/{spec.notebook}"
        for key, value in tags.items():
            client.set_model_version_tag(name, version.version, key, value)
        client.update_model_version(name, version.version, description=spec.kind)
        log.info("%s -> %s v%s", spec.name, name, version.version)


def latest_agent_versions(settings: PipelineSettings) -> dict[str, Any]:
    client = MlflowClient()
    latest: dict[str, Any] = {}
    for v in client.search_model_versions(f"name = '{settings.agent_model}'"):
        agent = v.tags.get("agent")
        if agent in AGENTS_BY_NAME and (
            agent not in latest or int(v.version) > int(latest[agent].version)
        ):
            latest[agent] = v
    return latest


def play(job: tuple[str, str, str, str, int, str]) -> Game:
    """Одна партия; каждый игрок это свежезагруженная pyfunc-модель."""
    from kaggle_environments import make

    a0, a1, path0, path1, seed, replays = job
    players = [mlflow.pyfunc.load_model(p).unwrap_python_model() for p in (path0, path1)]
    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed})
    env.run([players[0].act, players[1].act])
    final = env.steps[-1]
    rewards = [float(s.reward or 0.0) for s in final]
    ok = all(s.status == "DONE" for s in final)
    episode = env.toJSON()
    episode_id = f"league-{seed}-{a0}-{a1}"
    episode["info"] = {"EpisodeId": episode_id, "TeamNames": [a0, a1], "seed": seed}
    with gzip.open(Path(replays) / f"{episode_id}.json.gz", "wt") as fh:
        json.dump(episode, fh)
    banks = [
        [float(env.steps[d * 24][0].observation.farms[seat].money) for d in range(30)]
        for seat in (0, 1)
    ]
    return Game(a0, a1, seed, rewards[0], rewards[1], ok, banks[0], banks[1])


def plot_pairwise(games: list[Game], order: list[str], out: Path) -> None:
    score: dict[tuple[str, str], list[float]] = defaultdict(list)
    for g in games:
        score[g.a0, g.a1].append(g.score0)
        score[g.a1, g.a0].append(1 - g.score0)
    matrix = np.array([[np.mean(score[a, b]) if a != b else np.nan for b in order] for a in order])
    fig, ax = plots.figure(7, 6)
    image = ax.imshow(matrix, cmap="RdBu", vmin=0, vmax=1)
    ax.grid(False)
    for (i, j), value in np.ndenumerate(matrix):
        if not np.isnan(value):
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8, color=plots.INK)
    ax.set_xticks(range(len(order)), order, rotation=45, ha="right")
    ax.set_yticks(range(len(order)), order)
    ax.set_title("Доля очков агента строки против агента столбца")
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.savefig(out / "pairwise_score.png")
    plots.close(fig)


def plot_banks(games: list[Game], order: list[str], out: Path) -> None:
    curves: dict[str, list[list[float]]] = defaultdict(list)
    for g in games:
        curves[g.a0].append(g.bank0)
        curves[g.a1].append(g.bank1)
    fig, ax = plots.figure(7, 4.5)
    for i, agent in enumerate(order[:3]):
        ax.plot(range(30), np.median(curves[agent], axis=0), color=plots.SERIES[i], label=agent)
    for agent in order[3:]:
        ax.plot(range(30), np.median(curves[agent], axis=0), color=plots.NEUTRAL, linewidth=1)
    ax.set(xlabel="день", ylabel="медианный банк")
    ax.set_title("Банк по дням: топ-3 цветом, остальные серым")
    ax.legend(loc="upper left")
    fig.savefig(out / "bank_by_day.png")
    plots.close(fig)


def run_league(settings: PipelineSettings, seeds: int) -> None:
    use_experiment(settings, settings.league_experiment)
    client = MlflowClient()
    versions = latest_agent_versions(settings)
    if len(versions) < 2:
        raise SystemExit("register agents first: `python -m farmvalue.league register`")

    models_dir = settings.data_dir / "league" / "models"
    replays = settings.league_replays
    replays.mkdir(parents=True, exist_ok=True)
    local = {
        agent: mlflow.artifacts.download_artifacts(
            f"models:/{settings.agent_model}/{v.version}", dst_path=str(models_dir / agent)
        )
        for agent, v in versions.items()
    }
    jobs = [
        (a, b, local[a], local[b], SEED0 + s, str(replays))
        for a, b in itertools.permutations(sorted(local), 2)
        for s in range(seeds)
    ]
    log.info("playing %d games on %d workers", len(jobs), settings.workers)
    with Pool(settings.workers, maxtasksperchild=1) as pool:
        games = pool.map(play, jobs, chunksize=1)
    failed = [g for g in games if not g.ok]
    games = [g for g in games if g.ok]

    table = pd.DataFrame(
        [
            {k: v for k, v in asdict(g).items() if not k.startswith("bank")} | {"score0": g.score0}
            for g in games
        ]
    )
    bank = defaultdict(list)
    for g in games:
        bank[g.a0].append(g.reward0)
        bank[g.a1].append(g.reward1)
    extra = pd.DataFrame(
        [
            {"agent": a, "mean_bank": float(np.mean(v)), "version": versions[a].version}
            for a, v in bank.items()
        ]
    )
    board = leaderboard([(g.a0, g.a1, g.score0) for g in games], extra)

    with mlflow.start_run(run_name=f"round-robin-{seeds}seeds") as run:
        mlflow.set_tags({"stage": "league", "champion": board.loc[0, "agent"]})
        mlflow.log_params(
            {
                "agents": len(local),
                "seeds": seeds,
                "seed0": SEED0,
                "games": len(jobs),
                "versions": ",".join(f"{a}:v{versions[a].version}" for a in sorted(local)),
                "code_commit": git_commit(),
            }
        )
        mlflow.log_metrics({"games_ok": len(games), "games_failed": len(failed)})
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            table.to_parquet(out / "games.parquet", index=False)
            mlflow.log_artifact(str(out / "games.parquet"), artifact_path="dataset")
            order = list(board["agent"])
            plot_pairwise(games, order, out)
            plot_banks(games, order, out)
            board.round(4).to_csv(out / "leaderboard.csv", index=False)
            mlflow.log_artifacts(str(out), artifact_path="report")
        source = physical_uri(f"{run.info.artifact_uri}/dataset/games.parquet", settings)
        mlflow.log_input(
            from_pandas(table, source=source, targets="score0", name="kaggriculture-league-games"),
            context="evaluation",
        )
        mlflow.log_table(board.round(4), artifact_file="report/leaderboard.json")
        for agent in order[:4]:
            sample = next(g for g in games if g.a0 == agent)
            mlflow.log_artifact(
                str(replays / f"league-{sample.seed}-{sample.a0}-{sample.a1}.json.gz"),
                artifact_path="replays",
            )
        league_run = run.info.run_id

    for row in board.to_dict("records"):
        version = versions[str(row["agent"])]
        metrics = {
            key: float(row[key])
            for key in ("games", "score", "ci_low", "ci_high", "bt_rating", "mean_bank")
        }
        with mlflow.start_run(run_id=version.run_id):
            mlflow.log_metrics(metrics, model_id=version_model_id(version))
            mlflow.set_tag("league_run_id", league_run)
        client.set_model_version_tag(
            settings.agent_model, version.version, "bt_rating", f"{metrics['bt_rating']:.0f}"
        )
    client.set_registered_model_alias(settings.agent_model, CHAMPION, versions[order[0]].version)
    print(board.round(3).to_string(index=False))


def version_model_id(version: Any) -> str | None:
    source = str(version.source or "")
    return source.removeprefix("models:/") if source.startswith("models:/m-") else None


def agent_name(path: str) -> str:
    """sub/<name>/main.py -> name, agents/<name>.py -> name, starter -> starter."""
    p = Path(path)
    if "/" not in path:
        return path
    return p.parent.name if p.name == "main.py" else p.stem


def run_history(settings: PipelineSettings) -> None:
    """Старая лига проекта (league/games*.jsonl): отдельный dataset и таблица рейтинга."""
    files = sorted((settings.kaggriculture_dir / "league").glob("games*.jsonl"))
    rows = []
    for path in files:
        for line in path.read_text(encoding="utf-8").splitlines():
            g = json.loads(line)
            if g.get("status") != ["DONE", "DONE"]:
                continue
            rows.append(
                {
                    "a0": agent_name(g["a0"]),
                    "a1": agent_name(g["a1"]),
                    "seed": g["seed"],
                    "reward0": g["rewards"][0],
                    "reward1": g["rewards"][1],
                    "score0": float(g["res"]),
                    "file": path.name,
                }
            )
    table = pd.DataFrame(rows)
    board = leaderboard(list(zip(table["a0"], table["a1"], table["score0"], strict=True)))
    use_experiment(settings, settings.league_experiment)
    with mlflow.start_run(run_name="history-league"):
        mlflow.set_tags({"stage": "league-history"})
        mlflow.log_params({"files": ",".join(p.name for p in files), "games": len(table)})
        mlflow.log_input(
            from_pandas(
                table,
                source=str(settings.kaggriculture_dir / "league"),
                targets="score0",
                name="kaggriculture-league-history",
            ),
            context="evaluation",
        )
        mlflow.log_table(board.round(4), artifact_file="report/leaderboard.json")
    print(board.round(3).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("register")
    play_cmd = sub.add_parser("play")
    play_cmd.add_argument("--seeds", type=int, default=6)
    sub.add_parser("history")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = get_settings()
    match args.command:
        case "register":
            register_agents(settings)
        case "play":
            run_league(settings, args.seeds)
        case "history":
            run_history(settings)


if __name__ == "__main__":
    main()
