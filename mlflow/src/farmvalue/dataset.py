"""Датасет снимков партий: реплеи Kaggriculture -> parquet -> MLflow Dataset.

Строка датасета это состояние партии на начало дня d (hour 0, d = 1..29),
таргет `seat0_wins`. Сплит train/test считается по хешу episode_id, поэтому
снимки одной партии не попадают в обе части, а сплит старых партий не меняется
при добавлении новых реплеев в следующую версию датасета.

Запуск: `uv run python -m farmvalue.dataset v1`
"""

import argparse
import gzip
import hashlib
import json
import logging
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from mlflow.data.pandas_dataset import PandasDataset, from_pandas

from farmvalue.config import PipelineSettings, get_settings
from farmvalue.features import FEATURES, snapshot_features
from farmvalue.tracking import git_commit, use_experiment

log = logging.getLogger(__name__)

TARGET = "seat0_wins"
DAYS = range(1, 30)
TEST_SHARE = 5
PARQUET = "snapshots.parquet"

LADDER_DIRS = (
    "data/EF_eps/replays",
    "data/G_eps/replays",
    "data/G_eps_s51/replays",
    "data/elo/our_eps/replays",
)
VERSIONS = ("v1", "v2")


@dataclass(frozen=True)
class Source:
    name: str
    path: Path


def sources(settings: PipelineSettings, version: str) -> list[Source]:
    ladder = [Source("ladder", settings.kaggriculture_dir / d) for d in LADDER_DIRS]
    if version == "v1":
        return ladder
    if version == "v2":
        return [*ladder, Source("league", settings.league_replays)]
    raise ValueError(f"unknown dataset version {version!r}")


def split_of(episode_id: str) -> str:
    bucket = int(hashlib.sha1(episode_id.encode()).hexdigest(), 16) % TEST_SHARE  # noqa: S324
    return "test" if bucket == 0 else "train"


def load_episode(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt") as fh:
        episode: dict[str, Any] = json.load(fh)
    return episode


def episode_rows(path: Path, source: str) -> list[dict[str, Any]]:
    """Снимки одной партии; пустой список для незавершённых партий и ничьих."""
    episode = load_episode(path)
    statuses = episode.get("statuses") or []
    rewards = episode.get("rewards") or []
    if statuses != ["DONE", "DONE"] or len(rewards) != 2 or rewards[0] == rewards[1]:
        return []

    info = episode.get("info") or {}
    teams = info.get("TeamNames") or ["", ""]
    episode_id = str(info.get("EpisodeId") or path.name.removesuffix(".json.gz"))
    meta = {
        "episode_id": episode_id,
        "source": source,
        "team0": teams[0],
        "team1": teams[1],
        "reward0": float(rewards[0]),
        "reward1": float(rewards[1]),
        TARGET: int(rewards[0] > rewards[1]),
        "split": split_of(episode_id),
    }
    steps = episode["steps"]
    return [
        {**meta, **snapshot_features(steps[day * 24][0]["observation"])}
        for day in DAYS
        if day * 24 < len(steps)
    ]


def replay_files(srcs: list[Source]) -> list[tuple[Path, str]]:
    files: dict[str, tuple[Path, str]] = {}
    for src in srcs:
        for path in sorted(src.path.rglob("*.json.gz")):
            files.setdefault(path.name, (path, src.name))
    return sorted(files.values())


def files_digest(files: list[tuple[Path, str]]) -> str:
    h = hashlib.sha256()
    for path, _ in files:
        h.update(path.name.encode())
        h.update(str(path.stat().st_size).encode())
    return h.hexdigest()[:16]


def build_frame(files: list[tuple[Path, str]], workers: int) -> pd.DataFrame:
    with ProcessPoolExecutor(workers) as pool:
        chunks = pool.map(episode_rows, *zip(*files, strict=True), chunksize=4)
        rows = [row for chunk in chunks for row in chunk]
    df = pd.DataFrame(rows).sort_values(["episode_id", "day"], ignore_index=True)
    return df.astype(dict.fromkeys(FEATURES, "float64"))


def physical_uri(artifact_uri: str, settings: PipelineSettings) -> str:
    """mlflow-artifacts:/... -> адрес объекта в artifact store, куда сервер его положил."""
    path = artifact_uri.removeprefix("mlflow-artifacts:").lstrip("/")
    return f"{settings.artifacts_destination.rstrip('/')}/{path}"


def as_mlflow_dataset(df: pd.DataFrame, source: str, name: str) -> PandasDataset:
    return from_pandas(df, source=source, targets=TARGET, name=name)


def build(version: str, settings: PipelineSettings) -> str:
    """Собрать датасет, залогировать run в эксперимент данных, вернуть run_id."""
    files = replay_files(sources(settings, version))
    if not files:
        raise SystemExit("no replays found, check KAGGRICULTURE_DIR")
    df = build_frame(files, settings.workers)
    log.info("dataset %s: %d files, %d rows", version, len(files), len(df))

    use_experiment(settings, settings.data_experiment)
    with mlflow.start_run(run_name=f"dataset-{version}") as run:
        mlflow.set_tags({"stage": "dataset", "dataset_version": version})
        mlflow.log_params(
            {
                "version": version,
                "sources": ",".join(sorted({s for _, s in files})),
                "replay_dirs": ",".join(LADDER_DIRS),
                "replay_files": len(files),
                "replay_files_sha256": files_digest(files),
                "days": f"{DAYS.start}-{DAYS.stop - 1}",
                "test_share": f"1/{TEST_SHARE}",
                "code_commit": git_commit(),
            }
        )
        mlflow.log_metrics(
            {
                "episodes": df["episode_id"].nunique(),
                "rows": len(df),
                "positive_rate": float(df[TARGET].mean()),
                "test_episodes": df.loc[df["split"] == "test", "episode_id"].nunique(),
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / PARQUET
            df.to_parquet(path, index=False)
            mlflow.log_artifact(str(path), artifact_path="dataset")
        source = physical_uri(f"{run.info.artifact_uri}/dataset/{PARQUET}", settings)
        dataset = as_mlflow_dataset(df, source, f"kaggriculture-snapshots-{version}")
        mlflow.log_input(dataset, context="build")
        mlflow.set_tag("dataset_digest", dataset.digest)
        return str(run.info.run_id)


def find_run(version: str, settings: PipelineSettings) -> Any:
    use_experiment(settings, settings.data_experiment)
    runs = mlflow.search_runs(
        experiment_names=[settings.data_experiment],
        filter_string=(
            f"tags.stage = 'dataset' and tags.dataset_version = '{version}' "
            "and attributes.status = 'FINISHED'"
        ),
        order_by=["attributes.start_time DESC"],
        max_results=1,
        output_format="list",
    )
    if not runs:
        raise SystemExit(f"dataset {version} not found, run `make dataset` first")
    return runs[0]


def load(version: str, settings: PipelineSettings) -> tuple[pd.DataFrame, str, str]:
    """Скачать датасет через прокси сервера; вернуть (df, source uri, data run_id)."""
    run = find_run(version, settings)
    artifact = f"{run.info.artifact_uri}/dataset/{PARQUET}"
    local = mlflow.artifacts.download_artifacts(artifact_uri=artifact)
    return pd.read_parquet(local), physical_uri(artifact, settings), str(run.info.run_id)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", choices=VERSIONS)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run_id = build(args.version, get_settings())
    print(run_id)


if __name__ == "__main__":
    main()
