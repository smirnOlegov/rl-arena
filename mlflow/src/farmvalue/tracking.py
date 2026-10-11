"""Общие хелперы для работы с MLflow из пайплайна."""

import subprocess

import mlflow

from farmvalue.config import PipelineSettings


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip()


def use_experiment(settings: PipelineSettings, name: str) -> str:
    """Подключиться к tracking server и выбрать эксперимент, вернуть его id."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    return str(mlflow.set_experiment(name).experiment_id)
