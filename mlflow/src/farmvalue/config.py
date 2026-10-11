"""Настройки пайплайна (датасет, обучение, лига) из окружения."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class PipelineSettings(BaseSettings):
    """Каждое поле переопределяется env-переменной того же имени."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    mlflow_tracking_uri: str = "http://localhost:5000"
    artifacts_destination: str = "s3://mlflow"
    kaggriculture_dir: Path = Path.home() / "kaggriculture"
    data_dir: Path = Path("data")

    data_experiment: str = "kaggriculture-data"
    winprob_experiment: str = "kaggriculture-winprob"
    league_experiment: str = "kaggriculture-league"

    winprob_model: str = "kaggriculture-winprob"
    agent_model: str = "kaggriculture-agent"

    workers: int = 8

    @property
    def league_replays(self) -> Path:
        return self.data_dir / "league" / "replays"


def get_settings() -> PipelineSettings:
    return PipelineSettings()
