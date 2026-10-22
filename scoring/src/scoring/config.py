"""Настройки всех процессов конвейера: env-переменные или .env."""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    rabbit_url: str = "amqp://guest:guest@localhost:5672/"
    database_url: str = "postgresql+asyncpg://scoring:scoring@localhost:5435/scoring"

    queue_name: str = "tasks"
    queue_max_length: int = 100
    delivery_limit: int = 3

    prefetch_count: int = 4
    worker_threads: int = 4
    inference_delay_s: float = 0.2
    graceful_timeout_s: float = 30.0

    backoff_min_s: float = 0.01
    backoff_max_s: float = 1.0
    replays: Path = Path("../mlflow/data/league/replays")

    mlflow_tracking_uri: str = "http://localhost:5000"
    model_name: str = "kaggriculture-winprob"
    model_alias: str = "champion"

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


def get_settings() -> Settings:
    return Settings()
