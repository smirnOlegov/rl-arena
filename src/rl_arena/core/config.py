"""Настройки приложения из переменных окружения (pydantic-settings).

Версии приложения здесь сознательно НЕТ: единственный источник версии —
`pyproject.toml`, в рантайме она читается из метаданных установленного
пакета (см. `rl_arena.core.version`). Иначе версия в коде, в образе и в теге
реестра рано или поздно разъедутся.
"""

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Конфигурация сервиса; каждое поле переопределяется env-переменной того же имени."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "rl-arena"
    environment: Literal["development", "test", "staging", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["text", "json"] = "text"

    database_url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/rl_arena",
        description="SQLAlchemy URL с асинхронным драйвером.",
    )
    redis_url: str = "redis://localhost:6379/0"
    jobs_stream: str = Field(
        default="rl-arena:jobs",
        description="Redis Stream, из которого GPU-воркеры забирают задачи обучения/эвала.",
    )

    health_timeout_s: float = Field(default=2.0, gt=0)

    exploiter_resample_every: int = Field(
        default=5,
        ge=1,
        description="Раз в сколько эпох эксплоитеры пересоздаются от SFT-чекпоинта.",
    )
    exploiters_per_league: int = Field(default=2, ge=1)


def get_settings() -> Settings:
    """Прочитать настройки из окружения."""
    return Settings()
