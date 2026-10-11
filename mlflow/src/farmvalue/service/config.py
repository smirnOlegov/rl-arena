"""Настройки inference-сервиса из окружения."""

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Каждое поле переопределяется env-переменной того же имени."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    mlflow_tracking_uri: str = "http://localhost:5000"
    model_name: str = "kaggriculture-winprob"
    model_alias: str = "champion"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


def get_settings() -> Settings:
    return Settings()
