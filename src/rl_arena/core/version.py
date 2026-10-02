"""Версия и сборочные метаданные приложения.

Версия берётся из метаданных установленного дистрибутива, т.е. из
`pyproject.toml` — единственного источника правды. Хардкод в коде или в
настройках (которые к тому же можно переопределить env-переменной) даёт
ситуацию «в реестре образ 1.2.0, а /version отвечает 0.1.0».

Коммит и дата сборки не известны на этапе разработки — их передаёт CD
как build-args образа (`GIT_SHA`, `BUILD_DATE`).
"""

import os
from functools import lru_cache
from importlib.metadata import version

from pydantic import BaseModel

DISTRIBUTION = "rl-arena"
API_VERSION = "v1"


class BuildInfo(BaseModel):
    """Что именно сейчас запущено."""

    name: str
    version: str
    api_version: str
    commit: str
    build_date: str


@lru_cache(maxsize=1)
def get_build_info() -> BuildInfo:
    """Вернуть версию пакета и метаданные сборки (вычисляется один раз на процесс)."""
    return BuildInfo(
        name=DISTRIBUTION,
        version=version(DISTRIBUTION),
        api_version=API_VERSION,
        commit=os.environ.get("GIT_SHA", "unknown"),
        build_date=os.environ.get("BUILD_DATE", "unknown"),
    )
