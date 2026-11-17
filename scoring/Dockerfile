# syntax=docker/dockerfile:1.7

# Один образ на api, worker и source: процессы различаются только командой в compose.
# Контекст сборки — корень rl-arena, потому что scoring зависит от ../mlflow (farmvalue) по пути.

FROM python:3.13-slim-trixie AS builder

COPY --from=ghcr.io/astral-sh/uv:0.12.19 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0 \
    UV_PROJECT_ENVIRONMENT=/app/.venv

# farmvalue почти не меняется, а без него uv не соберёт окружение
COPY mlflow/pyproject.toml mlflow/README.md /app/mlflow/
COPY mlflow/src /app/mlflow/src

WORKDIR /app/scoring

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=scoring/uv.lock,target=uv.lock \
    --mount=type=bind,source=scoring/pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-default-groups --no-editable

COPY scoring/pyproject.toml scoring/uv.lock scoring/README.md ./
COPY scoring/src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-default-groups --no-editable

FROM python:3.13-slim-trixie AS runtime

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/tmp

RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --no-create-home app

WORKDIR /app
COPY --from=builder --chown=10001:10001 /app/.venv /app/.venv

USER 10001:10001
