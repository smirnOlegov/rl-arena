# syntax=docker/dockerfile:1.7

# ---- builder: собираем venv только с рантайм-зависимостями (группа web) ----
FROM python:3.13-slim-trixie AS builder

COPY --from=ghcr.io/astral-sh/uv:0.12.19 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0 \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

# Сначала только зависимости: этот слой кешируется, пока не меняется uv.lock.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-default-groups --group web

# Затем сам пакет. --no-editable: пакет копируется в site-packages вместе с
# метаданными, из которых /api/v1/version читает версию (src в рантайме не нужен).
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-default-groups --group web --no-editable

# ---- runtime: только интерпретатор + venv, без uv, компиляторов и исходников ----
FROM python:3.13-slim-trixie AS runtime

ARG GIT_SHA=unknown
ARG BUILD_DATE=unknown

LABEL org.opencontainers.image.title="rl-arena" \
    org.opencontainers.image.description="Платформа обучения и эвала игровых RL-агентов" \
    org.opencontainers.image.revision="${GIT_SHA}" \
    org.opencontainers.image.created="${BUILD_DATE}"

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    GIT_SHA=${GIT_SHA} \
    BUILD_DATE=${BUILD_DATE} \
    LOG_FORMAT=json

# Непривилегированный пользователь с числовым UID: k8s (runAsNonRoot) может
# проверить, что это не root, не заглядывая в /etc/passwd образа.
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --no-create-home app

WORKDIR /app
COPY --from=builder --chown=10001:10001 /app/.venv /app/.venv
COPY --chown=10001:10001 alembic.ini ./
COPY --chown=10001:10001 migrations ./migrations

USER 10001:10001
EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2)"]

CMD ["uvicorn", "rl_arena.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
