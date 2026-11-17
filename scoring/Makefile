.PHONY: install up observability down lint fmt typecheck test load scale poison \
	api worker source experiment plot

RUN = uv run --env-file .env
PORT ?= 8001
WORKERS ?= 2

install:
	uv sync
	test -f .env || cp .env.example .env

# Приложение в контейнерах: api, worker, RabbitMQ, Postgres
up:
	docker compose up -d --build --wait

# То же плюс Vector, Loki, Collector, Tempo, Prometheus и Grafana на :3000
observability:
	docker compose --profile observability up -d --build --wait

down:
	docker compose --profile observability down

lint:
	uv run ruff check .
	uv run ruff format --check .

fmt:
	uv run ruff check . --fix
	uv run ruff format .

typecheck:
	uv run mypy

test:
	uv run pytest

# make load LIMIT=6000: источник в контейнере по реплеям из REPLAYS_DIR
load:
	docker compose run --rm source $(if $(LIMIT),--limit $(LIMIT))

# make scale WORKERS=2
scale:
	docker compose up -d --no-recreate --scale worker=$(WORKERS) worker

# Задача без observation: DELIVERY_LIMIT + 1 попыток, потом DLQ
poison:
	docker compose run --rm --entrypoint python source -m scoring.poison

# Процессы на хосте, без контейнеров (эксперимент из прошлой работы)
api:
	$(RUN) uvicorn --factory scoring.api:create_app --port 8000 --no-access-log

# make worker PORT=8002 для второго воркера
worker:
	$(RUN) uvicorn --factory scoring.worker:create_app --port $(PORT) --no-access-log

# make source LIMIT=1000
source:
	$(RUN) python -m scoring.source $(if $(LIMIT),--limit $(LIMIT))

experiment:
	$(RUN) python -m scoring.experiment run bounded
	$(RUN) python -m scoring.experiment run unbounded
	$(RUN) python -m scoring.experiment plot

plot:
	$(RUN) python -m scoring.experiment plot
