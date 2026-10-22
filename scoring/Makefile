.PHONY: install up down lint fmt typecheck test api worker source poison experiment plot

RUN = uv run --env-file .env
PORT ?= 8001
QUEUE ?= tasks

install:
	uv sync
	test -f .env || cp .env.example .env

up:
	docker compose up -d --wait

down:
	docker compose down

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

api:
	$(RUN) uvicorn --factory scoring.api:create_app --port 8000

# make worker PORT=8002 для второго воркера
worker:
	$(RUN) uvicorn --factory scoring.worker:create_app --port $(PORT)

# make source LIMIT=1000
source:
	$(RUN) python -m scoring.source $(if $(LIMIT),--limit $(LIMIT))

# Заведомо некорректное сообщение: после DELIVERY_LIMIT попыток оно окажется в $(QUEUE).dlq
poison:
	curl -s -u guest:guest -X POST localhost:15672/api/exchanges/%2F/amq.default/publish \
		-d '{"properties":{},"routing_key":"$(QUEUE)","payload":"{\"id\":\"broken\"}","payload_encoding":"string"}'

experiment:
	$(RUN) python -m scoring.experiment run bounded
	$(RUN) python -m scoring.experiment run unbounded
	$(RUN) python -m scoring.experiment plot

plot:
	$(RUN) python -m scoring.experiment plot
