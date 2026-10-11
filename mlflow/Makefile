.PHONY: install lint fmt typecheck up down logs dataset dataset-v2 eda train register promote show \
	agents league history app app-restart app-logs all reset

RUN = uv run --env-file .env
SEEDS ?= 6

install:
	uv sync
	test -f .env || cp .env.example .env

lint:
	uv run ruff check .
	uv run ruff format --check .

fmt:
	uv run ruff check . --fix
	uv run ruff format .

typecheck:
	uv run mypy

# MLflow: postgres (backend store) + s3 (artifact store) + tracking server
up:
	docker compose up -d --build --wait

down:
	docker compose down

logs:
	docker compose logs -f mlflow

dataset:
	$(RUN) python -m farmvalue.dataset v1

eda:
	$(RUN) python -m farmvalue.eda v1

train:
	$(RUN) python -m farmvalue.train --dataset v1

register:
	$(RUN) python -m farmvalue.registry register --top 2

# make promote VERSION=1 [ALIAS=champion]
promote:
	$(RUN) python -m farmvalue.registry promote $(VERSION) --alias $(or $(ALIAS),champion)

show:
	$(RUN) python -m farmvalue.registry show

agents:
	$(RUN) python -m farmvalue.league register

league: agents
	$(RUN) python -m farmvalue.league play --seeds $(SEEDS)

history:
	$(RUN) python -m farmvalue.league history

# Датасет v2 = реплеи ladder + партии лиги; лучший вариант переобучается на нём
dataset-v2:
	$(RUN) python -m farmvalue.dataset v2
	$(RUN) python -m farmvalue.train --dataset v2 --variant hgb_tuned

# Inference-сервис в docker; перезапуск подхватывает текущий alias
app:
	docker compose --profile app up -d --build --wait app

app-restart:
	docker compose --profile app restart app

app-logs:
	docker compose --profile app logs -f app

# Весь путь с нуля: данные, EDA, эксперименты, лига, v2, реестр, сервис
all: up dataset eda train league history dataset-v2 register app

# Удалить все данные MLflow (тома postgres и s3) и локальные реплеи лиги
reset:
	docker compose --profile app down -v
	rm -rf data
