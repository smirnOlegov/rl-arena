.PHONY: install lint fmt typecheck test test-real cov run migrate up down logs

# Зависимости из lock-файла (группы web + dev) и git-хуки
install:
	uv sync
	uv run pre-commit install

# Все проверки pre-commit по всему репозиторию (то же, что в CI)
lint:
	uv run pre-commit run --all-files

# Автофикс линтера + форматирование
fmt:
	uv run ruff check . --fix
	uv run ruff format .

typecheck:
	uv run mypy

test:
	uv run pytest -q

# Те же тесты на настоящих Postgres и Redis из compose. Отдельная БД rl_arena_test
# и Redis db 1 — тесты пересоздают схему и чистят Redis, данные приложения не трогаем.
test-real:
	docker compose up -d --wait postgres redis
	docker compose exec postgres createdb -U postgres rl_arena_test 2>/dev/null || true
	TEST_DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5433/rl_arena_test \
		TEST_REDIS_URL=redis://localhost:6380/1 uv run pytest -q

# Тесты с покрытием и HTML-отчётом в htmlcov/
cov:
	uv run pytest --cov --cov-report=term-missing --cov-report=html

# Локальный запуск без docker (нужны Postgres и Redis, см. .env.example)
run:
	uv run uvicorn rl_arena.main:app --reload --port 8000

migrate:
	uv run alembic upgrade head

# Весь стек в docker: postgres + redis + migrate + app
up:
	GIT_SHA=$$(git rev-parse --short HEAD) BUILD_DATE=$$(date -u +%Y-%m-%dT%H:%M:%SZ) \
		docker compose up -d --build --wait

down:
	docker compose down

logs:
	docker compose logs -f app
