# rl-arena

Платформа для обучения и эвала игровых RL-агентов (в духе Kaggle-соревнований по RL).

Веб-сервис — **control plane** лиги в стиле AlphaStar League:

1. **SFT по свежим реплеям**: задача `sft` уходит GPU-воркерам → получаем SFT-агента.
2. **Лига**: автоматически собранные агенты регистрируются в тренировочном пуле.
3. **Отбор подходов**: лидерборд ранжирует агентов по *нижней границе* 95% CI винрейта
   (агент с 3/3 победами не обгонит агента с 80% на 50 играх).
4. **Self-play + эксплоитеры**: оппонент для обучения выбирается через PFSP
   (Prioritized Fictitious Self-Play); эксплоитеры пересоздаются от SFT-чекпоинта
   каждые `N` эпох, чтобы не скатиться в self-play-равновесие и продолжать находить дыры.
5. **Held-out эвал**: агенты из пула `evaluation` никогда не участвуют в обучении —
   это инвариант на уровне сервиса (матч, нарушающий его, отклоняется с `409 eval_leakage`).
   Поэтому винрейт против эвал-пула — честная оценка, а не переобучение под эвал.
6. **«Залил агента — посмотрел винрейт»**: регистрируем агента `kind=uploaded`,
   эвал-матчи против held-out пула → `GET /api/v1/agents/{id}/evaluation`.

Тяжёлые вычисления (обучение, прогон матчей) — во внешних GPU-воркерах, которые читают
задачи из Redis Stream `rl-arena:jobs` и пишут результаты матчей обратно через API.

## Стек

| Что | Чем |
|---|---|
| Пакеты и окружение | [uv](https://docs.astral.sh/uv/), группы зависимостей `web` и `dev` |
| Веб | FastAPI + uvicorn, полностью асинхронный |
| БД | Postgres + SQLAlchemy 2 (async, asyncpg) + Alembic |
| Очередь задач | Redis Streams |
| Качество | ruff (lint + format), mypy `--strict`, hadolint, editorconfig-checker, pre-commit |
| Тесты | pytest + pytest-asyncio + httpx, coverage ≥ 90% (сейчас ~99%) |
| CI/CD | GitHub Actions → Docker Hub |

## Структура

```
src/rl_arena/
├── main.py            # сборка приложения, lifespan, middleware логирования
├── core/              # config (pydantic-settings), logging, version
├── domain.py          # доменные enum'ы: роли агентов, пулы, исходы, типы задач
├── exceptions.py      # доменные ошибки (→ HTTP в api/errors.py)
├── league/            # чистые алгоритмы без I/O: PFSP, Wilson CI, расписание эксплоитеров
├── db/                # ORM-модели и фабрики сессий
├── schemas/           # Pydantic-контракты API
├── services/          # бизнес-логика: агенты, матчи (защита от утечки), лига, задачи, health
└── api/               # тонкие роуты FastAPI
migrations/            # Alembic
tests/unit, tests/api  # юнит-тесты алгоритмов и сервисов + тесты HTTP-API
```

## Эндпоинты

| Метод | Путь | Назначение |
|---|---|---|
| GET | `/healthz` | liveness: процесс жив |
| GET | `/api/v1/version` | версия приложения, коммит, дата сборки |
| GET | `/api/v1/health` | end-to-end health: версия и время ответа каждого компонента |
| POST/GET | `/api/v1/agents` | регистрация / список агентов |
| GET | `/api/v1/agents/{id}/evaluation` | винрейт против held-out пула с CI |
| POST | `/api/v1/matches` | результат матча (с проверкой на утечку эвала) |
| GET | `/api/v1/league/opponent` | PFSP-выбор тренировочного оппонента |
| GET | `/api/v1/league/leaderboard` | топ агентов по нижней границе CI |
| POST | `/api/v1/league/exploiters/resample` | тик эпохи: ресемплинг эксплоитеров |
| POST/GET | `/api/v1/jobs` | задачи GPU-воркерам: `sft`, `self_play`, `exploiter_training`, `evaluation` |

Swagger: `http://localhost:8000/docs`.

### Почему `/healthz` без `/api/v1`?

`/healthz` — контракт с **инфраструктурой** (Docker `HEALTHCHECK`, k8s `livenessProbe`,
балансировщик), а не часть версионируемого бизнес-API. Версия API меняется, когда меняется
контракт для клиентов; проба живости от этого меняться не должна — при выходе `/api/v2`
манифесты деплоя трогать не придётся. Суффикс `z` — конвенция Google для служебных
z-pages, которая снижает шанс конфликта с бизнес-роутами. И `/healthz` намеренно **не**
ходит во внешние зависимости: упавшая БД — не повод перезапускать живой процесс
(иначе оркестратор устроит каскад рестартов). Глубокая проверка — в `/api/v1/health`.

### `/api/v1/version` и ловушка с версией

Типичная ошибка — захардкодить версию в коде или в `Settings` (её ещё и можно случайно
переопределить env-переменной `VERSION`). Тогда версия в `pyproject.toml`, в теге образа
в реестре и в ответе `/version` неизбежно разъезжаются. Ещё вариант той же ловушки —
читать `pyproject.toml` с диска в рантайме: в образе его нет, а `uv sync --no-install-project`
не ставит метаданные пакета, и `importlib.metadata` падает.

Здесь:

- **единственный источник версии — `pyproject.toml`**; в рантайме она читается из
  метаданных установленного пакета (`importlib.metadata.version`), в образ пакет ставится
  `--no-editable` вместе с метаданными;
- коммит и дата сборки приходят из build-args (`GIT_SHA`, `BUILD_DATE`) — по ответу
  `/version` видно, какой именно образ запущен;
- CD **отказывается публиковать** тег `vX.Y.Z`, если он не совпадает с версией в
  `pyproject.toml`, а после публикации скачивает образ из реестра и сверяет
  `/api/v1/version` с ожидаемыми версией и коммитом;
- тест `test_version_matches_pyproject` и e2e-шаг CI проверяют то же самое.

### `/api/v1/health`

Каждый компонент проверяется реальным round-trip запросом, который заодно возвращает версию:
Postgres — `SHOW server_version`, Redis — `INFO server`. Проверки идут параллельно, каждая
под таймаутом (`HEALTH_TIMEOUT_S`), один зависший компонент не вешает весь отчёт.
Если что-то недоступно — `503` и `status: degraded`; наружу уходит только класс ошибки,
детали (хосты, пароли из DSN) — только в лог.

```json
{
  "status": "ok",
  "version": "0.1.0",
  "checked_at": "2026-10-02T15:05:13.052164Z",
  "latency_ms": 5.42,
  "components": {
    "database": {"status": "up", "version": "postgresql 16.2", "latency_ms": 5.35, "error": null},
    "redis": {"status": "up", "version": "7.4.2", "latency_ms": 1.83, "error": null}
  }
}
```

## Разработка

```bash
uv sync                       # группы web + dev из uv.lock
uv run pre-commit install     # git-хуки
cp .env.example .env
docker compose up -d postgres redis
uv run alembic upgrade head
uv run uvicorn rl_arena.main:app --reload
```

Или всё одной командой: `make up` (postgres + redis + migrate + app в docker).

### Проверки

```bash
make lint   # все хуки pre-commit: ruff, ruff-format, mypy, hadolint, editorconfig, пробелы, размер файлов, uv.lock
make test   # pytest
make cov    # pytest + coverage (порог 90%) + HTML-отчёт в htmlcov/
```

Тесты по умолчанию герметичны (SQLite + fakeredis) и идут ~2 секунды. Тот же набор
гоняется против настоящих Postgres и Redis, если задать `TEST_DATABASE_URL` и
`TEST_REDIS_URL` (так делает CI).

## Логирование

- Единый формат для приложения и uvicorn; `LOG_FORMAT=text` локально, `json` в образе
  (одна строка — один JSON для Loki/ELK).
- Каждый запрос получает `request_id` (или переиспользует входящий `X-Request-ID`),
  он есть во всех строках лога этого запроса и в заголовке ответа.
- Middleware логирует метод, путь, статус и длительность; необработанные исключения — с трейсбеком.
- В docker compose у контейнеров ротация логов (`max-size: 10m`, `max-file: 3`).

## CI/CD

### CI — `.github/workflows/ci.yml`

Запускается на pull request, push в рабочие ветки и вызывается из CD:

| Job | Что делает |
|---|---|
| `lint` | `pre-commit run --all-files` — ровно те же хуки, что локально |
| `test (sqlite)` | pytest + coverage, порог 90%, отчёт в summary и артефакты |
| `test (postgres)` | тот же набор против Postgres 17 и Redis 8 + миграции: `upgrade` → `alembic check` (нет дрейфа моделей) → `downgrade` → `upgrade` |
| `docker` | сборка образа, `docker compose up --wait` всего стека, e2e-проверки `/healthz`, `/api/v1/version` (версия = pyproject, коммит = SHA), `/api/v1/health` = ok, сценарий «залил агента → эвал» |

### CD — `.github/workflows/cd.yml`

**Когда запускается и почему:**

- **push тега `vX.Y.Z`** — релиз. Тег ставится осознанно после мержа в main;
- **push в `main`** — каждый смерженный коммит → образ для стейджинга;
- **pull request — не публикует**: непроверенный код (в том числе из форков) не должен
  попадать в реестр, а секреты реестра форкам недоступны. В PR образ только собирается
  и проходит e2e в CI;
- `workflow_dispatch` — ручной перезапуск (например, после сбоя реестра).

Перед публикацией CD вызывает весь CI как reusable workflow — образ из непротестированного
кода в реестр не попадает (и CI на main не гоняется дважды).

**Версионирование образов** `docker.io/<user>/rl-arena`:

| Событие | Теги | Назначение |
|---|---|---|
| тег `v1.4.2` | `1.4.2`, `1.4`, `1`, `latest`, `sha-abc1234` | `1.4.2` неизменяемый для прода; `1.4`/`1` — плавающие для автоподхвата патчей |
| тег `v0.3.1` | `0.3.1`, `0.3`, `latest`, `sha-…` | для `0.x` мажорный тег не ставим: по semver в 0.x минорная версия может ломать API |
| push в `main` | `main`, `sha-abc1234` | `main` — последний стейджинг, `sha-*` — точная ссылка на коммит для отката |

Плюс к образу публикуются SBOM и provenance-аттестации, а после пуша CD скачивает образ по
digest и проверяет, что он стартует и отдаёт ожидаемые версию и коммит.

**Как выпустить релиз:** поднять `version` в `pyproject.toml` → `uv lock` → PR → мерж →
`git tag v0.2.0 && git push origin v0.2.0`.

**Настройка репозитория** (Settings → Secrets and variables → Actions):

- `DOCKERHUB_USERNAME` — логин/namespace на Docker Hub (variable или secret — подходят оба);
- secret `DOCKERHUB_TOKEN` — access token Docker Hub (Account settings → Personal access tokens, права Read & Write).
