# rl-arena

Платформа для обучения и эвала игровых RL-агентов (Kaggle-style), лига в духе AlphaStar.
Сервис — control plane: решает, что учить и с кем играть; обучение и матчи выполняют
внешние GPU-воркеры, которые берут задачи из Redis и присылают результаты матчей в API.

- **SFT по реплеям, self-play, эксплоитеры** — задачи воркерам через `POST /api/v1/jobs`.
- **PFSP-матчмейкинг** — оппонент выбирается тем чаще, чем хуже агент против него играет.
- **Эксплоитеры** пересоздаются от SFT-чекпоинта каждые `N` эпох, чтобы не выродиться в self-play.
- **Held-out эвал** — агенты пула `evaluation` никогда не участвуют в обучении
  (такой матч отклоняется с `409 eval_leakage`), поэтому винрейт против них честный.
- **Лидерборд** — по нижней границе 95% доверительного интервала (Уилсон), а не по среднему.
- **«Залить агента и посмотреть винрейт»** — `kind=uploaded` → `GET /api/v1/agents/{id}/evaluation`.

## Быстрый старт

```bash
make up        # postgres + redis + миграции + приложение в docker
make down      # остановить
```

Swagger: http://localhost:8000/docs

## Эндпоинты

| Метод | Путь | Что делает |
|---|---|---|
| GET | `/healthz` | процесс жив (без обращения к БД) |
| GET | `/api/v1/version` | версия, коммит, дата сборки |
| GET | `/api/v1/health` | версия и время ответа Postgres и Redis; `503`, если что-то недоступно |
| POST/GET | `/api/v1/agents` | регистрация / список агентов |
| GET | `/api/v1/agents/{id}/evaluation` | винрейт против held-out пула с доверительным интервалом |
| POST | `/api/v1/matches` | результат матча |
| GET | `/api/v1/league/opponent` | выбор тренировочного оппонента (PFSP) |
| GET | `/api/v1/league/leaderboard` | топ агентов |
| POST | `/api/v1/league/exploiters/resample` | тик эпохи: пересоздать эксплоитеров |
| POST/GET | `/api/v1/jobs` | задачи воркерам |

**Почему `/healthz` без `/api/v1`.** Это проба для инфраструктуры (Docker healthcheck,
k8s liveness), а не бизнес-API: она не должна меняться при выходе `/api/v2`. В БД не ходит —
упавшая база не повод перезапускать живой процесс.

**Версия (ловушка).** Версия задаётся только в `pyproject.toml` и читается из метаданных
установленного пакета, а не хардкодится. Коммит и дата сборки приходят из build-args образа.
CD не опубликует тег `vX.Y.Z`, если он не совпадает с версией в `pyproject.toml`.

## Стек

- **uv** — менеджер пакетов; группы зависимостей `web` (в образ) и `dev` (линтеры, тесты).
- **FastAPI + uvicorn** — асинхронный веб.
- **Postgres + SQLAlchemy 2 (async) + Alembic** — данные и миграции.
- **Redis Streams** — очередь задач.
- **ruff, mypy, hadolint, pre-commit** — линт, формат, типы.
- **pytest + coverage** — тесты, порог покрытия 90%.
- **GitHub Actions → Docker Hub** — CI/CD.

## Структура

```
src/rl_arena/
├── main.py        # сборка приложения, middleware логов
├── api/           # роуты
├── services/      # бизнес-логика
├── league/        # алгоритмы без I/O: PFSP, интервал Уилсона, расписание эксплоитеров
├── db/models.py   # таблицы
├── schemas/       # Pydantic-контракты ответов
└── core/          # настройки, логирование, версия
migrations/        # Alembic
tests/             # unit/ — алгоритмы и сервисы, api/ — HTTP
```

## Данные

- **Postgres** — источник правды: `agents` (агент, роль, пул, ссылка на чекпоинт),
  `matches` (исход с точки зрения `agent_id`, `purpose`: training/evaluation), `jobs`.
- **Redis** — только доставка задач воркерам (Stream `rl-arena:jobs`). Задача сначала
  сохраняется в Postgres, потом публикуется; если Redis недоступен — `503`, задача остаётся в БД.

## Разработка

```bash
make install    # зависимости + git-хуки pre-commit
make lint       # все проверки pre-commit (то же, что в CI)
make test       # быстрые тесты: SQLite + fakeredis
make test-real  # те же тесты на Postgres и Redis из docker compose
make cov        # покрытие + HTML-отчёт в htmlcov/
make run        # приложение локально с автоперезагрузкой
```

Новая миграция после изменения `db/models.py` (нужен запущенный Postgres):

```bash
uv run alembic revision --autogenerate -m "описание"
uv run alembic upgrade head
```

## Логи

JSON в образе (`LOG_FORMAT=json`), текст локально. В каждой строке `request_id`;
входящий `X-Request-ID` переиспользуется и возвращается в ответе.

## CI/CD

**CI** (`ci.yml`) — на pull request и push в рабочие ветки:
pre-commit → тесты на SQLite и на Postgres+Redis (с проверкой миграций) →
сборка образа и e2e через `docker compose`.

**CD** (`cd.yml`) — сначала весь CI, затем публикация образа `<логин>/rl-arena`:

| Событие | Теги образа |
|---|---|
| push в `main` | `main`, `sha-<коммит>` — стейджинг и точный откат |
| тег `v1.2.3` | `1.2.3`, `1.2`, `1`, `latest` (для `0.x` без мажорного тега) |
| pull request | не публикуется — непроверенный код не попадает в реестр |

После публикации CD скачивает образ из реестра и проверяет `/api/v1/version`.

**Новый релиз:**

```bash
# 1. поднять version в pyproject.toml, затем:
uv lock
git commit -am "release: 0.2.0" && git push
git tag v0.2.0 && git push origin v0.2.0
```

**Секреты** (Settings → Secrets and variables → Actions): `DOCKERHUB_USERNAME`
(variable или secret) и secret `DOCKERHUB_TOKEN`.
