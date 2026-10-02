"""Доменные понятия лиги: роли агентов, пулы, исходы матчей, типы задач."""

from enum import StrEnum


class AgentKind(StrEnum):
    """Откуда агент взялся и какую роль играет в лиге (по мотивам AlphaStar League)."""

    sft = "sft"  # behaviour cloning по свежим реплеям
    main = "main"  # основной агент: self-play + игры против лиги
    exploiter = "exploiter"  # ищет слабости main-агентов, периодически ресемплится
    uploaded = "uploaded"  # залит пользователем «просто посмотреть винрейт»
    baseline = "baseline"  # фиксированный бот: эвристика, публичный сабмит и т.п.


class AgentPool(StrEnum):
    """Пул агента. Неизменяем после создания.

    `evaluation` — held-out оппоненты: против них только меряемся, но никогда не
    учимся. Иначе агент переобучится под эвал-пул и скор будет завышен.
    """

    training = "training"
    evaluation = "evaluation"


class MatchPurpose(StrEnum):
    """Зачем сыгран матч: шаг обучения или замер качества."""

    training = "training"
    evaluation = "evaluation"


class MatchResult(StrEnum):
    """Исход матча с точки зрения `agent_id` (не оппонента)."""

    win = "win"
    draw = "draw"
    loss = "loss"


class JobKind(StrEnum):
    """Тип задачи для GPU-воркера."""

    sft = "sft"
    self_play = "self_play"
    exploiter_training = "exploiter_training"
    evaluation = "evaluation"


class JobStatus(StrEnum):
    """Жизненный цикл задачи; переводы статусов делают воркеры."""

    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
