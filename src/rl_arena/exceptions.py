"""Доменные исключения; в HTTP-коды их переводит `rl_arena.api.errors`."""


class DomainError(Exception):
    """Базовое доменное исключение с человекочитаемым сообщением."""


class NotFoundError(DomainError):
    """Сущность не найдена."""


class ConflictError(DomainError):
    """Операция противоречит текущему состоянию (дубликат имени и т.п.)."""


class EvalLeakageError(ConflictError):
    """Попытка использовать held-out эвал-агента в обучении (или наоборот)."""


class NoOpponentsError(ConflictError):
    """В тренировочном пуле нет ни одного доступного оппонента."""


class QueueUnavailableError(DomainError):
    """Очередь задач (Redis) недоступна."""
