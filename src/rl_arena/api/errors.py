"""Перевод доменных исключений в HTTP-ответы."""

import logging

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from rl_arena.exceptions import (
    ConflictError,
    DomainError,
    EvalLeakageError,
    NotFoundError,
    QueueUnavailableError,
)

log = logging.getLogger(__name__)

# Порядок важен: более специфичные классы раньше базовых.
_STATUS_BY_ERROR: list[tuple[type[DomainError], int, str]] = [
    (NotFoundError, status.HTTP_404_NOT_FOUND, "not_found"),
    (EvalLeakageError, status.HTTP_409_CONFLICT, "eval_leakage"),
    (ConflictError, status.HTTP_409_CONFLICT, "conflict"),
    (QueueUnavailableError, status.HTTP_503_SERVICE_UNAVAILABLE, "queue_unavailable"),
]


def classify(exc: Exception) -> tuple[int, str]:
    """Вернуть HTTP-статус и машинный код ошибки."""
    for error_cls, status_code, code in _STATUS_BY_ERROR:
        if isinstance(exc, error_cls):
            return status_code, code
    return status.HTTP_500_INTERNAL_SERVER_ERROR, "domain_error"


async def domain_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Вернуть стабильный контракт ошибки: машинный код + человеческое сообщение."""
    status_code, code = classify(exc)
    log.info("%s %s -> %s: %s", request.method, request.url.path, code, exc)
    return JSONResponse(status_code=status_code, content={"error": code, "detail": str(exc)})


def register_error_handlers(app: FastAPI) -> None:
    """Подключить обработчики доменных ошибок."""
    app.add_exception_handler(DomainError, domain_error_handler)
