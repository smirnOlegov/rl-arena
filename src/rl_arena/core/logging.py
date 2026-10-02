"""Логирование: единый формат для приложения и uvicorn, request_id в каждой строке.

`request_id` живёт в ContextVar: middleware выставляет его на время запроса,
поэтому все логи одного запроса (роут, сервис, БД) связаны одним id.

Два формата:
- `text` — для локальной разработки, читается глазами;
- `json` — для прода, одна строка = один JSON-объект, парсится Loki/ELK.
"""

import contextvars
import json
import logging
import sys
from datetime import UTC, datetime
from typing import Literal

REQUEST_ID: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

_TEXT_FORMAT = "%(asctime)s | %(levelname)-8s | %(request_id)s | %(name)s | %(message)s"


class RequestIdFilter(logging.Filter):
    """Подмешать request_id из контекста в каждую запись."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Добавить атрибут `request_id` и пропустить запись дальше."""
        record.request_id = REQUEST_ID.get()
        return True


class JsonFormatter(logging.Formatter):
    """Сериализовать запись лога в одну JSON-строку."""

    def format(self, record: logging.LogRecord) -> str:
        """Вернуть JSON с временем, уровнем, логгером, request_id и сообщением."""
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "request_id": getattr(record, "request_id", "-"),
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(level: str, fmt: Literal["text", "json"]) -> None:
    """Настроить корневой логгер и переподчинить ему логгеры uvicorn."""
    handler = logging.StreamHandler(sys.stdout)
    formatter = JsonFormatter() if fmt == "json" else logging.Formatter(_TEXT_FORMAT)
    handler.setFormatter(formatter)
    handler.addFilter(RequestIdFilter())

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)

    # access-лог uvicorn дублирует наш middleware-лог, поэтому глушим его;
    # остальные логгеры uvicorn пишут в общий handler и общем формате.
    for name in ("uvicorn", "uvicorn.error"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers = [handler]
        uvicorn_logger.propagate = False
    logging.getLogger("uvicorn.access").disabled = True
