"""JSON-логи в stdout: одна строка на запись, одинаковые поля у всех процессов и библиотек."""

import logging
import sys
from contextvars import ContextVar

from opentelemetry import trace
from pythonjsonlogger.core import RESERVED_ATTRS
from pythonjsonlogger.json import JsonFormatter

# Задача, которую сейчас обрабатывает этот asyncio-таск или поток пула
current_task_id: ContextVar[str | None] = ContextVar("current_task_id", default=None)

FIELDS = ("levelname", "name", "message")
RENAMED = {"levelname": "level", "name": "logger"}
# uvicorn дублирует сообщение с ANSI-цветами в extra
SKIPPED = [*RESERVED_ATTRS, "color_message"]
# Библиотеки, которые ставят себе текстовые обработчики и не передают записи корневому логгеру
SELF_CONFIGURED = ("uvicorn", "uvicorn.error", "mlflow", "faststream")


class TaskContextFilter(logging.Filter):
    """Добавляет к записи сервис, task_id и контекст трейса, внутри которого она сделана."""

    def __init__(self, service: str) -> None:
        super().__init__()
        self.service = service

    def filter(self, record: logging.LogRecord) -> bool:
        record.service = self.service
        if (task_id := current_task_id.get()) is not None:
            record.task_id = task_id
        span_context = trace.get_current_span().get_span_context()
        if span_context.is_valid:
            record.trace_id = trace.format_trace_id(span_context.trace_id)
            record.span_id = trace.format_span_id(span_context.span_id)
        return True


def configure_logging(level: str, service: str) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        JsonFormatter(FIELDS, rename_fields=RENAMED, reserved_attrs=SKIPPED, timestamp="time")
    )
    handler.addFilter(TaskContextFilter(service))
    logging.basicConfig(level=level, handlers=[handler], force=True)
    for name in SELF_CONFIGURED:
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.propagate = True
