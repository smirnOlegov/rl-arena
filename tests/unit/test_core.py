"""Ядро: логирование, конфигурация, маппинг ошибок."""

import json
import logging
import sys

import pytest
from pydantic import ValidationError

from rl_arena.api.errors import classify
from rl_arena.core.config import Settings
from rl_arena.core.logging import REQUEST_ID, JsonFormatter, RequestIdFilter, setup_logging
from rl_arena.exceptions import DomainError, EvalLeakageError, QueueUnavailableError


def _record(message: str = "hello") -> logging.LogRecord:
    return logging.LogRecord("rl_arena.test", logging.INFO, __file__, 1, message, None, None)


def test_request_id_filter_uses_context_value() -> None:
    record = _record()
    token = REQUEST_ID.set("req-1")
    try:
        RequestIdFilter().filter(record)
    finally:
        REQUEST_ID.reset(token)
    assert record.request_id == "req-1"


def test_request_id_defaults_to_dash_outside_request() -> None:
    record = _record()
    RequestIdFilter().filter(record)
    assert record.request_id == "-"


def test_json_formatter_emits_one_parsable_line() -> None:
    record = _record("матч записан")
    record.request_id = "abc"
    payload = json.loads(JsonFormatter().format(record))
    assert payload["message"] == "матч записан"
    assert payload["level"] == "INFO"
    assert payload["request_id"] == "abc"
    assert payload["logger"] == "rl_arena.test"


def test_json_formatter_includes_traceback() -> None:
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        record = logging.LogRecord(
            "x", logging.ERROR, __file__, 1, "failed", None, exc_info=sys.exc_info()
        )
    assert "RuntimeError: boom" in json.loads(JsonFormatter().format(record))["exc_info"]


@pytest.mark.parametrize(("fmt", "formatter_cls"), [("json", JsonFormatter), ("text", None)])
def test_setup_logging_installs_single_handler(fmt: str, formatter_cls: type | None) -> None:
    setup_logging("DEBUG", fmt)
    root = logging.getLogger()
    assert len(root.handlers) == 1
    assert root.level == logging.DEBUG
    if formatter_cls is not None:
        assert isinstance(root.handlers[0].formatter, formatter_cls)
    assert logging.getLogger("uvicorn.access").disabled
    setup_logging("INFO", "text")


def test_settings_validate_values() -> None:
    with pytest.raises(ValidationError):
        Settings(exploiter_resample_every=0)
    with pytest.raises(ValidationError):
        Settings(log_level="LOUD")


def test_settings_read_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXPLOITER_RESAMPLE_EVERY", "7")
    monkeypatch.setenv("LOG_FORMAT", "json")
    settings = Settings()
    assert settings.exploiter_resample_every == 7
    assert settings.log_format == "json"


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (EvalLeakageError("x"), (409, "eval_leakage")),
        (QueueUnavailableError("x"), (503, "queue_unavailable")),
        (DomainError("x"), (500, "domain_error")),
    ],
)
def test_classify_domain_errors(error: Exception, expected: tuple[int, str]) -> None:
    assert classify(error) == expected
