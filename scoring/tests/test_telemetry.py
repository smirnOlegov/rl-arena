"""Контекст трейса и task_id проходят через брокер и пул потоков."""

import json

import pytest
from faststream.rabbit import TestRabbitBroker
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from scoring.worker import create_app
from tests.conftest import make_task

EXPORTER = InMemorySpanExporter()


@pytest.fixture(scope="module", autouse=True)
def tracer_provider() -> None:
    # Глобальный провайдер ставится один раз на процесс, поэтому здесь, а не в create_app
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(EXPORTER))
    trace.set_tracer_provider(provider)


@pytest.fixture(autouse=True)
def spans() -> InMemorySpanExporter:
    EXPORTER.clear()
    return EXPORTER


def by_name(spans: InMemorySpanExporter) -> dict[str, ReadableSpan]:
    return {span.name: span for span in spans.get_finished_spans()}


async def test_task_continues_publisher_trace(settings, sessions, model, spans):
    app = create_app(settings, model)
    task = make_task()
    async with TestRabbitBroker(app.broker) as broker:
        with trace.get_tracer(__name__).start_as_current_span("POST /api/v1/tasks") as root:
            await broker.publish(task, queue=settings.queue_name, correlation_id=str(task.id))

    named = by_name(spans)
    process = named[f"default.{settings.queue_name} process"]
    inference, save = named["inference"], named["save result"]
    trace_id = root.get_span_context().trace_id
    assert {span.context.trace_id for span in (process, inference, save)} == {trace_id}
    assert inference.parent is not None
    assert inference.parent.span_id == process.context.span_id
    assert inference.attributes is not None
    assert inference.attributes["model.version"] == "7"
    assert process.attributes is not None
    assert process.attributes["task.id"] == str(task.id)
    assert process.attributes["messaging.delivery.attempt"] == 1


async def test_logs_carry_task_and_trace(settings, sessions, model, spans, capsys):
    settings.log_level = "DEBUG"
    app = create_app(settings, model)
    task = make_task()
    async with TestRabbitBroker(app.broker) as broker:
        await broker.publish(task, queue=settings.queue_name, correlation_id=str(task.id))

    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    process = by_name(spans)[f"default.{settings.queue_name} process"]
    task_records = [record for record in records if record.get("task_id") == str(task.id)]
    assert task_records
    assert {record["trace_id"] for record in task_records} == {
        trace.format_trace_id(process.context.trace_id)
    }


async def test_last_attempt_goes_to_dlq(settings, sessions, model, spans, capsys):
    app = create_app(settings, model)
    async with TestRabbitBroker(app.broker) as broker:
        with pytest.raises(Exception, match="validation error"):
            await broker.publish(
                {"id": "broken"},
                queue=settings.queue_name,
                correlation_id="broken",
                headers={"x-delivery-count": settings.delivery_limit},
            )

    process = by_name(spans)[f"default.{settings.queue_name} process"]
    assert process.status.status_code is StatusCode.ERROR
    assert process.attributes is not None
    assert process.attributes["messaging.delivery.attempt"] == settings.delivery_limit + 1
    assert [event.name for event in process.events] == ["moved to DLQ", "exception"]
    messages = [json.loads(line)["message"] for line in capsys.readouterr().out.splitlines()]
    assert "task moved to DLQ" in messages
