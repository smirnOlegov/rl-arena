"""OpenTelemetry: трейсы и метрики по OTLP в Collector.

Куда и что отправлять, задают стандартные переменные OTEL_* (endpoint, sampler,
service.name, resource-атрибуты, интервал экспорта), код их не дублирует.
"""

from collections.abc import Iterable, Sequence

from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.system_metrics import SystemMetricsInstrumentor
from opentelemetry.metrics import CallbackOptions, Observation
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.metrics.view import View
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from scoring.config import Settings

PROCESS_METRICS: dict[str, list[str] | None] = {
    "process.memory.usage": None,
    "process.cpu.time": ["user", "system"],
}


def setup_telemetry(settings: Settings, views: Sequence[View] = ()) -> None:
    """Поставить глобальные провайдеры. Без Collector экспорт падает в фоне, сервис работает."""
    if settings.otel_sdk_disabled:
        return
    # service.instance.id SDK генерирует на каждый процесс, поэтому реплики из `--scale`
    # с одинаковыми env различаются; OTEL_RESOURCE_ATTRIBUTES может задать его явно
    resource = Resource.create()

    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(tracer_provider)

    reader = PeriodicExportingMetricReader(OTLPMetricExporter())
    metrics.set_meter_provider(
        MeterProvider(resource=resource, metric_readers=[reader], views=views)
    )
    SystemMetricsInstrumentor(config=PROCESS_METRICS).instrument()

    # Лимит очереди нужен дашборду и алерту как знаменатель заполненности;
    # в метриках RabbitMQ аргументов очереди нет
    def queue_limit(_: CallbackOptions) -> Iterable[Observation]:
        if settings.queue_max_length > 0:
            yield Observation(settings.queue_max_length, {"queue": settings.queue_name})

    metrics.get_meter(__name__).create_observable_gauge(
        "scoring.queue.max_length", [queue_limit], unit="{message}"
    )


def shutdown_telemetry() -> None:
    """Дослать буферы. Источник живёт недолго, без этого он потеряет последние спаны и счётчики."""
    tracer_provider = trace.get_tracer_provider()
    if isinstance(tracer_provider, TracerProvider):
        tracer_provider.shutdown()
    meter_provider = metrics.get_meter_provider()
    if isinstance(meter_provider, MeterProvider):
        meter_provider.shutdown()
