"""API на TestRabbitBroker: задача публикуется и читается по Location."""

from asgi_lifespan import LifespanManager
from faststream.rabbit import TestRabbitBroker
from httpx import ASGITransport, AsyncClient

from scoring.api import create_app
from scoring.topology import Task
from tests.conftest import observation


async def test_task_published_and_readable(settings, sessions):
    app = create_app(settings)
    published: list[Task] = []

    @app.state.broker.subscriber(settings.queue_name)
    async def sink(task: Task) -> None:
        published.append(task)

    async with (
        TestRabbitBroker(app.state.broker),
        LifespanManager(app),
        AsyncClient(transport=ASGITransport(app), base_url="http://test") as client,
    ):
        created = await client.post("/api/v1/tasks", json=observation())
        assert created.status_code == 202
        assert created.json()["status"] == "pending"
        assert [str(t.id) for t in published] == [created.json()["id"]]

        fetched = await client.get(created.headers["location"])
        assert fetched.status_code == 200
        assert fetched.json()["id"] == created.json()["id"]


async def test_unknown_task_is_404(settings, sessions):
    app = create_app(settings)
    async with (
        TestRabbitBroker(app.state.broker),
        LifespanManager(app),
        AsyncClient(transport=ASGITransport(app), base_url="http://test") as client,
    ):
        response = await client.get("/api/v1/tasks/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404


async def test_invalid_observation_is_422(settings, sessions):
    app = create_app(settings)
    async with (
        TestRabbitBroker(app.state.broker),
        LifespanManager(app),
        AsyncClient(transport=ASGITransport(app), base_url="http://test") as client,
    ):
        response = await client.post("/api/v1/tasks", json={"day": 1, "farms": []})
    assert response.status_code == 422
