"""Inference-сервис: модель грузится из MLflow один раз при старте.

Запуск: `uv run uvicorn farmvalue.service.main:app --port 8000`
"""

import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Request

from farmvalue.service.config import Settings, get_settings
from farmvalue.service.model import LoadedModel, load_from_registry
from farmvalue.service.schemas import ModelMeta, Prediction, ProcessRequest, ProcessResponse

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    settings: Settings = app.state.settings
    if getattr(app.state, "model", None) is None:
        app.state.model = await asyncio.to_thread(
            load_from_registry,
            settings.mlflow_tracking_uri,
            settings.model_name,
            settings.model_alias,
        )
    yield


def get_model(request: Request) -> LoadedModel:
    model: LoadedModel = request.app.state.model
    return model


ModelDep = Annotated[LoadedModel, Depends(get_model)]


def model_meta(model: LoadedModel) -> ModelMeta:
    info = model.info
    return ModelMeta(
        name=info.name,
        alias=info.alias,
        version=info.version,
        run_id=info.run_id,
        model_id=info.model_id,
    )


def create_app(settings: Settings | None = None, model: LoadedModel | None = None) -> FastAPI:
    settings = settings or get_settings()
    logging.basicConfig(
        level=settings.log_level, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )
    app = FastAPI(title="farmvalue", lifespan=lifespan)
    app.state.settings = settings
    app.state.model = model

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/model")
    async def model_info(model: ModelDep) -> ModelMeta:
        return model_meta(model)

    @app.post("/process")
    async def process(body: ProcessRequest, model: ModelDep) -> ProcessResponse:
        observations = [obs.model_dump() for obs in body.observations]
        probs = await asyncio.to_thread(model.win_probability, observations)
        predictions = [
            Prediction(
                day=obs.day,
                hour=obs.hour,
                win_prob_seat0=p,
                win_prob_player=None if obs.player is None else (p if obs.player == 0 else 1 - p),
            )
            for obs, p in zip(body.observations, probs, strict=True)
        ]
        return ProcessResponse(model=model_meta(model), predictions=predictions)

    return app


app = create_app()
