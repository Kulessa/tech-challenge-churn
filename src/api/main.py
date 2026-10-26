"""FastAPI endpoints for churn model health and inference."""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import pandas as pd
from fastapi import FastAPI, HTTPException, Request

from src.config import METADATA_PATH, MODEL_PATH
from src.api.schemas import CustomerData, PredictionResponse
from src.predict import LoadedModel, load_model, predict_churn


def create_app(
    model_path: Path = MODEL_PATH,
    metadata_path: Path = METADATA_PATH,
) -> FastAPI:
    """Create the API application with configurable model artifact paths."""

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.loaded_model = (
            load_model(model_path, metadata_path) if model_path.is_file() else None
        )
        yield

    application = FastAPI(
        title="Telco Churn Prediction API",
        description="Estimates customer churn probability from telecom service data.",
        version="1.0.0",
        lifespan=lifespan,
    )

    @application.get("/health")
    def health(request: Request) -> dict[str, str | bool]:
        """Report API availability and model loading status."""
        loaded_model: LoadedModel | None = request.app.state.loaded_model
        return {"status": "ok", "model_loaded": loaded_model is not None}

    @application.post("/predict", response_model=PredictionResponse)
    def predict(
        customer: CustomerData,
        request: Request,
    ) -> PredictionResponse:
        """Return a churn score and threshold-based prediction."""
        loaded_model: LoadedModel | None = request.app.state.loaded_model
        if loaded_model is None:
            raise HTTPException(
                status_code=503,
                detail="Model is unavailable. Train it with `python -m src.train`.",
            )

        customer_frame = pd.DataFrame([customer.model_dump()])
        result = predict_churn(
            loaded_model.pipeline,
            customer_frame,
            threshold=loaded_model.threshold,
        )
        return PredictionResponse(
            **result,
            model_version=loaded_model.model_version,
        )

    return application


app = create_app()
