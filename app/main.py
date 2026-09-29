"""FastAPI application entry point for fraud prediction."""

import os
from dotenv import load_dotenv

load_dotenv()

from time import perf_counter

from fastapi import FastAPI, Request
from starlette.middleware.base import RequestResponseEndpoint
from starlette.middleware.cors import CORSMiddleware

from app.api.routes.analysis import router as analysis_router
from app.api.routes.investigation import router as investigation_router
from app.api.routes.operations import router as operations_router
from app.api.routes.prediction import router as prediction_router
from app.services.monitoring_service import monitoring_service

app = FastAPI(title="AI Fraud Detection & Investigation API")


def cors_allowed_origins() -> list[str]:
    """Read comma-separated browser origins allowed to call the API directly."""

    default_origins = "http://localhost:5173,http://127.0.0.1:5173"
    configured_origins = os.getenv("CORS_ALLOWED_ORIGINS", default_origins)
    return [origin.strip() for origin in configured_origins.split(",") if origin.strip()]


app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_allowed_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["Content-Type", "Authorization"],
)


@app.middleware("http")
async def monitor_prediction_request(
    request: Request,
    call_next: RequestResponseEndpoint,
):
    """Capture in-memory latency and error metrics for ``/predict`` only."""

    if request.url.path != "/predict":
        return await call_next(request)

    start_time = perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        monitoring_service.record_prediction_request(
            latency_ms=(perf_counter() - start_time) * 1_000,
            is_error=status_code >= 400,
        )


app.include_router(operations_router)
app.include_router(prediction_router)
app.include_router(analysis_router)
app.include_router(investigation_router)
