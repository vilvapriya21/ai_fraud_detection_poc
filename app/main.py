"""FastAPI application entry point for fraud prediction."""

from time import perf_counter

from fastapi import FastAPI, Request
from starlette.middleware.base import RequestResponseEndpoint

from app.api.routes.prediction import router as prediction_router
from app.services.monitoring_service import monitoring_service

app = FastAPI(title="AI Fraud Detection & Investigation API")


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


app.include_router(prediction_router)
