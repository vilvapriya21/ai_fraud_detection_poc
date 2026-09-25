"""Operational health and runtime-metrics API routes."""

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from app.schemas.prediction import HealthResponse
from app.services.monitoring_service import monitoring_service
from app.services.prediction_service import prediction_service


router = APIRouter(tags=["operations"])


@router.get("/metrics")
def metrics() -> dict[str, int | float | None]:
    """Return in-memory runtime metrics for prediction requests."""

    return monitoring_service.get_metrics()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse | JSONResponse:
    """Report whether the saved fraud model is available for predictions."""

    if prediction_service.model_loaded:
        return HealthResponse(status="healthy", model_loaded=True)

    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"status": "unhealthy", "model_loaded": False},
    )
