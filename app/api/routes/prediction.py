"""Prediction and health API routes."""

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse

from app.schemas.prediction import HealthResponse, PredictionRequest, PredictionResponse
from app.services.prediction_service import ModelUnavailableError, prediction_service

router = APIRouter(tags=["prediction"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse | JSONResponse:
    """Report whether the saved fraud model is available for predictions."""

    if prediction_service.model_loaded:
        return HealthResponse(status="healthy", model_loaded=True)

    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"status": "unhealthy", "model_loaded": False},
    )


@router.post("/predict", response_model=PredictionResponse)
def predict(transaction: PredictionRequest) -> PredictionResponse:
    """Predict fraud risk for one validated transaction."""

    try:
        return prediction_service.predict(transaction)
    except ModelUnavailableError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error
