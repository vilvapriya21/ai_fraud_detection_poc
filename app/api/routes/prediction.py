"""Fraud-risk prediction API route."""

from fastapi import APIRouter, HTTPException, status

from app.schemas.prediction import PredictionRequest, PredictionResponse
from app.services.prediction_service import ModelUnavailableError, prediction_service


router = APIRouter(tags=["prediction"])


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
