"""Model-comparison, explainability, and fairness API routes."""

from fastapi import APIRouter, HTTPException, status

from app.schemas.explainability import ExplainResponse, FairnessResponse
from app.schemas.prediction import PredictionRequest
from app.services.explainability_service import ExplainabilityUnavailableError, explainability_service
from app.services.model_comparison_service import (
    ModelComparisonUnavailableError,
    model_comparison_service,
)


router = APIRouter(tags=["analysis"])


@router.get("/model-comparison")
def model_comparison() -> dict:
    """Return saved text-model comparison results without training models."""

    try:
        return model_comparison_service.get_results()
    except ModelComparisonUnavailableError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="SHAP explanation is temporarily unavailable.",
        ) from error


@router.post("/explain", response_model=ExplainResponse)
def explain(transaction: PredictionRequest) -> ExplainResponse:
    """Return saved-model prediction evidence using local SHAP contributions."""

    try:
        response = explainability_service.explain(transaction)
    except ExplainabilityUnavailableError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Fairness analysis is temporarily unavailable.",
        ) from error
    return ExplainResponse(**response)


@router.get("/fairness", response_model=FairnessResponse)
def fairness() -> FairnessResponse:
    """Return cached basic group-rate comparisons for the processed hold-out data."""

    try:
        response = explainability_service.fairness_summary()
    except ExplainabilityUnavailableError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error
    return FairnessResponse(**response)
