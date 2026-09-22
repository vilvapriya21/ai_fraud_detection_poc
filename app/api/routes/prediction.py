"""Prediction and health API routes."""

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse

from app.schemas.prediction import HealthResponse, PredictionRequest, PredictionResponse
from app.schemas.agent_investigation import AgentInvestigationRequest, AgentInvestigationResponse
from app.schemas.investigation import InvestigationRequest, InvestigationResponse
from app.schemas.similar_case import SimilarCasesRequest, SimilarCasesResponse
from app.services.model_comparison_service import (
    ModelComparisonUnavailableError,
    model_comparison_service,
)
from app.services.prediction_service import ModelUnavailableError, prediction_service
from app.services.agent_investigation_service import agent_investigation_service
from app.services.investigation_service import InvestigationUnavailableError, investigation_service
from app.services.similar_case_service import SimilarCaseUnavailableError, similar_case_service

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


@router.get("/model-comparison")
def model_comparison() -> dict:
    """Return saved text-model comparison results without training models."""

    try:
        return model_comparison_service.get_results()
    except ModelComparisonUnavailableError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error


@router.post("/similar-cases", response_model=SimilarCasesResponse)
def similar_cases(request: SimilarCasesRequest) -> SimilarCasesResponse:
    """Retrieve saved historical cases using semantic vector similarity."""

    try:
        cases = similar_case_service.search(request.description, request.top_k)
    except SimilarCaseUnavailableError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error
    return SimilarCasesResponse(cases=cases)


@router.post("/investigate", response_model=InvestigationResponse)
def investigate(request: InvestigationRequest) -> InvestigationResponse:
    """Provide source-grounded local investigation context without retraining assets."""

    try:
        response = investigation_service.investigate(
            request.question,
            request.transaction_description,
        )
    except InvestigationUnavailableError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error
    return InvestigationResponse(**response)


@router.post("/agent-investigate", response_model=AgentInvestigationResponse)
def agent_investigate(request: AgentInvestigationRequest) -> AgentInvestigationResponse:
    """Run the LangGraph investigation workflow using existing application tools."""

    response = agent_investigation_service.investigate(
        request.question,
        request.transaction_description,
        request.transaction,
    )
    return AgentInvestigationResponse(**response)
