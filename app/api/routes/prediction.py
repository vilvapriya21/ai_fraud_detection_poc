"""Prediction and health API routes."""

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse

from app.schemas.prediction import HealthResponse, PredictionRequest, PredictionResponse
from app.schemas.agent_investigation import AgentInvestigationRequest, AgentInvestigationResponse
from app.schemas.explainability import ExplainResponse, FairnessResponse
from app.schemas.investigation import InvestigationRequest, InvestigationResponse
from app.schemas.similar_case import SimilarCasesRequest, SimilarCasesResponse
from app.schemas.security import SecurityBlockedResponse
from app.services.model_comparison_service import (
    ModelComparisonUnavailableError,
    model_comparison_service,
)
from app.services.prediction_service import ModelUnavailableError, prediction_service
from app.services.agent_investigation_service import agent_investigation_service
from app.services.explainability_service import ExplainabilityUnavailableError, explainability_service
from app.services.investigation_service import InvestigationUnavailableError, investigation_service
from app.services.similar_case_service import SimilarCaseUnavailableError, similar_case_service
from app.services.security_service import security_service

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


@router.post("/investigate", response_model=InvestigationResponse | SecurityBlockedResponse)
def investigate(request: InvestigationRequest) -> InvestigationResponse | JSONResponse:
    """Provide source-grounded local investigation context without retraining assets."""

    decision = security_service.validate_investigation_input(
        request.question,
        request.transaction_description,
        endpoint="/investigate",
    )
    if not decision.allowed:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=security_service.blocked_response(decision.blocked_category or "unsafe_content"),
        )
    try:
        response = investigation_service.investigate(
            decision.question,
            decision.transaction_description,
        )
    except InvestigationUnavailableError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error
    return InvestigationResponse(**response)


@router.post("/agent-investigate", response_model=AgentInvestigationResponse | SecurityBlockedResponse)
def agent_investigate(request: AgentInvestigationRequest) -> AgentInvestigationResponse | JSONResponse:
    """Run the LangGraph investigation workflow using existing application tools."""

    decision = security_service.validate_investigation_input(
        request.question,
        request.transaction_description,
        endpoint="/agent-investigate",
    )
    if not decision.allowed:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=security_service.blocked_response(decision.blocked_category or "unsafe_content"),
        )
    response = agent_investigation_service.investigate(
        decision.question,
        decision.transaction_description,
        request.transaction,
    )
    return AgentInvestigationResponse(**response)


@router.post("/explain", response_model=ExplainResponse)
def explain(transaction: PredictionRequest) -> ExplainResponse:
    """Return saved-model prediction evidence using local SHAP contributions."""

    try:
        response = explainability_service.explain(transaction)
    except ExplainabilityUnavailableError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
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
