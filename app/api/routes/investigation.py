"""Similar-case and investigation workflow API routes."""

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse

from app.schemas.agent_investigation import AgentInvestigationRequest, AgentInvestigationResponse
from app.schemas.investigation import InvestigationRequest, InvestigationResponse
from app.schemas.security import SecurityBlockedResponse
from app.schemas.similar_case import SimilarCasesRequest, SimilarCasesResponse
from app.services.agent_investigation_service import (
    AgentInvestigationUnavailableError,
    agent_investigation_service,
)
from app.services.investigation_service import InvestigationUnavailableError, investigation_service
from app.services.security_service import security_service
from app.services.similar_case_service import SimilarCaseUnavailableError, similar_case_service


router = APIRouter(tags=["investigation"])


@router.post("/similar-cases", response_model=SimilarCasesResponse)
def similar_cases(request: SimilarCasesRequest) -> SimilarCasesResponse:
    """Retrieve saved historical cases using semantic vector similarity."""

    try:
        cases = similar_case_service.search(request.description, request.top_k)
    except SimilarCaseUnavailableError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Similar-case search is temporarily unavailable.",
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
            detail="Investigation retrieval is temporarily unavailable.",
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
    try:
        response = agent_investigation_service.investigate(
            decision.question,
            decision.transaction_description,
            request.transaction,
        )
    except AgentInvestigationUnavailableError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent investigation is temporarily unavailable.",
        ) from error
    if response["tool_failures"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Agent investigation is temporarily unavailable because "
                "a required tool failed."
            ),
        )
    return AgentInvestigationResponse(**response)
