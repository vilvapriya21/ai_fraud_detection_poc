"""Schemas for the LangGraph fraud-investigation workflow."""

from typing import Any, Literal

from pydantic import BaseModel, Field

from app.schemas.prediction import PredictionRequest


class AgentInvestigationRequest(BaseModel):
    """Input for an agent-assisted investigation of a reported transaction."""

    question: str = Field(min_length=1, max_length=2_000)
    transaction_description: str = Field(min_length=1, max_length=5_000)
    transaction: PredictionRequest | None = None


class AgentInvestigationResponse(BaseModel):
    """The shared-state output of a completed LangGraph investigation."""

    case_id: str
    route_taken: Literal["direct_assessment", "multi_agent"]
    agent_findings: dict[str, str]
    tools_used: list[str]
    final_investigation_summary: str
    sources: list[dict[str, Any]]
    evidence: list[str]
    tool_failures: list[str]
