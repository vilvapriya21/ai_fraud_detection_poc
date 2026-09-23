"""Schemas for grounded fraud-investigation retrieval responses."""

from typing import Literal

from pydantic import BaseModel, Field


class InvestigationRequest(BaseModel):
    """A question and reported transaction details for investigation support."""

    question: str = Field(min_length=1, max_length=2_000)
    transaction_description: str = Field(min_length=1, max_length=5_000)


class InvestigationSource(BaseModel):
    """A visible local knowledge-base source used in an investigation response."""

    document_id: str
    title: str
    document_type: str
    excerpt: str
    similarity_score: float


class InvestigationResponse(BaseModel):
    """A source-grounded response for a fraud investigation question."""

    investigation_response: str
    observed_evidence: list[str]
    relevant_context: list[str]
    recommended_next_steps: list[str]
    sources: list[InvestigationSource]
    evidence_insufficient: bool
    generation_mode: Literal["llm", "fallback"]
