"""Schemas for similar historical fraud case retrieval."""

from pydantic import BaseModel, Field


class SimilarCasesRequest(BaseModel):
    """A transaction description to compare with historical case descriptions."""

    description: str = Field(min_length=1, max_length=5_000)
    top_k: int = Field(default=5, ge=1, le=10)


class SimilarCaseResult(BaseModel):
    """One historical case returned by vector similarity search."""

    case_id: str
    fraud_type: str
    key_observations: str
    outcome: str
    similarity_score: float


class SimilarCasesResponse(BaseModel):
    """The ranked results of a historical case search."""

    cases: list[SimilarCaseResult]
