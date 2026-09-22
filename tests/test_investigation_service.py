"""Tests for the local, source-grounded investigation RAG service."""

from app.services.investigation_service import InvestigationService


RELEVANT_DESCRIPTION = (
    "An international night-time transaction at a crypto exchange followed several "
    "failed attempts and a recent PIN change."
)


def test_investigation_retrieves_relevant_local_context() -> None:
    """A transaction-control question retrieves policy or guideline context."""

    response = InvestigationService().investigate(
        "What account-control checks should be reviewed?",
        RELEVANT_DESCRIPTION,
    )

    assert not response["evidence_insufficient"]
    assert response["relevant_context"]
    assert any(source["document_type"] in {"policy", "guideline"} for source in response["sources"])


def test_investigation_returns_traceable_sources() -> None:
    """Retrieved context includes visible document identifiers and similarity scores."""

    response = InvestigationService().investigate(
        "How should transaction evidence be recorded?",
        RELEVANT_DESCRIPTION,
    )

    assert response["sources"]
    assert all(source["document_id"] and source["similarity_score"] > 0 for source in response["sources"])


def test_investigation_marks_irrelevant_input_as_insufficient() -> None:
    """Unrelated input must not be turned into an unsupported investigation conclusion."""

    response = InvestigationService().investigate(
        "How should I fertilize tomatoes in a backyard garden?",
        "The vegetable seedlings were watered after a sunny afternoon.",
    )

    assert response["evidence_insufficient"]
    assert not response["sources"]
    assert "insufficient" in response["investigation_response"].lower()
