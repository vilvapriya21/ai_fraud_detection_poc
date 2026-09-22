"""Tests for persisted semantic historical-case retrieval."""

from app.services.similar_case_service import SimilarCaseService


def test_exact_case_description_retrieves_its_stored_case() -> None:
    """A saved case description should retrieve the same record as the best match."""

    service = SimilarCaseService()
    cases, _ = service._load_assets()
    source_case = cases[0]

    results = service.search(source_case["description"], top_k=3)

    assert results[0]["case_id"] == source_case["case_id"]
    assert results[0]["similarity_score"] > 0.99
    assert len(results) == 3
