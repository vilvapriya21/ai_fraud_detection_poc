"""Tests for the similar historical fraud case API endpoint."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_similar_cases_returns_ranked_case_records() -> None:
    """The endpoint returns saved records with real similarity scores."""

    response = client.post(
        "/similar-cases",
        json={
            "description": (
                "An international night-time transaction used a mobile device at a "
                "crypto exchange after several failed attempts."
            ),
            "top_k": 3,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body["cases"]) == 3
    assert {
        "case_id",
        "fraud_type",
        "key_observations",
        "outcome",
        "similarity_score",
    }.issubset(body["cases"][0])
    scores = [case["similarity_score"] for case in body["cases"]]
    assert scores == sorted(scores, reverse=True)


def test_similar_cases_rejects_invalid_top_k() -> None:
    """The endpoint validates the bounded retrieval count."""

    response = client.post(
        "/similar-cases",
        json={"description": "A transaction description", "top_k": 0},
    )

    assert response.status_code == 422
