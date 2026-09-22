"""Tests for the grounded fraud investigation API endpoint."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_investigate_returns_sources_and_evidence() -> None:
    """The API exposes evidence, next steps, and source metadata."""

    response = client.post(
        "/investigate",
        json={
            "question": "What should be reviewed for a possible account-control issue?",
            "transaction_description": (
                "An international night-time transaction at a crypto exchange followed "
                "several failed attempts and a recent PIN change."
            ),
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["observed_evidence"]
    assert body["recommended_next_steps"]
    assert body["sources"]
    assert not body["evidence_insufficient"]
