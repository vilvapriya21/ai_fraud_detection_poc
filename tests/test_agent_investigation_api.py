"""Tests for the LangGraph agent-investigation API endpoint."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_agent_investigate_returns_direct_assessment() -> None:
    """The API returns a public shared-state response for a simple case."""

    response = client.post(
        "/agent-investigate",
        json={
            "question": "What should be recorded during review?",
            "transaction_description": "A daytime domestic grocery transaction was reported on a card terminal.",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["route_taken"] == "direct_assessment"
    assert body["tools_used"] == ["investigation_rag_service"]
    assert body["final_investigation_summary"]


def test_get_agent_investigation_returns_completed_case() -> None:
    """A completed agent investigation can be retrieved by its case identifier."""

    created = client.post(
        "/agent-investigate",
        json={
            "question": "What should be recorded during review?",
            "transaction_description": "A daytime domestic grocery transaction was reported on a card terminal.",
        },
    )

    assert created.status_code == 200
    retrieved = client.get(f"/agent-investigate/{created.json()['case_id']}")

    assert retrieved.status_code == 200
    assert retrieved.json() == created.json()


def test_get_agent_investigation_returns_404_for_unknown_case() -> None:
    """Unknown case identifiers return the public not-found response."""

    response = client.get("/agent-investigate/not-a-stored-case")

    assert response.status_code == 404
    assert response.json() == {"detail": "Case not found."}
