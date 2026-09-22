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
