"""Tests for the LangGraph multi-agent investigation workflow."""

from app.schemas.prediction import PredictionRequest
from app.services.agent_investigation_service import AgentInvestigationService
from app.services.similar_case_service import SimilarCaseUnavailableError


def complex_transaction() -> PredictionRequest:
    """Return valid structured inputs that require the multi-agent route."""

    return PredictionRequest(
        hour_of_day=2,
        is_weekend=0,
        is_night_transaction=1,
        country="USA",
        city="New York",
        merchant_category="Crypto Exchange",
        payment_method="Credit Card",
        device_type="Mobile",
        customer_age=35,
        credit_score=620,
        account_age_years=2.0,
        account_balance=1_500.0,
        transaction_amount=450.0,
        num_prev_transactions=20,
        transaction_freq_monthly=15,
        distance_from_home_km=80.0,
        time_since_last_txn_hrs=0.5,
        is_international=1,
        failed_attempts=3,
        pin_changed_recently=1,
    )


def test_simple_case_uses_direct_assessment_route() -> None:
    """Routine descriptions use a direct RAG assessment rather than all agents."""

    service = AgentInvestigationService()
    response = service.investigate(
        "What should be recorded during review?",
        "A daytime domestic grocery transaction was reported on a card terminal.",
    )

    assert response["route_taken"] == "direct_assessment"
    assert "Direct Assessment" in response["agent_findings"]
    assert "investigation_rag_service" in response["tools_used"]
    assert service.get_case(response["case_id"]) == response


def test_complex_case_runs_all_specialist_agents() -> None:
    """High-risk structured inputs run the prediction, evidence, and summary agents."""

    response = AgentInvestigationService().investigate(
        "What should be reviewed for possible account compromise?",
        "An international night-time crypto transaction followed failed attempts and a recent PIN change.",
        complex_transaction(),
    )

    assert response["route_taken"] == "multi_agent"
    assert {"Fraud Analysis Agent", "Similar Case / Evidence Agent"}.issubset(response["agent_findings"])
    assert {"fraud_prediction_service", "similar_case_service", "investigation_rag_service"}.issubset(response["tools_used"])
    assert response["sources"]


class FailingSimilarCaseTool:
    """Test double that exercises graceful similar-case tool failure handling."""

    def search(self, description: str, top_k: int = 3) -> list[dict[str, str]]:
        """Simulate an unavailable persisted similar-case index."""

        raise SimilarCaseUnavailableError("index unavailable for test")


def test_tool_failure_produces_qualified_summary() -> None:
    """A failing tool is recorded while the remaining workflow completes."""

    service = AgentInvestigationService(similar_case_tool=FailingSimilarCaseTool())
    response = service.investigate(
        "What should be reviewed for possible account compromise?",
        "An international night-time crypto transaction followed failed attempts and a recent PIN change.",
        complex_transaction(),
    )

    assert any("similar_case_service" in failure for failure in response["tool_failures"])
    assert "incomplete" in response["final_investigation_summary"].lower()
    assert "investigation_rag_service" in response["tools_used"]
