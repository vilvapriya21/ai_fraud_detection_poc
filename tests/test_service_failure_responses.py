"""Focused API tests for unavailable prediction and investigation dependencies."""

from fastapi.testclient import TestClient

from app.main import app
from app.services.agent_investigation_service import AgentInvestigationUnavailableError
from app.services.explainability_service import ExplainabilityUnavailableError
from app.services.investigation_service import InvestigationUnavailableError
from app.services.prediction_service import ModelUnavailableError
from app.services.similar_case_service import SimilarCaseUnavailableError


client = TestClient(app)
VALID_TRANSACTION = {
    "hour_of_day": 14,
    "is_weekend": 0,
    "is_night_transaction": 0,
    "country": "USA",
    "city": "New York",
    "merchant_category": "Grocery",
    "payment_method": "Credit Card",
    "device_type": "Mobile",
    "customer_age": 35,
    "credit_score": 720,
    "account_age_years": 5.0,
    "account_balance": 5000.0,
    "transaction_amount": 120.0,
    "num_prev_transactions": 50,
    "transaction_freq_monthly": 12,
    "distance_from_home_km": 4.0,
    "time_since_last_txn_hrs": 8.0,
    "is_international": 0,
    "failed_attempts": 0,
    "pin_changed_recently": 0,
}


def test_prediction_failure_returns_sanitized_503(monkeypatch) -> None:
    """Model failures are unavailable responses without internal exception details."""

    def fail_prediction(transaction):
        raise ModelUnavailableError("secret model path and internal failure")

    monkeypatch.setattr(
        "app.api.routes.prediction.prediction_service.predict",
        fail_prediction,
    )
    response = client.post("/predict", json=VALID_TRANSACTION)

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Fraud prediction is temporarily unavailable."
    }


def test_shap_failure_returns_sanitized_503(monkeypatch) -> None:
    """SHAP failures are reported as a clear service-unavailable response."""

    def fail_explanation(transaction):
        raise ExplainabilityUnavailableError("private SHAP exception")

    monkeypatch.setattr(
        "app.api.routes.analysis.explainability_service.explain",
        fail_explanation,
    )
    response = client.post("/explain", json=VALID_TRANSACTION)

    assert response.status_code == 503
    assert response.json() == {
        "detail": "SHAP explanation is temporarily unavailable."
    }


def test_faiss_failure_returns_sanitized_503(monkeypatch) -> None:
    """FAISS failures do not leak index paths or library exception details."""

    def fail_search(description, top_k):
        raise SimilarCaseUnavailableError("/private/index.faiss failed")

    monkeypatch.setattr(
        "app.api.routes.investigation.similar_case_service.search",
        fail_search,
    )
    response = client.post(
        "/similar-cases",
        json={"description": "A reported transaction needs review.", "top_k": 3},
    )

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Similar-case search is temporarily unavailable."
    }


def test_rag_asset_failure_returns_sanitized_503(monkeypatch) -> None:
    """Required RAG retrieval failures produce a stable unavailable response."""

    def fail_investigation(question, transaction_description):
        raise InvestigationUnavailableError("sensitive retrieval failure")

    monkeypatch.setattr(
        "app.api.routes.investigation.investigation_service.investigate",
        fail_investigation,
    )
    response = client.post(
        "/investigate",
        json={
            "question": "What should be reviewed?",
            "transaction_description": "A reported transaction needs review.",
        },
    )

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Investigation retrieval is temporarily unavailable."
    }


def test_agent_workflow_failure_returns_sanitized_503(monkeypatch) -> None:
    """Agent orchestration failures are service failures, not unhandled errors."""

    def fail_agent(question, transaction_description, transaction):
        raise AgentInvestigationUnavailableError("private graph exception")

    monkeypatch.setattr(
        "app.api.routes.investigation.agent_investigation_service.investigate",
        fail_agent,
    )
    response = client.post(
        "/agent-investigate",
        json={
            "question": "What should be reviewed?",
            "transaction_description": "A reported transaction needs review.",
        },
    )

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Agent investigation is temporarily unavailable."
    }


def test_agent_required_tool_failure_returns_503(monkeypatch) -> None:
    """A recorded required-tool failure prevents a misleading successful response."""

    def incomplete_agent_response(question, transaction_description, transaction):
        return {"tool_failures": ["investigation_rag_service unavailable"]}

    monkeypatch.setattr(
        "app.api.routes.investigation.agent_investigation_service.investigate",
        incomplete_agent_response,
    )
    response = client.post(
        "/agent-investigate",
        json={
            "question": "What should be reviewed?",
            "transaction_description": "A reported transaction needs review.",
        },
    )

    assert response.status_code == 503
    assert response.json() == {
        "detail": (
            "Agent investigation is temporarily unavailable because "
            "a required tool failed."
        )
    }
