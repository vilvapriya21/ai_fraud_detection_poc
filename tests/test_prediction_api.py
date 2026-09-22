"""Tests for the fraud prediction API."""

from fastapi.testclient import TestClient

from app.main import app

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


def test_health_endpoint_reports_loaded_model() -> None:
    """The health endpoint reports model availability."""

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "model_loaded": True}


def test_predict_returns_complete_response() -> None:
    """A valid transaction returns a bounded fraud prediction response."""

    response = client.post("/predict", json=VALID_TRANSACTION)

    assert response.status_code == 200
    response_body = response.json()
    assert response_body["prediction"] in {"Fraudulent", "Legitimate"}
    assert 0.0 <= response_body["fraud_probability"] <= 1.0
    assert response_body["risk_level"] in {"Low", "Medium", "High"}


def test_predict_rejects_invalid_hour() -> None:
    """Hours outside the derived-feature range are rejected."""

    invalid_transaction = {**VALID_TRANSACTION, "hour_of_day": 24}

    response = client.post("/predict", json=invalid_transaction)

    assert response.status_code == 422


def test_predict_rejects_negative_transaction_amount() -> None:
    """Negative monetary values are rejected before model inference."""

    invalid_transaction = {**VALID_TRANSACTION, "transaction_amount": -1.0}

    response = client.post("/predict", json=invalid_transaction)

    assert response.status_code == 422
