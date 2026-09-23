"""Tests for saved-model SHAP explanations and fairness API responses."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def prediction_payload() -> dict[str, object]:
    """Return one valid transaction request shared by prediction and explain tests."""

    return {
        "hour_of_day": 2,
        "is_weekend": 0,
        "is_night_transaction": 1,
        "country": "USA",
        "city": "New York",
        "merchant_category": "Crypto Exchange",
        "payment_method": "Credit Card",
        "device_type": "Mobile",
        "customer_age": 35,
        "credit_score": 620,
        "account_age_years": 2.0,
        "account_balance": 1500.0,
        "transaction_amount": 450.0,
        "num_prev_transactions": 20,
        "transaction_freq_monthly": 15,
        "distance_from_home_km": 80.0,
        "time_since_last_txn_hrs": 0.5,
        "is_international": 1,
        "failed_attempts": 3,
        "pin_changed_recently": 1,
    }


def test_explain_returns_prediction_and_five_shap_features() -> None:
    """The explain endpoint returns model evidence with signed feature directions."""

    response = client.post("/explain", json=prediction_payload())

    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in {"Fraudulent", "Legitimate"}
    assert 0 <= body["fraud_probability"] <= 1
    assert len(body["top_contributing_features"]) == 5
    assert {"feature", "feature_value", "shap_value", "direction"}.issubset(
        body["top_contributing_features"][0]
    )
    assert {feature["direction"] for feature in body["top_contributing_features"]}.issubset(
        {"higher", "lower"}
    )


def test_fairness_returns_existing_operational_groups_and_limitations() -> None:
    """The fairness endpoint reports bounded group rates plus limitations."""

    response = client.get("/fairness")

    assert response.status_code == 200
    body = response.json()
    assert body["evaluation_rows"] > 0
    assert set(body["groups_checked"]) == {
        "country",
        "device_type",
        "merchant_category",
        "age_band",
    }
    assert body["limitations"]
    assert body["summaries"]["country"]["groups"]
