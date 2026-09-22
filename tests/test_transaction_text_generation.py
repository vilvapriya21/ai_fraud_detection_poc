"""Tests for deterministic transaction text generation."""

import pandas as pd

from scripts.generate_transaction_text_dataset import (
    contains_label_leakage,
    generate_transaction_text,
    stratified_sample,
    validate_text_dataset,
)


def make_transaction() -> pd.Series:
    """Create a representative non-target transaction record."""

    return pd.Series(
        {
            "transaction_id": "TXN001",
            "hour_of_day": 3,
            "is_weekend": 1,
            "is_night_transaction": 1,
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
            "failed_attempts": 1,
            "pin_changed_recently": 0,
            "is_fraud": 1,
            "fraud_type": "Account Takeover",
        }
    )


def test_generated_text_is_non_empty_and_has_no_label_leakage() -> None:
    """Text is populated without prohibited target-revealing words."""

    text = generate_transaction_text(make_transaction())

    assert text.strip()
    assert not contains_label_leakage(text)


def test_text_generation_is_deterministic() -> None:
    """The same transaction record always produces the same description."""

    transaction = make_transaction()

    assert generate_transaction_text(transaction) == generate_transaction_text(transaction)


def test_transaction_identifier_is_not_in_generated_text() -> None:
    """The output text does not expose the transaction identifier."""

    transaction = make_transaction()

    assert transaction["transaction_id"] not in generate_transaction_text(transaction)


def test_stratified_sample_retains_both_classes() -> None:
    """Stratified sampling preserves both binary target classes."""

    source_data = pd.DataFrame(
        [
            {"transaction_id": f"TXN{index}", "is_fraud": 0}
            for index in range(16)
        ]
        + [
            {"transaction_id": f"TXN{index}", "is_fraud": 1}
            for index in range(16, 20)
        ]
    )

    sampled_data = stratified_sample(source_data, sample_size=10, random_state=42)

    assert set(sampled_data["is_fraud"].unique()) == {0, 1}


def test_validation_rejects_label_leakage() -> None:
    """Validation rejects prohibited language if it reaches the output table."""

    output_data = pd.DataFrame(
        {
            "transaction_id": ["TXN001"],
            "transaction_text": ["This transaction is suspicious."],
            "is_fraud": [0],
        }
    )

    try:
        validate_text_dataset(output_data)
    except ValueError as error:
        assert "target-leakage" in str(error)
    else:
        raise AssertionError("Expected validation to reject label leakage.")
