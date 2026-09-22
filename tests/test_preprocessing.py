"""Tests for bank fraud dataset preprocessing."""

import numpy as np
import pandas as pd

from app.ml.preprocessing import (
    build_preprocessor,
    identify_feature_columns,
    prepare_features_and_target,
    remove_exact_duplicates,
    split_features_and_target,
)


def make_sample_data() -> pd.DataFrame:
    """Create representative bank transactions for preprocessing tests."""

    rows = [
        {
            "transaction_id": "TXN001",
            "customer_id": "CUST001",
            "transaction_date": "2024-01-01",
            "transaction_time": "10:30:00",
            "hour_of_day": 10,
            "is_weekend": 0,
            "is_night_transaction": 0,
            "country": "USA",
            "city": "New York",
            "merchant_category": "Grocery",
            "payment_method": "Debit Card",
            "device_type": "Mobile",
            "customer_age": 35,
            "credit_score": 720,
            "account_age_years": 5.2,
            "account_balance": 5000.0,
            "transaction_amount": 120.5,
            "num_prev_transactions": 50,
            "transaction_freq_monthly": 12,
            "distance_from_home_km": 3.5,
            "time_since_last_txn_hrs": 10.0,
            "is_international": 0,
            "failed_attempts": 0,
            "pin_changed_recently": 0,
            "is_fraud": 0,
            "fraud_type": np.nan,
        },
        {
            "transaction_id": "TXN002",
            "customer_id": "CUST002",
            "transaction_date": "2024-01-02",
            "transaction_time": "02:15:00",
            "hour_of_day": 2,
            "is_weekend": 0,
            "is_night_transaction": 1,
            "country": "UK",
            "city": "London",
            "merchant_category": "ATM Withdrawal",
            "payment_method": "Credit Card",
            "device_type": np.nan,
            "customer_age": 42,
            "credit_score": 610,
            "account_age_years": 1.0,
            "account_balance": 1200.0,
            "transaction_amount": np.nan,
            "num_prev_transactions": 10,
            "transaction_freq_monthly": 20,
            "distance_from_home_km": 150.0,
            "time_since_last_txn_hrs": 1.0,
            "is_international": 1,
            "failed_attempts": 3,
            "pin_changed_recently": 1,
            "is_fraud": 1,
            "fraud_type": "Account Takeover",
        },
        {
            "transaction_id": "TXN003",
            "customer_id": "CUST003",
            "transaction_date": "2024-01-03",
            "transaction_time": "15:45:00",
            "hour_of_day": 15,
            "is_weekend": 0,
            "is_night_transaction": 0,
            "country": "Canada",
            "city": "Toronto",
            "merchant_category": "Healthcare",
            "payment_method": "Bank Transfer",
            "device_type": "Desktop",
            "customer_age": 29,
            "credit_score": 680,
            "account_age_years": 3.5,
            "account_balance": 8500.0,
            "transaction_amount": 500.0,
            "num_prev_transactions": 80,
            "transaction_freq_monthly": 15,
            "distance_from_home_km": 15.0,
            "time_since_last_txn_hrs": 24.0,
            "is_international": 0,
            "failed_attempts": 1,
            "pin_changed_recently": 0,
            "is_fraud": 0,
            "fraud_type": np.nan,
        },
    ]

    return pd.DataFrame(
        [
            *rows,
            rows[0].copy(),
        ]
    )


def dense_values(transformed: object) -> np.ndarray:
    """Convert dense or sparse transformer output to a NumPy array."""

    if hasattr(transformed, "toarray"):
        return transformed.toarray()

    return np.asarray(transformed)


def test_remove_exact_duplicates_preserves_input() -> None:
    data = make_sample_data()

    deduplicated = remove_exact_duplicates(data)

    assert len(data) == 4
    assert len(deduplicated) == 3


def test_identifiers_target_and_leakage_columns_are_excluded() -> None:
    features, target = prepare_features_and_target(
        make_sample_data()
    )

    assert "transaction_id" not in features.columns
    assert "customer_id" not in features.columns
    assert "transaction_date" not in features.columns
    assert "transaction_time" not in features.columns
    assert "fraud_type" not in features.columns
    assert "is_fraud" not in features.columns

    assert target.name == "is_fraud"
    assert target.tolist() == [0, 1, 0]


def test_split_rejects_missing_required_column() -> None:
    data = make_sample_data().drop(
        columns="customer_id"
    )

    try:
        split_features_and_target(data)
    except ValueError as error:
        assert "customer_id" in str(error)
    else:
        raise AssertionError(
            "Expected a missing-column ValueError"
        )


def test_feature_types_match_schema() -> None:
    features, _ = prepare_features_and_target(
        make_sample_data()
    )

    numeric_columns, categorical_columns = (
        identify_feature_columns(features)
    )

    assert "transaction_amount" in numeric_columns
    assert "failed_attempts" in numeric_columns
    assert "hour_of_day" in numeric_columns

    assert "merchant_category" in categorical_columns
    assert "payment_method" in categorical_columns
    assert "device_type" in categorical_columns


def test_preprocessor_handles_missing_values() -> None:
    features, _ = prepare_features_and_target(
        make_sample_data()
    )

    preprocessor = build_preprocessor(features)

    transformed = dense_values(
        preprocessor.fit_transform(features)
    )

    assert transformed.shape[0] == len(features)
    assert not np.isnan(transformed).any()


def test_unseen_categories_do_not_break_transformation() -> None:
    features, _ = prepare_features_and_target(
        make_sample_data()
    )

    preprocessor = build_preprocessor(features).fit(
        features
    )

    unseen = features.iloc[[0]].copy()

    unseen.loc[:, "merchant_category"] = (
        "Unknown Merchant Category"
    )

    unseen.loc[:, "device_type"] = "Smart Watch"

    transformed = preprocessor.transform(unseen)

    assert transformed.shape[0] == 1

    assert transformed.shape[1] == len(
        preprocessor.get_feature_names_out()
    )

def test_preprocessor_can_scale_numeric_features() -> None:
    """Verify optional numeric scaling works without missing values."""

    features, _ = prepare_features_and_target(
        make_sample_data()
    )

    preprocessor = build_preprocessor(
        features,
        scale_numeric=True,
    )

    transformed = dense_values(
        preprocessor.fit_transform(features)
    )

    assert transformed.shape[0] == len(features)
    assert not np.isnan(transformed).any()