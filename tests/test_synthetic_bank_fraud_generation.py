"""Tests for the deterministic CI bank-fraud dataset generator."""

from app.services.prediction_service import PREDICTION_FEATURE_COLUMNS
from scripts.generate_synthetic_bank_fraud import FRAUD_TYPES, ROW_COUNT, generate_dataset


def test_synthetic_dataset_matches_pipeline_schema_and_is_deterministic() -> None:
    """The CI source data has all required columns and repeatable labels."""

    first_dataset = generate_dataset()
    second_dataset = generate_dataset()

    assert list(first_dataset.columns) == [
        *PREDICTION_FEATURE_COLUMNS,
        "is_fraud",
        "fraud_type",
        "transaction_id",
        "customer_id",
        "transaction_date",
        "transaction_time",
    ]
    assert len(first_dataset) == ROW_COUNT
    assert first_dataset.equals(second_dataset)
    assert first_dataset["transaction_id"].is_unique
    assert first_dataset["is_fraud"].sum() == ROW_COUNT // 4
    assert (
        first_dataset.loc[first_dataset["is_fraud"].eq(1), "fraud_type"]
        .value_counts()
        .reindex(FRAUD_TYPES, fill_value=0)
        .ge(60)
        .all()
    )
