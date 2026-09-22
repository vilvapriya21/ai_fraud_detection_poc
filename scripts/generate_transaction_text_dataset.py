"""Generate a deterministic text dataset from bank transaction attributes."""

from __future__ import annotations

from pathlib import Path
import re

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = PROJECT_ROOT / "data" / "processed" / "bank_fraud_poc_sample.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "transaction_text_dataset.csv"
SAMPLE_SIZE = 30_000
RANDOM_STATE = 42
TARGET_COLUMN = "is_fraud"
OUTPUT_COLUMNS = ("transaction_id", "transaction_text", TARGET_COLUMN)
LABEL_LEAKAGE_WORDS = (
    "fraud",
    "fraudulent",
    "legitimate",
    "suspicious",
    "safe",
    "risky",
)


def format_value(value: object) -> str:
    """Render a numeric source value without adding a unit or interpretation."""

    return f"{float(value):g}"


def generate_transaction_text(transaction: pd.Series) -> str:
    """Create an objective description from non-target transaction attributes."""

    weekend_label = "weekend" if int(transaction["is_weekend"]) else "weekday"
    time_label = "night-time" if int(transaction["is_night_transaction"]) else "daytime"
    location_label = "international" if int(transaction["is_international"]) else "domestic"
    pin_label = "a recent PIN change" if int(transaction["pin_changed_recently"]) else "no recent PIN change"
    failed_attempts = int(transaction["failed_attempts"])
    attempt_label = "attempt" if failed_attempts == 1 else "attempts"

    return (
        f"At {int(transaction['hour_of_day']):02d}:00 on a {weekend_label}, "
        f"a {time_label} {location_label} transaction was recorded in "
        f"{transaction['city']}, {transaction['country']}. The "
        f"{transaction['merchant_category']} merchant used "
        f"{transaction['payment_method']} through a {transaction['device_type']} "
        f"device. The customer is {int(transaction['customer_age'])} years old "
        f"with credit score {int(transaction['credit_score'])}, account age "
        f"{format_value(transaction['account_age_years'])} years, and balance "
        f"{format_value(transaction['account_balance'])}. Transaction amount was "
        f"{format_value(transaction['transaction_amount'])}; previous transactions "
        f"numbered {int(transaction['num_prev_transactions'])}, monthly frequency "
        f"was {int(transaction['transaction_freq_monthly'])}, home distance was "
        f"{format_value(transaction['distance_from_home_km'])} km, and time since "
        f"the prior transaction was {format_value(transaction['time_since_last_txn_hrs'])} "
        f"hours. There was {pin_label} and {failed_attempts} failed {attempt_label}."
    )


def stratified_sample(
    data: pd.DataFrame,
    sample_size: int = SAMPLE_SIZE,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Return a deterministic stratified transaction sample."""

    if sample_size > len(data):
        raise ValueError("Sample size cannot exceed the source dataset size.")

    sample, _ = train_test_split(
        data,
        train_size=sample_size,
        stratify=data[TARGET_COLUMN],
        random_state=random_state,
    )
    return sample.sort_values("transaction_id").reset_index(drop=True)


def contains_label_leakage(text: str) -> bool:
    """Return whether text contains prohibited target-revealing words."""

    pattern = r"\b(?:" + "|".join(LABEL_LEAKAGE_WORDS) + r")\b"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def validate_text_dataset(data: pd.DataFrame) -> None:
    """Validate output structure, labels, identifiers, and text safety."""

    if tuple(data.columns) != OUTPUT_COLUMNS:
        raise ValueError(f"Output columns must be exactly: {OUTPUT_COLUMNS}")
    if data["transaction_text"].isna().any() or not data["transaction_text"].str.strip().all():
        raise ValueError("Generated transaction text must not be empty.")
    if not set(data[TARGET_COLUMN].unique()).issubset({0, 1}):
        raise ValueError("is_fraud must contain only 0 and 1.")
    if not data["transaction_id"].is_unique:
        raise ValueError("transaction_id must be unique.")
    if data["transaction_text"].map(contains_label_leakage).any():
        raise ValueError("Generated text contains a target-leakage word.")


def create_text_dataset(
    source_path: Path = SOURCE_PATH,
    output_path: Path = OUTPUT_PATH,
) -> pd.DataFrame:
    """Generate, validate, save, and return the text classification dataset."""

    source_data = pd.read_csv(source_path)
    sampled_data = stratified_sample(source_data)
    output_data = pd.DataFrame(
        {
            "transaction_id": sampled_data["transaction_id"],
            "transaction_text": sampled_data.apply(generate_transaction_text, axis=1),
            TARGET_COLUMN: sampled_data[TARGET_COLUMN],
        }
    )
    validate_text_dataset(output_data)
    output_data.to_csv(output_path, index=False)
    return output_data


def main() -> None:
    """Create the text dataset and print required validation summaries."""

    text_dataset = create_text_dataset()
    print(f"Dataset shape: {text_dataset.shape}")
    print("Class distribution:")
    print(text_dataset[TARGET_COLUMN].value_counts(normalize=True).sort_index().mul(100))
    print("Sample descriptions:")
    for description in text_dataset["transaction_text"].head(3):
        print(f"- {description}")


if __name__ == "__main__":
    main()
