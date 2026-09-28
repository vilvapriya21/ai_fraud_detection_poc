"""Generate deterministic synthetic bank-fraud data for CI artifact bootstrap."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.prediction_service import PREDICTION_FEATURE_COLUMNS


OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "bank_fraud.csv"
RANDOM_STATE = 42
ROW_COUNT = 2_000
FRAUD_TYPES = (
    "Account Takeover",
    "Card Not Present",
    "Identity Theft",
    "Authorized Push Payment",
)

CITIES_BY_COUNTRY = {
    "USA": ("New York", "Chicago", "Austin"),
    "Canada": ("Toronto", "Vancouver", "Montreal"),
    "United Kingdom": ("London", "Manchester", "Birmingham"),
    "India": ("Mumbai", "Delhi", "Bengaluru"),
}


def _fraud_labels(risk_score: np.ndarray) -> np.ndarray:
    """Assign a deterministic 25% fraud class to the highest-risk records."""

    fraud_count = ROW_COUNT // 4
    labels = np.zeros(ROW_COUNT, dtype=int)
    labels[np.argsort(risk_score)[-fraud_count:]] = 1
    return labels


def generate_dataset(row_count: int = ROW_COUNT) -> pd.DataFrame:
    """Return a reproducible data frame with all model and source columns."""

    if row_count != ROW_COUNT:
        raise ValueError(f"row_count must be {ROW_COUNT} for the CI dataset.")

    random = np.random.default_rng(RANDOM_STATE)
    hour_of_day = random.integers(0, 24, size=row_count)
    countries = random.choice(tuple(CITIES_BY_COUNTRY), size=row_count)
    cities = np.array(
        [random.choice(CITIES_BY_COUNTRY[country]) for country in countries],
        dtype=object,
    )
    is_international = random.binomial(1, 0.18, size=row_count)
    failed_attempts = random.poisson(0.45, size=row_count).clip(0, 5)
    pin_changed_recently = random.binomial(1, 0.12, size=row_count)
    transaction_amount = random.lognormal(mean=4.5, sigma=0.8, size=row_count).round(2)
    distance_from_home_km = random.gamma(shape=2.0, scale=12.0, size=row_count).round(2)
    time_since_last_txn_hrs = random.gamma(shape=1.8, scale=8.0, size=row_count).round(2)
    credit_score = random.integers(300, 851, size=row_count)
    merchant_category = random.choice(
        ("Grocery", "Electronics", "Travel", "Crypto Exchange", "Restaurant"),
        size=row_count,
    )
    payment_method = random.choice(
        ("Credit Card", "Debit Card", "Bank Transfer", "Digital Wallet"),
        size=row_count,
    )
    device_type = random.choice(("Mobile", "Desktop", "POS Terminal"), size=row_count)
    is_night_transaction = ((hour_of_day <= 5) | (hour_of_day >= 22)).astype(int)
    risk_score = (
        1.4 * is_international
        + 1.2 * is_night_transaction
        + 0.8 * (failed_attempts >= 2)
        + 0.9 * pin_changed_recently
        + 0.7 * (merchant_category == "Crypto Exchange")
        + 0.5 * (transaction_amount >= np.quantile(transaction_amount, 0.85))
        + 0.4 * (credit_score < 560)
        + random.normal(0.0, 0.35, size=row_count)
    )
    is_fraud = _fraud_labels(risk_score)
    fraud_type = np.full(row_count, "", dtype=object)
    fraud_indices = np.flatnonzero(is_fraud)
    fraud_type[fraud_indices] = [
        FRAUD_TYPES[index % len(FRAUD_TYPES)] for index in range(len(fraud_indices))
    ]
    transaction_dates = pd.Timestamp("2024-01-01") + pd.to_timedelta(
        random.integers(0, 366, size=row_count), unit="D"
    )
    transaction_minutes = random.integers(0, 60, size=row_count)
    transaction_seconds = random.integers(0, 60, size=row_count)

    data = pd.DataFrame(
        {
            "hour_of_day": hour_of_day,
            "is_weekend": random.binomial(1, 2 / 7, size=row_count),
            "is_night_transaction": is_night_transaction,
            "country": countries,
            "city": cities,
            "merchant_category": merchant_category,
            "payment_method": payment_method,
            "device_type": device_type,
            "customer_age": random.integers(18, 81, size=row_count),
            "credit_score": credit_score,
            "account_age_years": random.uniform(0.1, 20.0, size=row_count).round(2),
            "account_balance": random.lognormal(8.4, 1.0, size=row_count).round(2),
            "transaction_amount": transaction_amount,
            "num_prev_transactions": random.integers(0, 500, size=row_count),
            "transaction_freq_monthly": random.integers(0, 61, size=row_count),
            "distance_from_home_km": distance_from_home_km,
            "time_since_last_txn_hrs": time_since_last_txn_hrs,
            "is_international": is_international,
            "failed_attempts": failed_attempts,
            "pin_changed_recently": pin_changed_recently,
            "is_fraud": is_fraud,
            "fraud_type": fraud_type,
            "transaction_id": [f"TXN{index:06d}" for index in range(1, row_count + 1)],
            "customer_id": [f"CUST{index:05d}" for index in random.integers(1, 801, size=row_count)],
            "transaction_date": transaction_dates.strftime("%Y-%m-%d"),
            "transaction_time": [
                f"{hour:02d}:{minute:02d}:{second:02d}"
                for hour, minute, second in zip(
                    hour_of_day,
                    transaction_minutes,
                    transaction_seconds,
                    strict=True,
                )
            ],
        }
    )
    expected_columns = {
        *PREDICTION_FEATURE_COLUMNS,
        "is_fraud",
        "fraud_type",
        "transaction_id",
        "customer_id",
        "transaction_date",
        "transaction_time",
    }
    if set(data.columns) != expected_columns:
        raise ValueError("Synthetic dataset schema does not match the application schema.")
    return data


def main() -> None:
    """Write the deterministic CI dataset to the expected raw-data location."""

    dataset = generate_dataset()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(dataset)} synthetic transactions to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
