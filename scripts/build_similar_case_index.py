"""Create deterministic historical fraud cases and a FAISS similarity index."""

from __future__ import annotations

from pathlib import Path
import sys

import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.similar_case_service import EMBEDDING_MODEL_NAME


SOURCE_PATH = PROJECT_ROOT / "data" / "processed" / "bank_fraud_poc_sample.csv"
CASES_PATH = PROJECT_ROOT / "data" / "cases" / "historical_fraud_cases.csv"
INDEX_PATH = PROJECT_ROOT / "data" / "cases" / "historical_fraud_cases.faiss"
CASES_PER_FRAUD_TYPE = 60
RANDOM_STATE = 42


def format_number(value: object) -> str:
    """Format numeric source values compactly for case text."""

    return f"{float(value):g}"


def build_description(record: pd.Series) -> str:
    """Create an objective case description from recorded transaction fields."""

    transaction_scope = "international" if int(record["is_international"]) else "domestic"
    timing = "night-time" if int(record["is_night_transaction"]) else "daytime"
    return (
        f"A {timing} {transaction_scope} transaction in {record['city']}, "
        f"{record['country']} involved a {record['merchant_category']} merchant. "
        f"Payment used {record['payment_method']} on a {record['device_type']} device. "
        f"The amount was {format_number(record['transaction_amount'])}; distance from "
        f"home was {format_number(record['distance_from_home_km'])} km; time since the "
        f"previous transaction was {format_number(record['time_since_last_txn_hrs'])} hours."
    )


def build_observations(record: pd.Series) -> str:
    """Summarize recorded contextual fields without inventing new evidence."""

    observations = [
        "international transaction" if int(record["is_international"]) else "domestic transaction",
        "night-time transaction" if int(record["is_night_transaction"]) else "daytime transaction",
        f"{int(record['failed_attempts'])} failed attempts",
        "recent PIN change" if int(record["pin_changed_recently"]) else "no recent PIN change",
        f"monthly transaction frequency {int(record['transaction_freq_monthly'])}",
    ]
    return "; ".join(observations)


def create_case_dataset(source_data: pd.DataFrame) -> pd.DataFrame:
    """Return a reproducible balanced sample of labeled synthetic fraud cases."""

    fraud_records = source_data.loc[source_data["is_fraud"].eq(1)].dropna(subset=["fraud_type"])
    sampled = (
        fraud_records.groupby("fraud_type", group_keys=False)
        .sample(n=CASES_PER_FRAUD_TYPE, random_state=RANDOM_STATE)
        .sort_values("transaction_id")
        .reset_index(drop=True)
    )
    cases = pd.DataFrame(
        {
            "case_id": [f"CASE-{transaction_id}" for transaction_id in sampled["transaction_id"]],
            "description": sampled.apply(build_description, axis=1),
            "fraud_type": sampled["fraud_type"].astype(str),
            "key_observations": sampled.apply(build_observations, axis=1),
            "outcome": sampled["fraud_type"].map(lambda value: f"Confirmed {value} case."),
        }
    )
    if not cases["case_id"].is_unique or cases.isna().any().any():
        raise ValueError("Generated historical cases must be complete and uniquely identified.")
    return cases


def build_and_save_index(cases: pd.DataFrame) -> None:
    """Embed case descriptions and persist their normalized FAISS inner-product index."""

    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")
    embeddings = embedding_model.encode(
        cases["description"].tolist(),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(INDEX_PATH))


def main() -> None:
    """Build persisted synthetic case records and their FAISS index."""

    source_data = pd.read_csv(SOURCE_PATH)
    cases = create_case_dataset(source_data)
    CASES_PATH.parent.mkdir(parents=True, exist_ok=True)
    cases.to_csv(CASES_PATH, index=False)
    build_and_save_index(cases)
    print(f"Saved {len(cases)} historical fraud cases to: {CASES_PATH}")
    print(f"Saved FAISS index to: {INDEX_PATH}")


if __name__ == "__main__":
    main()
