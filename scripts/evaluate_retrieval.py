"""Evaluate persisted similar-fraud-case retrieval quality without rebuilding assets."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.similar_case_service import SimilarCaseService


CASES_PATH = PROJECT_ROOT / "data" / "cases" / "historical_fraud_cases.csv"
INDEX_PATH = PROJECT_ROOT / "data" / "cases" / "historical_fraud_cases.faiss"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "retrieval_eval.json"
RANDOM_STATE = 42
EVALUATION_FRACTION = 0.20
RETRIEVAL_CUTOFFS = (1, 3, 5)


def load_cases(path: Path) -> pd.DataFrame:
    """Load persisted historical cases used to build the existing FAISS index."""

    if not path.is_file():
        raise FileNotFoundError(f"Historical case dataset not found: {path}")

    cases = pd.read_csv(path, dtype=str).fillna("")
    required_columns = {"case_id", "description", "fraud_type"}
    missing_columns = required_columns - set(cases.columns)
    if missing_columns:
        raise ValueError(f"Historical case dataset is missing columns: {sorted(missing_columns)}")
    if cases.empty or not cases["case_id"].is_unique:
        raise ValueError("Historical case dataset must contain unique, non-empty case records.")
    return cases


def select_evaluation_cases(cases: pd.DataFrame) -> pd.DataFrame:
    """Select a deterministic fraud-type-stratified subset as evaluation queries."""

    selected = (
        cases.groupby("fraud_type", group_keys=False)
        .sample(frac=EVALUATION_FRACTION, random_state=RANDOM_STATE)
        .sort_values("case_id")
        .reset_index(drop=True)
    )
    if selected.empty:
        raise ValueError("The evaluation subset is empty.")
    return selected


def rank_retrieved_cases(
    service: SimilarCaseService,
    query_case: pd.Series,
    case_count: int,
) -> list[dict[str, Any]]:
    """Retrieve all saved cases and remove the exact query case before scoring."""

    retrieved = service.search(query_case["description"], top_k=case_count)
    return [
        case
        for case in retrieved
        if case["case_id"] != query_case["case_id"]
    ]


def first_relevant_rank(
    retrieved_cases: list[dict[str, Any]],
    fraud_type: str,
) -> int | None:
    """Return the one-based rank of the first retrieved case with a matching type."""

    for rank, case in enumerate(retrieved_cases, start=1):
        if case["fraud_type"] == fraud_type:
            return rank
    return None


def calculate_metrics(relevant_ranks: list[int | None]) -> dict[str, float]:
    """Calculate Hit@K and mean reciprocal rank from first relevant result ranks."""

    evaluation_count = len(relevant_ranks)
    if evaluation_count == 0:
        raise ValueError("Cannot calculate retrieval metrics for zero evaluation cases.")

    metrics = {
        f"hit_at_{cutoff}": sum(
            rank is not None and rank <= cutoff
            for rank in relevant_ranks
        )
        / evaluation_count
        for cutoff in RETRIEVAL_CUTOFFS
    }
    metrics["mrr"] = sum(
        1 / rank if rank is not None else 0
        for rank in relevant_ranks
    ) / evaluation_count
    return metrics


def evaluate_retrieval() -> dict[str, Any]:
    """Evaluate the existing FAISS search service against held-out query cases."""

    cases = load_cases(CASES_PATH)
    evaluation_cases = select_evaluation_cases(cases)
    service = SimilarCaseService(cases_path=CASES_PATH, index_path=INDEX_PATH)

    relevant_ranks = [
        first_relevant_rank(
            rank_retrieved_cases(service, query_case, len(cases)),
            query_case["fraud_type"],
        )
        for _, query_case in evaluation_cases.iterrows()
    ]

    return {
        "evaluation_count": len(evaluation_cases),
        "case_count": len(cases),
        "random_state": RANDOM_STATE,
        "metrics": calculate_metrics(relevant_ranks),
        "methodology": {
            "query_selection": (
                "A deterministic 20% stratified sample by fraud_type from the "
                "persisted historical case records."
            ),
            "query_input": "The saved description for each selected case.",
            "retrieval": (
                "The existing SimilarCaseService with the persisted MiniLM FAISS "
                "index; the query case itself is removed before scoring."
            ),
            "relevance": (
                "A retrieved case is relevant when its fraud_type matches the "
                "query case fraud_type."
            ),
            "metrics": "Hit@1, Hit@3, Hit@5, and mean reciprocal rank (MRR).",
        },
    }


def save_results(results: dict[str, Any], output_path: Path) -> None:
    """Persist evaluation results as readable JSON under processed data."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")


def main() -> None:
    """Run the retrieval evaluation and save its reproducible results."""

    results = evaluate_retrieval()
    save_results(results, OUTPUT_PATH)
    print(json.dumps(results, indent=2))
    print(f"Saved retrieval evaluation to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
