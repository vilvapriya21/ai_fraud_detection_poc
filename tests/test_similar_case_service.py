"""Tests for persisted semantic historical-case retrieval."""

from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import pytest

from app.services.similar_case_service import (
    REQUIRED_CASE_COLUMNS,
    SimilarCaseService,
    SimilarCaseUnavailableError,
)


class FakeEmbeddingModel:
    """Return a fixed normalized query embedding without loading MiniLM."""

    def encode(self, descriptions: list[str], **_: object) -> np.ndarray:
        """Return the query vector used by the temporary FAISS index."""

        return np.array([[1.0, 0.0]], dtype="float32")


def case_records() -> list[dict[str, str]]:
    """Return ordered case records paired with the temporary FAISS vectors."""

    return [
        {
            "case_id": "CASE-1",
            "description": "First case",
            "fraud_type": "Account Takeover",
            "key_observations": "international transaction",
            "outcome": "Confirmed account takeover case.",
        },
        {
            "case_id": "CASE-2",
            "description": "Second case",
            "fraud_type": "Card Not Present",
            "key_observations": "night-time transaction",
            "outcome": "Confirmed card-not-present case.",
        },
        {
            "case_id": "CASE-3",
            "description": "Third case",
            "fraud_type": "Identity Theft",
            "key_observations": "recent PIN change",
            "outcome": "Confirmed identity theft case.",
        },
    ]


def write_case_assets(
    directory: Path,
    records: list[dict[str, str]] | None = None,
    vectors: np.ndarray | None = None,
) -> tuple[Path, Path]:
    """Write a small ordered case CSV and compatible FAISS index."""

    cases_path = directory / "cases.csv"
    index_path = directory / "cases.faiss"
    pd.DataFrame(records or case_records()).to_csv(cases_path, index=False)
    index_vectors = vectors if vectors is not None else np.array(
        [[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]], dtype="float32"
    )
    index = faiss.IndexFlatIP(index_vectors.shape[1])
    index.add(index_vectors)
    faiss.write_index(index, str(index_path))
    return cases_path, index_path


@pytest.fixture
def similar_case_service(tmp_path: Path) -> SimilarCaseService:
    """Return a service using temporary assets and deterministic embeddings."""

    cases_path, index_path = write_case_assets(tmp_path)
    service = SimilarCaseService(cases_path=cases_path, index_path=index_path)
    service._embedding_model = FakeEmbeddingModel()
    return service


@pytest.mark.parametrize("missing_asset", ("cases.csv", "cases.faiss"))
def test_load_assets_rejects_missing_csv_or_faiss_index(
    tmp_path: Path,
    missing_asset: str,
) -> None:
    """Both persisted asset files are required before a search can run."""

    cases_path, index_path = write_case_assets(tmp_path)
    (tmp_path / missing_asset).unlink()
    service = SimilarCaseService(cases_path=cases_path, index_path=index_path)

    with pytest.raises(SimilarCaseUnavailableError, match="assets are unavailable"):
        service._load_assets()


def test_load_assets_rejects_csv_without_required_columns(tmp_path: Path) -> None:
    """Case records must expose every field used in API search results."""

    incomplete_records = [{"case_id": "CASE-1", "description": "Incomplete case"}]
    cases_path, index_path = write_case_assets(
        tmp_path,
        records=incomplete_records,
        vectors=np.array([[1.0, 0.0]], dtype="float32"),
    )
    service = SimilarCaseService(cases_path=cases_path, index_path=index_path)

    with pytest.raises(SimilarCaseUnavailableError, match="could not be loaded"):
        service._load_assets()


def test_load_assets_rejects_mismatched_case_and_index_counts(tmp_path: Path) -> None:
    """The ordered CSV records and FAISS positions must have equal counts."""

    cases_path, index_path = write_case_assets(
        tmp_path,
        vectors=np.array([[1.0, 0.0], [0.8, 0.6]], dtype="float32"),
    )
    service = SimilarCaseService(cases_path=cases_path, index_path=index_path)

    with pytest.raises(SimilarCaseUnavailableError, match="inconsistent"):
        service._load_assets()


def test_search_rejects_empty_description() -> None:
    """Descriptions containing no query text are rejected before asset loading."""

    service = SimilarCaseService()

    with pytest.raises(ValueError, match="description must not be empty"):
        service.search("   ")


def test_search_orders_results_and_returns_all_public_fields(
    similar_case_service: SimilarCaseService,
) -> None:
    """FAISS scores determine result order and the public result schema is complete."""

    results = similar_case_service.search("query", top_k=3)

    assert [result["case_id"] for result in results] == ["CASE-1", "CASE-2", "CASE-3"]
    assert [result["similarity_score"] for result in results] == sorted(
        (result["similarity_score"] for result in results), reverse=True
    )
    assert set(results[0]) == {
        "case_id",
        "fraud_type",
        "key_observations",
        "outcome",
        "similarity_score",
    }
    assert REQUIRED_CASE_COLUMNS.issuperset(set(results[0]) - {"similarity_score"})


def test_search_respects_top_k_and_available_case_count(
    similar_case_service: SimilarCaseService,
) -> None:
    """Search returns no more than either top_k or the persisted case count."""

    assert len(similar_case_service.search("query", top_k=2)) == 2
    assert len(similar_case_service.search("query", top_k=10)) == 3
