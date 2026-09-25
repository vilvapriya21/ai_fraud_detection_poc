"""FAISS-backed semantic retrieval for saved historical fraud cases."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
REQUIRED_CASE_COLUMNS = {
    "case_id",
    "description",
    "fraud_type",
    "key_observations",
    "outcome",
}


class SimilarCaseUnavailableError(RuntimeError):
    """Raised when persisted case-search assets are unavailable or invalid."""


def project_root() -> Path:
    """Return the repository root from this module location."""

    return Path(__file__).resolve().parents[2]


class SimilarCaseService:
    """Load persisted case records and perform cosine-similarity retrieval."""

    def __init__(
        self,
        cases_path: Path | None = None,
        index_path: Path | None = None,
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
    ) -> None:
        """Configure locations for the saved cases and FAISS index."""

        cases_directory = project_root() / "data" / "cases"
        self.cases_path = cases_path or cases_directory / "historical_fraud_cases.csv"
        self.index_path = index_path or cases_directory / "historical_fraud_cases.faiss"
        self.embedding_model_name = embedding_model_name
        self._cases: list[dict[str, str]] | None = None
        self._index: faiss.Index | None = None
        self._embedding_model: SentenceTransformer | None = None

    def search(self, description: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Return the most semantically similar saved fraud cases."""

        if not description.strip():
            raise ValueError("description must not be empty.")

        try:
            cases, index = self._load_assets()
            query_embedding = self._get_embedding_model().encode(
                [description],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            result_count = min(top_k, len(cases))
            scores, positions = index.search(query_embedding.astype("float32"), result_count)
        except SimilarCaseUnavailableError:
            raise
        except Exception as error:
            raise SimilarCaseUnavailableError(
                "Similar-case search is temporarily unavailable."
            ) from error

        try:
            results: list[dict[str, Any]] = []
            for score, position in zip(scores[0], positions[0], strict=True):
                if position < 0:
                    continue
                case = cases[int(position)]
                results.append(
                    {
                        "case_id": case["case_id"],
                        "fraud_type": case["fraud_type"],
                        "key_observations": case["key_observations"],
                        "outcome": case["outcome"],
                        "similarity_score": float(score),
                    }
                )
        except Exception as error:
            raise SimilarCaseUnavailableError(
                "Similar-case search is temporarily unavailable."
            ) from error
        return results

    def _load_assets(self) -> tuple[list[dict[str, str]], faiss.Index]:
        """Load and cache the persisted FAISS index and its ordered records."""

        if self._cases is not None and self._index is not None:
            return self._cases, self._index
        if not self.cases_path.is_file() or not self.index_path.is_file():
            raise SimilarCaseUnavailableError(
                "Saved similar-case assets are unavailable. Run "
                "scripts/build_similar_case_index.py first."
            )

        try:
            case_frame = pd.read_csv(self.cases_path, dtype=str).fillna("")
            if not REQUIRED_CASE_COLUMNS.issubset(case_frame.columns):
                raise ValueError("Case dataset is missing required columns.")
            index = faiss.read_index(str(self.index_path))
        except (OSError, ValueError, RuntimeError) as error:
            raise SimilarCaseUnavailableError("Saved similar-case assets could not be loaded.") from error

        if case_frame.empty or index.ntotal != len(case_frame):
            raise SimilarCaseUnavailableError("Case records and FAISS index are inconsistent.")

        self._cases = case_frame.to_dict(orient="records")
        self._index = index
        return self._cases, self._index

    def _get_embedding_model(self) -> SentenceTransformer:
        """Load and cache the embedding model only when a search is requested."""

        if self._embedding_model is None:
            try:
                self._embedding_model = SentenceTransformer(
                    self.embedding_model_name,
                    device="cpu",
                    local_files_only=True,
                )
            except Exception as error:
                raise SimilarCaseUnavailableError(
                    "The saved embedding model is unavailable locally. Run "
                    "scripts/build_similar_case_index.py first."
                ) from error
        return self._embedding_model


similar_case_service = SimilarCaseService()
