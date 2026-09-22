"""Read saved text-model comparison results for the API."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ModelComparisonUnavailableError(RuntimeError):
    """Raised when the saved comparison artifact cannot be read."""


def project_root() -> Path:
    """Return the repository root from this service module location."""

    return Path(__file__).resolve().parents[2]


class ModelComparisonService:
    """Load and cache the saved comparison artifact without model execution."""

    def __init__(self, results_path: Path | None = None) -> None:
        """Configure the saved JSON artifact location."""

        self.results_path = results_path or (
            project_root() / "data" / "processed" / "text_model_comparison_results.json"
        )
        self._results: dict[str, Any] | None = None

    def get_results(self) -> dict[str, Any]:
        """Return cached comparison results or read and validate them once."""

        if self._results is None:
            self._results = self._load_results()
        return self._results

    def _load_results(self) -> dict[str, Any]:
        """Read the JSON artifact without retraining or loading model weights."""

        if not self.results_path.exists():
            raise ModelComparisonUnavailableError(
                f"Comparison results artifact not found: {self.results_path}"
            )

        try:
            with self.results_path.open(encoding="utf-8") as results_file:
                results = json.load(results_file)
        except (OSError, json.JSONDecodeError) as error:
            raise ModelComparisonUnavailableError(
                f"Unable to read comparison results: {error}"
            ) from error

        models = results.get("models")
        if not isinstance(models, list) or not models:
            raise ModelComparisonUnavailableError(
                "Comparison results must contain a non-empty 'models' list."
            )

        return results


model_comparison_service = ModelComparisonService()
