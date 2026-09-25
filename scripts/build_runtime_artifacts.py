"""Create the local runtime artifacts needed by the fraud FastAPI application."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "bank_fraud.csv"
PROCESSED_SAMPLE_PATH = PROJECT_ROOT / "data" / "processed" / "bank_fraud_poc_sample.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "fraud_detection_pipeline.joblib"
SIMILAR_CASE_ARTIFACTS = (
    PROJECT_ROOT / "data" / "cases" / "historical_fraud_cases.csv",
    PROJECT_ROOT / "data" / "cases" / "historical_fraud_cases.faiss",
)
KNOWLEDGE_BASE_ARTIFACTS = (
    PROJECT_ROOT / "data" / "knowledge_base" / "fraud_investigation_documents.json",
    PROJECT_ROOT / "data" / "knowledge_base" / "fraud_investigation.faiss",
)


def run_script(script_name: str) -> None:
    """Run one existing project build script from the repository root."""

    script_path = PROJECT_ROOT / "scripts" / script_name
    print(f"Running {script_name}...")
    subprocess.run(
        [sys.executable, str(script_path)],
        cwd=PROJECT_ROOT,
        check=True,
    )


def artifacts_exist(artifacts: tuple[Path, ...]) -> bool:
    """Return whether every artifact in a required artifact set exists."""

    return all(path.is_file() for path in artifacts)


def ensure_processed_sample() -> None:
    """Create the reproducible processed sample when it is not already present."""

    if PROCESSED_SAMPLE_PATH.is_file():
        print(f"Processed sample already exists: {PROCESSED_SAMPLE_PATH}")
        return

    if not RAW_DATASET_PATH.is_file():
        raise FileNotFoundError(
            "Cannot create the processed sample because the required raw dataset "
            f"is missing: {RAW_DATASET_PATH}"
        )

    run_script("create_poc_sample.py")


def ensure_final_model() -> None:
    """Train the selected final model only when its saved artifact is missing."""

    if MODEL_PATH.is_file():
        print(f"Final model already exists: {MODEL_PATH}")
        return

    run_script("train_final_model.py")


def ensure_similar_case_artifacts() -> None:
    """Create historical case data and its FAISS index when either is missing."""

    if artifacts_exist(SIMILAR_CASE_ARTIFACTS):
        print("Similar-case data and FAISS index already exist.")
        return

    run_script("build_similar_case_index.py")


def ensure_knowledge_base_artifacts() -> None:
    """Create the investigation knowledge-base documents and FAISS index if needed."""

    if artifacts_exist(KNOWLEDGE_BASE_ARTIFACTS):
        print("Investigation knowledge-base documents and FAISS index already exist.")
        return

    run_script("build_investigation_knowledge_base.py")


def main() -> None:
    """Prepare the required saved assets in application dependency order."""

    ensure_processed_sample()
    ensure_final_model()
    ensure_similar_case_artifacts()
    ensure_knowledge_base_artifacts()
    print("Runtime artifacts are ready.")


if __name__ == "__main__":
    main()
