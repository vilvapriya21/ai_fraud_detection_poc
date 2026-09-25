"""Lightweight Airflow validation flow for saved fraud-application assets."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

import joblib
import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.prediction_service import PREDICTION_FEATURE_COLUMNS


DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "bank_fraud_poc_sample.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "fraud_detection_pipeline.joblib"
RETRIEVAL_ARTIFACT_PATHS = (
    PROJECT_ROOT / "data" / "cases" / "historical_fraud_cases.csv",
    PROJECT_ROOT / "data" / "cases" / "historical_fraud_cases.faiss",
    PROJECT_ROOT / "data" / "knowledge_base" / "fraud_investigation_documents.json",
    PROJECT_ROOT / "data" / "knowledge_base" / "fraud_investigation.faiss",
)
TARGET_COLUMN = "is_fraud"
EVALUATION_SAMPLE_SIZE = 1_000


def validate_processed_dataset() -> None:
    """Confirm the processed dataset exists and has model features plus target."""

    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"Processed dataset not found: {DATASET_PATH}")

    columns = set(pd.read_csv(DATASET_PATH, nrows=1).columns)
    required_columns = set(PREDICTION_FEATURE_COLUMNS) | {TARGET_COLUMN}
    missing_columns = required_columns - columns
    if missing_columns:
        raise ValueError(f"Processed dataset is missing columns: {sorted(missing_columns)}")


def check_final_model_artifact() -> None:
    """Confirm the final saved prediction pipeline is available."""

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Final model artifact not found: {MODEL_PATH}")


def check_retrieval_artifacts() -> None:
    """Confirm persisted FAISS indexes and their source records are available."""

    missing_artifacts = [str(path) for path in RETRIEVAL_ARTIFACT_PATHS if not path.is_file()]
    if missing_artifacts:
        raise FileNotFoundError(f"Retrieval artifacts not found: {missing_artifacts}")


def evaluate_saved_model() -> None:
    """Evaluate the saved model on a deterministic small processed-data sample."""

    data = pd.read_csv(DATASET_PATH)
    sample = data.sample(
        n=min(EVALUATION_SAMPLE_SIZE, len(data)),
        random_state=42,
    )
    labels = sample[TARGET_COLUMN].astype(int)
    if labels.nunique() != 2:
        raise ValueError("Evaluation sample must contain both fraud target classes.")

    model = joblib.load(MODEL_PATH)
    features = sample.loc[:, PREDICTION_FEATURE_COLUMNS]
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]
    metrics = {
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "roc_auc": roc_auc_score(labels, probabilities),
    }
    print({name: round(value, 4) for name, value in metrics.items()})


with DAG(
    dag_id="fraud_mlops_dag",
    description="Validate saved fraud assets and run lightweight evaluation.",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["fraud", "mlops", "poc"],
) as dag:
    validate_dataset_task = PythonOperator(
        task_id="validate_processed_dataset",
        python_callable=validate_processed_dataset,
    )
    check_model_task = PythonOperator(
        task_id="check_final_model_artifact",
        python_callable=check_final_model_artifact,
    )
    check_retrieval_task = PythonOperator(
        task_id="check_retrieval_artifacts",
        python_callable=check_retrieval_artifacts,
    )
    evaluate_model_task = PythonOperator(
        task_id="evaluate_saved_model",
        python_callable=evaluate_saved_model,
    )

    validate_dataset_task >> check_model_task >> check_retrieval_task >> evaluate_model_task
