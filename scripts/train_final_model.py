"""Train and save the selected fraud detection pipeline."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from app.ml.preprocessing import (
    build_preprocessor,
    prepare_features_and_target,
)

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "bank_fraud_poc_sample.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "fraud_detection_pipeline.joblib"
MLFLOW_TRACKING_URI = f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').resolve().as_posix()}"
MLFLOW_EXPERIMENT_NAME = "fraud_detection_final_model"

RANDOM_STATE = 42


def load_training_data(
    path: Path,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load the selected fraud dataset and prepare features and target."""

    data = pd.read_csv(path)

    return prepare_features_and_target(data)


def build_final_pipeline(
    features: pd.DataFrame,
) -> Pipeline:
    """Build the selected preprocessing and Random Forest pipeline."""

    preprocessor = build_preprocessor(
        features,
        scale_numeric=False,
    )

    classifier = RandomForestClassifier(
        n_estimators=300,
        max_depth=16,
        min_samples_split=4,
        min_samples_leaf=4,
        max_features="log2",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=2,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def save_model(
    model: Pipeline,
    path: Path,
) -> None:
    """Persist the trained pipeline to disk."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        path,
    )


def evaluate_model(
    model: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
) -> dict[str, float]:
    """Calculate fraud-detection metrics for MLflow tracking."""

    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]

    return {
        "precision": precision_score(target, predictions, zero_division=0),
        "recall": recall_score(target, predictions, zero_division=0),
        "f1": f1_score(target, predictions, zero_division=0),
        "roc_auc": roc_auc_score(target, probabilities),
    }


def log_final_model_to_mlflow(
    model: Pipeline,
    test_features: pd.DataFrame,
    test_target: pd.Series,
    model_path: Path,
) -> None:
    """Log the selected fitted model, metrics, and local artifact reference."""

    classifier = model.named_steps["classifier"]
    classifier_parameters = {
        f"classifier_{name}": value
        for name, value in classifier.get_params().items()
    }
    metrics = evaluate_model(model, test_features, test_target)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name="final_random_forest"):
        mlflow.log_params(classifier_parameters)
        mlflow.log_metrics(metrics)
        mlflow.set_tag("model_type", "RandomForestClassifier")
        mlflow.set_tag("local_model_path", str(model_path.relative_to(PROJECT_ROOT)))
        mlflow.sklearn.log_model(
            model,
            name="fraud_detection_pipeline",
            skops_trusted_types=[
                "numpy.dtype",
                "sklearn.tree._tree.Tree",
            ],
        )
        mlflow.log_artifact(str(model_path), artifact_path="model_reference")


def main() -> None:
    """Train the selected model and save the complete pipeline."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Training data not found: {DATA_PATH}"
        )

    X, y = load_training_data(
        DATA_PATH
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = build_final_pipeline(
        X_train
    )

    print("Training final fraud detection pipeline...")

    model.fit(
        X_train,
        y_train,
    )

    save_model(
        model,
        MODEL_PATH,
    )

    log_final_model_to_mlflow(
        model,
        X_test,
        y_test,
        MODEL_PATH,
    )

    print(
        f"Saved model to: {MODEL_PATH}"
    )


if __name__ == "__main__":
    main()
