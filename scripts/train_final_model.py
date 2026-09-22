"""Train and save the selected fraud detection pipeline."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from app.ml.preprocessing import (
    build_preprocessor,
    prepare_features_and_target,
)

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "bank_fraud_poc_sample.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "fraud_detection_pipeline.joblib"

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


def main() -> None:
    """Train the selected model and save the complete pipeline."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Training data not found: {DATA_PATH}"
        )

    X, y = load_training_data(
        DATA_PATH
    )

    X_train, _, y_train, _ = train_test_split(
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

    print(
        f"Saved model to: {MODEL_PATH}"
    )


if __name__ == "__main__":
    main()
