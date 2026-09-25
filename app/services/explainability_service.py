"""SHAP model evidence and basic held-out fairness analysis for the saved fraud model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from app.schemas.prediction import PredictionRequest
from app.services.prediction_service import (
    PREDICTION_FEATURE_COLUMNS,
    ModelUnavailableError,
    PredictionService,
    prediction_service,
    project_root,
)


TARGET_COLUMN = "is_fraud"
FAIRNESS_GROUP_COLUMNS = ("country", "device_type", "merchant_category", "age_band")
TOP_FEATURE_COUNT = 5


class ExplainabilityUnavailableError(RuntimeError):
    """Raised when the saved pipeline or processed evaluation data is unavailable."""


class ExplainabilityService:
    """Provide cached Tree SHAP explanations and reusable fairness calculations."""

    def __init__(
        self,
        model_path: Path | None = None,
        dataset_path: Path | None = None,
        prediction_tool: PredictionService = prediction_service,
    ) -> None:
        """Configure saved artifacts without loading them until requested."""

        root = project_root()
        self.model_path = model_path or root / "models" / "fraud_detection_pipeline.joblib"
        self.dataset_path = dataset_path or root / "data" / "processed" / "bank_fraud_poc_sample.csv"
        self.prediction_tool = prediction_tool
        self._model: Pipeline | None = None
        self._explainer: shap.TreeExplainer | None = None
        self._fairness_summary: dict[str, Any] | None = None

    def explain(self, transaction: PredictionRequest) -> dict[str, Any]:
        """Return a saved-model prediction and the five largest local SHAP contributions."""

        try:
            prediction = self.prediction_tool.predict(transaction)
        except ModelUnavailableError as error:
            raise ExplainabilityUnavailableError(str(error)) from error

        try:
            model = self._load_model()
            feature_frame = pd.DataFrame(
                [transaction.model_dump()],
                columns=PREDICTION_FEATURE_COLUMNS,
            )
            contributions = self._feature_contributions(model, feature_frame)
        except ExplainabilityUnavailableError:
            raise
        except Exception as error:
            raise ExplainabilityUnavailableError(
                "SHAP explanation is temporarily unavailable."
            ) from error
        return {
            "prediction": prediction.prediction,
            "fraud_probability": prediction.fraud_probability,
            "top_contributing_features": contributions[:TOP_FEATURE_COUNT],
        }

    def fairness_summary(self) -> dict[str, Any]:
        """Return cached group metrics computed from the deterministic held-out split."""

        if self._fairness_summary is None:
            try:
                self._fairness_summary = self._compute_fairness_summary()
            except ExplainabilityUnavailableError:
                raise
            except Exception as error:
                raise ExplainabilityUnavailableError(
                    "Fairness analysis is temporarily unavailable."
                ) from error
        return self._fairness_summary

    def _load_model(self) -> Pipeline:
        """Load and cache the exact persisted model pipeline used for explanations."""

        if self._model is not None:
            return self._model
        if not self.model_path.is_file():
            raise ExplainabilityUnavailableError(f"Model artifact not found: {self.model_path}")
        try:
            model = joblib.load(self.model_path)
            if not isinstance(model, Pipeline):
                raise TypeError("Saved model must be a scikit-learn Pipeline.")
            if {"preprocessor", "classifier"} - set(model.named_steps):
                raise ValueError("Saved model lacks the required preprocessing or classifier step.")
        except (OSError, TypeError, ValueError) as error:
            raise ExplainabilityUnavailableError("Saved model could not be loaded for explanation.") from error
        self._model = model
        return model

    def _get_explainer(self, model: Pipeline) -> shap.TreeExplainer:
        """Create and cache a SHAP TreeExplainer for the saved tree classifier."""

        if self._explainer is None:
            self._explainer = shap.TreeExplainer(model.named_steps["classifier"])
        return self._explainer

    def _feature_contributions(
        self,
        model: Pipeline,
        feature_frame: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        """Aggregate transformed SHAP values into original model-input features."""

        preprocessor = model.named_steps["preprocessor"]
        transformed = preprocessor.transform(feature_frame)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        explainer = self._get_explainer(model)
        shap_values = explainer.shap_values(transformed)
        fraud_values = self._fraud_class_values(
            np.asarray(shap_values),
            list(model.named_steps["classifier"].classes_),
        )
        transformed_names = preprocessor.get_feature_names_out().tolist()
        aggregate_values = self._aggregate_original_features(fraud_values, transformed_names)

        ranked = sorted(aggregate_values.items(), key=lambda item: abs(item[1]), reverse=True)
        return [
            {
                "feature": feature,
                "feature_value": str(feature_frame.iloc[0][feature]),
                "shap_value": float(value),
                "direction": "higher" if value >= 0 else "lower",
            }
            for feature, value in ranked
        ]

    @staticmethod
    def _fraud_class_values(shap_values: np.ndarray, classes: list[Any]) -> np.ndarray:
        """Extract class-1 SHAP values across supported SHAP binary output layouts."""

        if 1 not in classes:
            raise ExplainabilityUnavailableError("Saved classifier does not expose fraud class 1.")
        fraud_index = classes.index(1)
        if shap_values.ndim == 3:
            return shap_values[0, :, fraud_index]
        if shap_values.ndim == 2:
            return shap_values[0]
        raise ExplainabilityUnavailableError("Unexpected SHAP output shape for the saved model.")

    @staticmethod
    def _aggregate_original_features(
        shap_values: np.ndarray,
        transformed_names: list[str],
    ) -> dict[str, float]:
        """Sum one-hot encoded SHAP contributions into their original feature groups."""

        grouped: dict[str, float] = {feature: 0.0 for feature in PREDICTION_FEATURE_COLUMNS}
        categorical_features = {"country", "city", "merchant_category", "payment_method", "device_type"}
        for name, value in zip(transformed_names, shap_values, strict=True):
            original_name = name
            for categorical_feature in categorical_features:
                if name.startswith(f"{categorical_feature}_"):
                    original_name = categorical_feature
                    break
            grouped[original_name] += float(value)
        return grouped

    def _compute_fairness_summary(self) -> dict[str, Any]:
        """Compute basic rates on the deterministic 20% hold-out split used by training."""

        if not self.dataset_path.is_file():
            raise ExplainabilityUnavailableError(f"Processed dataset not found: {self.dataset_path}")
        data = pd.read_csv(self.dataset_path)
        required_columns = set(PREDICTION_FEATURE_COLUMNS) | {TARGET_COLUMN}
        if not required_columns.issubset(data.columns):
            raise ExplainabilityUnavailableError("Processed dataset lacks required model or target columns.")

        _, evaluation_data = train_test_split(
            data,
            test_size=0.2,
            random_state=42,
            stratify=data[TARGET_COLUMN],
        )
        model = self._load_model()
        features = evaluation_data.loc[:, PREDICTION_FEATURE_COLUMNS]
        predictions = model.predict(features).astype(int)
        labels = evaluation_data[TARGET_COLUMN].astype(int).to_numpy()
        fairness_data = evaluation_data.loc[:, ["country", "device_type", "merchant_category", "customer_age"]].copy()
        fairness_data["age_band"] = pd.cut(
            fairness_data["customer_age"],
            bins=[0, 24, 39, 54, np.inf],
            labels=["18-24", "25-39", "40-54", "55+"],
            include_lowest=True,
        ).astype(str)
        fairness_data["actual"] = labels
        fairness_data["predicted"] = predictions

        summaries = {
            group_column: self._group_metrics(fairness_data, group_column)
            for group_column in FAIRNESS_GROUP_COLUMNS
        }
        return {
            "evaluation_rows": int(len(fairness_data)),
            "groups_checked": list(FAIRNESS_GROUP_COLUMNS),
            "overall_predicted_fraud_rate": float(predictions.mean()),
            "summaries": summaries,
            "limitations": [
                "The data is synthetic and these results do not establish real-world fairness.",
                "Country, device type, merchant category, and age bands are operational groups, not protected-attribute analysis.",
                "Metrics are descriptive on one held-out split and do not establish causation or policy compliance.",
                "Small group differences should be reviewed with sample counts and domain context before action.",
            ],
        }

    @staticmethod
    def _group_metrics(data: pd.DataFrame, group_column: str) -> dict[str, Any]:
        """Calculate predicted rate, actual rate, TPR, and FPR for each observed group."""

        group_metrics: list[dict[str, Any]] = []
        for group, values in data.groupby(group_column, dropna=False, observed=True):
            actual = values["actual"].to_numpy()
            predicted = values["predicted"].to_numpy()
            positives = int(actual.sum())
            negatives = int(len(actual) - positives)
            true_positives = int(((actual == 1) & (predicted == 1)).sum())
            false_positives = int(((actual == 0) & (predicted == 1)).sum())
            group_metrics.append(
                {
                    "group": str(group),
                    "sample_count": int(len(values)),
                    "actual_fraud_rate": float(actual.mean()),
                    "predicted_fraud_rate": float(predicted.mean()),
                    "true_positive_rate": float(true_positives / positives) if positives else None,
                    "false_positive_rate": float(false_positives / negatives) if negatives else None,
                }
            )
        selection_rates = [metric["predicted_fraud_rate"] for metric in group_metrics]
        return {
            "selection_rate_gap": float(max(selection_rates) - min(selection_rates)),
            "groups": group_metrics,
        }


explainability_service = ExplainabilityService()
