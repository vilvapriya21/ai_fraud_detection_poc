"""Model loading and fraud prediction business logic."""

from pathlib import Path
from typing import Any, Literal

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from app.schemas.prediction import PredictionRequest, PredictionResponse

PREDICTION_FEATURE_COLUMNS = (
    "hour_of_day",
    "is_weekend",
    "is_night_transaction",
    "country",
    "city",
    "merchant_category",
    "payment_method",
    "device_type",
    "customer_age",
    "credit_score",
    "account_age_years",
    "account_balance",
    "transaction_amount",
    "num_prev_transactions",
    "transaction_freq_monthly",
    "distance_from_home_km",
    "time_since_last_txn_hrs",
    "is_international",
    "failed_attempts",
    "pin_changed_recently",
)


class ModelUnavailableError(RuntimeError):
    """Raised when the saved prediction pipeline is unavailable."""


def project_root() -> Path:
    """Return the repository root from this service module's location."""

    return Path(__file__).resolve().parents[2]


def risk_level_for_probability(
    fraud_probability: float,
) -> Literal["Low", "Medium", "High"]:
    """Map a fraud probability to the API's deterministic risk level."""

    if fraud_probability < 0.30:
        return "Low"
    if fraud_probability < 0.60:
        return "Medium"
    return "High"


class PredictionService:
    """Load the saved pipeline once and serve one-row fraud predictions."""

    def __init__(self, model_path: Path | None = None) -> None:
        """Initialize the service and attempt one model load."""

        self.model_path = model_path or (
            project_root() / "models" / "fraud_detection_pipeline.joblib"
        )
        self._model: Pipeline | None = None
        self._load_error: str | None = None
        self._load_attempted = False
        self.load_model()

    @property
    def model_loaded(self) -> bool:
        """Whether the saved pipeline loaded successfully."""

        return self._model is not None

    @property
    def load_error(self) -> str | None:
        """Return the model loading error, if one occurred."""

        return self._load_error

    def load_model(self) -> None:
        """Load the configured joblib pipeline once without fallback predictions."""

        if self._load_attempted:
            return

        self._load_attempted = True
        if not self.model_path.exists():
            self._load_error = f"Model artifact not found: {self.model_path}"
            return

        try:
            model = joblib.load(self.model_path)
            self._validate_model_features(model)
            self._model = model
        except Exception as error:
            self._load_error = (
                f"Unable to load model artifact at {self.model_path}: {error}"
            )

    def predict(self, transaction: PredictionRequest) -> PredictionResponse:
        """Return the fraud label, class-1 probability, and risk level."""

        if self._model is None:
            reason = self._load_error or "Model loading did not complete."
            raise ModelUnavailableError(reason)

        try:
            feature_frame = self._to_feature_frame(transaction)
            prediction_value = int(self._model.predict(feature_frame)[0])
            fraud_probability = self._fraud_probability(feature_frame)
        except ModelUnavailableError:
            raise
        except Exception as error:
            raise ModelUnavailableError(
                "Fraud prediction is temporarily unavailable."
            ) from error

        return PredictionResponse(
            prediction="Fraudulent" if prediction_value == 1 else "Legitimate",
            fraud_probability=fraud_probability,
            risk_level=risk_level_for_probability(fraud_probability),
        )

    def _to_feature_frame(
        self,
        transaction: PredictionRequest,
    ) -> pd.DataFrame:
        """Convert a validated API request into the pipeline feature order."""

        return pd.DataFrame(
            [transaction.model_dump()],
            columns=PREDICTION_FEATURE_COLUMNS,
        )

    def _fraud_probability(self, feature_frame: pd.DataFrame) -> float:
        """Extract the probability associated with fraud class ``1``."""

        if self._model is None:
            raise ModelUnavailableError("Model loading did not complete.")

        classes = list(self._model.classes_)
        if 1 not in classes:
            raise ModelUnavailableError("Model does not expose fraud class 1.")

        fraud_index = classes.index(1)
        return float(self._model.predict_proba(feature_frame)[0][fraud_index])

    @staticmethod
    def _validate_model_features(model: Any) -> None:
        """Confirm the persisted pipeline accepts the API schema's feature order."""

        feature_names = getattr(model, "feature_names_in_", None)
        if feature_names is None:
            raise ValueError("Saved model does not expose input feature names.")

        if tuple(feature_names.tolist()) != PREDICTION_FEATURE_COLUMNS:
            raise ValueError("Saved model feature schema does not match the API schema.")


prediction_service = PredictionService()
