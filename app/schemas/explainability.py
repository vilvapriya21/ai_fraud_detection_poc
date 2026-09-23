"""Schemas for model-evidence explanations and fairness summaries."""

from typing import Literal

from pydantic import BaseModel, Field


class FeatureContribution(BaseModel):
    """A local SHAP contribution aggregated to an original input feature."""

    feature: str
    feature_value: str
    shap_value: float
    direction: Literal["higher", "lower"]


class ExplainResponse(BaseModel):
    """A saved-model prediction plus its top local SHAP evidence."""

    prediction: Literal["Fraudulent", "Legitimate"]
    fraud_probability: float = Field(ge=0, le=1)
    top_contributing_features: list[FeatureContribution]


class FairnessGroupMetric(BaseModel):
    """Outcome and prediction rates for one existing dataset group."""

    group: str
    sample_count: int
    actual_fraud_rate: float
    predicted_fraud_rate: float
    true_positive_rate: float | None
    false_positive_rate: float | None


class FairnessAttributeSummary(BaseModel):
    """Group-level metrics and a basic selection-rate gap for one attribute."""

    selection_rate_gap: float
    groups: list[FairnessGroupMetric]


class FairnessResponse(BaseModel):
    """A basic held-out group comparison with required interpretive limitations."""

    evaluation_rows: int
    groups_checked: list[str]
    overall_predicted_fraud_rate: float
    summaries: dict[str, FairnessAttributeSummary]
    limitations: list[str]
