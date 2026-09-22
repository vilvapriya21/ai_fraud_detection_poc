"""Request and response schemas for fraud predictions."""

from typing import Literal

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Validated transaction inputs required by the saved model pipeline."""

    hour_of_day: int = Field(ge=0, le=23)
    is_weekend: int = Field(ge=0, le=1)
    is_night_transaction: int = Field(ge=0, le=1)
    country: str = Field(min_length=1)
    city: str = Field(min_length=1)
    merchant_category: str = Field(min_length=1)
    payment_method: str = Field(min_length=1)
    device_type: str = Field(min_length=1)
    customer_age: int = Field(gt=0)
    credit_score: int = Field(gt=0)
    account_age_years: float = Field(ge=0)
    account_balance: float = Field(ge=0)
    transaction_amount: float = Field(ge=0)
    num_prev_transactions: int = Field(ge=0)
    transaction_freq_monthly: int = Field(ge=0)
    distance_from_home_km: float = Field(ge=0)
    time_since_last_txn_hrs: float = Field(ge=0)
    is_international: int = Field(ge=0, le=1)
    failed_attempts: int = Field(ge=0)
    pin_changed_recently: int = Field(ge=0, le=1)


class PredictionResponse(BaseModel):
    """Fraud prediction result from the selected model pipeline."""

    prediction: Literal["Fraudulent", "Legitimate"]
    fraud_probability: float = Field(ge=0, le=1)
    risk_level: Literal["Low", "Medium", "High"]


class HealthResponse(BaseModel):
    """Model availability reported by the service health endpoint."""

    status: Literal["healthy", "unhealthy"]
    model_loaded: bool
