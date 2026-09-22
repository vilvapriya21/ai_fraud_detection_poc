"""FastAPI application entry point for fraud prediction."""

from fastapi import FastAPI

from app.api.routes.prediction import router as prediction_router

app = FastAPI(title="AI Fraud Detection & Investigation API")
app.include_router(prediction_router)
