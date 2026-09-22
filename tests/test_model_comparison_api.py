"""Tests for the saved text-model comparison endpoint."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_model_comparison_returns_saved_results() -> None:
    """The endpoint exposes saved metrics without starting model training."""

    response = client.get("/model-comparison")

    assert response.status_code == 200
    body = response.json()
    assert {"models", "best_by_f1", "best_by_roc_auc"}.issubset(body)
    assert len(body["models"]) == 5
    assert {model["model"] for model in body["models"]} == {
        "CNN",
        "Simple RNN",
        "LSTM",
        "Attention/Transformer",
        "Pretrained MiniLM",
    }
    for model in body["models"]:
        assert {
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "inference_time_seconds",
        }.issubset(model)
