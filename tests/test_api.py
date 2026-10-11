"""
Automated API tests for FastAPI backend using TestClient.
Tests validate endpoint behaviors, request validation, error states, and predictions.
Crucially, these tests do NOT require downloading a pretrained model or training.
"""
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch
import pytest

# Ensure project root is available on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastapi.testclient import TestClient
from backend.main import app
from models.predict import shared_predictor


@pytest.fixture
def client():
    """Create a TestClient instance without running lifespan model loads."""
    with TestClient(app) as test_client:
        yield test_client


def test_root_endpoint(client):
    """GET / should return 200 and online status."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Fake News Detector API"
    assert data["status"] == "online"
    assert "version" in data


def test_health_endpoint_unloaded(client):
    """GET /health should report status when model is not loaded."""
    original_state = shared_predictor.is_loaded
    try:
        shared_predictor.is_loaded = False
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is False
    finally:
        shared_predictor.is_loaded = original_state


def test_predict_model_unloaded_returns_503(client):
    """POST /predict should return 503 Service Unavailable if model is missing/unloaded."""
    original_state = shared_predictor.is_loaded
    try:
        shared_predictor.is_loaded = False
        response = client.post("/predict", json={"text": "Some sample news text"})
        assert response.status_code == 503
        data = response.json()
        assert "detail" in data
        assert "Please train a model" in data["detail"]
    finally:
        shared_predictor.is_loaded = original_state


def test_predict_empty_text_returns_422(client):
    """POST /predict should reject empty string or whitespace with 422 Unprocessable Entity."""
    # Test empty string
    resp_empty = client.post("/predict", json={"text": ""})
    assert resp_empty.status_code == 422

    # Test whitespace string
    resp_ws = client.post("/predict", json={"text": "   \n\t  "})
    assert resp_ws.status_code == 422


def test_predict_missing_text_field_returns_422(client):
    """POST /predict should reject requests missing the required 'text' field."""
    response = client.post("/predict", json={"title": "Headline only"})
    assert response.status_code == 422


def test_predict_text_exceeds_max_length_returns_422(client):
    """POST /predict should reject text exceeding the maximum allowed character limit."""
    huge_text = "word " * 11000  # > 50,000 characters
    response = client.post("/predict", json={"text": huge_text})
    assert response.status_code == 422


def test_predict_success_real_news_mocked(client):
    """Verify prediction endpoint with mocked Real News prediction (label 0)."""
    mock_prediction = {
        "label": 0,
        "label_name": "Real News",
        "confidence": 0.9421,
        "probabilities": {"Real News": 0.9421, "Fake News": 0.0579},
        "cleaned_text": "senate passes bill today",
    }

    original_state = shared_predictor.is_loaded
    try:
        shared_predictor.is_loaded = True
        with patch.object(shared_predictor, "predict", return_value=mock_prediction):
            response = client.post(
                "/predict",
                json={
                    "title": "Senate Approves Budget",
                    "text": "The Senate passed the proposed federal budget on Tuesday.",
                },
            )
            assert response.status_code == 200
            data = response.json()
            assert data["label"] == 0
            assert data["label_name"] == "Real News"
            assert data["confidence"] == 0.9421
            assert data["probabilities"]["Real News"] == 0.9421
            assert data["probabilities"]["Fake News"] == 0.0579
            assert data["cleaned_text"] == "senate passes bill today"
    finally:
        shared_predictor.is_loaded = original_state


def test_predict_success_fake_news_mocked(client):
    """Verify prediction endpoint with mocked Fake News prediction (label 1)."""
    mock_prediction = {
        "label": 1,
        "label_name": "Fake News",
        "confidence": 0.9850,
        "probabilities": {"Real News": 0.0150, "Fake News": 0.9850},
        "cleaned_text": "aliens landed secret bunker",
    }

    original_state = shared_predictor.is_loaded
    try:
        shared_predictor.is_loaded = True
        with patch.object(shared_predictor, "predict", return_value=mock_prediction):
            response = client.post(
                "/predict",
                json={"text": "SHOCKING: Extraterrestrial beings have taken over the bunker!"},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["label"] == 1
            assert data["label_name"] == "Fake News"
            assert data["confidence"] == 0.9850
            assert data["probabilities"]["Fake News"] == 0.9850
            assert data["probabilities"]["Real News"] == 0.0150
    finally:
        shared_predictor.is_loaded = original_state


def test_health_endpoint_loaded(client):
    """GET /health should report model_loaded: True when model is loaded."""
    original_state = shared_predictor.is_loaded
    try:
        shared_predictor.is_loaded = True
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["model_loaded"] is True
    finally:
        shared_predictor.is_loaded = original_state


def test_predict_internal_error_returns_500(client):
    """Inference failures should be caught and returned as HTTP 500."""
    original_state = shared_predictor.is_loaded
    try:
        shared_predictor.is_loaded = True
        with patch.object(shared_predictor, "predict", side_effect=RuntimeError("GPU OOM test error")):
            response = client.post("/predict", json={"text": "Valid news text"})
            assert response.status_code == 500
            data = response.json()
            assert "Inference processing error" in data["detail"]
    finally:
        shared_predictor.is_loaded = original_state


if __name__ == "__main__":
    pytest.main(["-v", "tests/test_api.py"])
