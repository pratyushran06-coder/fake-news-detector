"""
Unit tests for the frontend API client (frontend/api_client.py).
Tests all success, failure, timeout, and connection edge cases.
"""
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch
import pytest
import requests

# Ensure project root is available on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from frontend.api_client import FakeNewsAPIClient


@pytest.fixture
def client():
    return FakeNewsAPIClient(base_url="http://127.0.0.1:8000")


def test_client_empty_text_validation(client):
    """Client should catch empty or whitespace-only inputs without making network calls."""
    res_empty = client.predict("")
    assert res_empty["success"] is False
    assert res_empty["error_type"] == "VALIDATION_ERROR"

    res_ws = client.predict("   \n\t  ")
    assert res_ws["success"] is False
    assert res_ws["error_type"] == "VALIDATION_ERROR"


@patch("requests.post")
def test_client_predict_success(mock_post, client):
    """Client should parse successful HTTP 200 prediction response."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "label": 0,
        "label_name": "Real News",
        "confidence": 0.952,
        "probabilities": {"Real News": 0.952, "Fake News": 0.048},
        "cleaned_text": "sample clean news",
    }
    mock_post.return_value = mock_resp

    res = client.predict("Legitimate economic report here.", title="Economy News")
    assert res["success"] is True
    assert res["error_type"] is None
    assert res["data"]["label"] == 0
    assert res["data"]["label_name"] == "Real News"
    assert res["data"]["confidence"] == 0.952


@patch("requests.post")
def test_client_model_unavailable_503(mock_post, client):
    """Client should handle HTTP 503 when model is not trained/loaded."""
    mock_resp = MagicMock()
    mock_resp.status_code = 503
    mock_resp.json.return_value = {"detail": "Model is not loaded or has not been trained yet."}
    mock_post.return_value = mock_resp

    res = client.predict("Some article text.")
    assert res["success"] is False
    assert res["error_type"] == "MODEL_UNAVAILABLE"
    assert "not been trained yet" in res["message"]


@patch("requests.post")
def test_client_validation_error_422(mock_post, client):
    """Client should handle HTTP 422 input validation errors."""
    mock_resp = MagicMock()
    mock_resp.status_code = 422
    mock_resp.json.return_value = {
        "detail": [{"msg": "Article text cannot be empty or whitespace only."}]
    }
    mock_post.return_value = mock_resp

    res = client.predict("Some text")
    assert res["success"] is False
    assert res["error_type"] == "VALIDATION_ERROR"


@patch("requests.post")
def test_client_connection_error(mock_post, client):
    """Client should catch connection failures gracefully when backend is offline."""
    mock_post.side_effect = requests.exceptions.ConnectionError("Connection refused")

    res = client.predict("Breaking news story")
    assert res["success"] is False
    assert res["error_type"] == "CONNECTION_ERROR"
    assert "Unable to reach the backend service" in res["message"]


@patch("requests.post")
def test_client_timeout_error(mock_post, client):
    """Client should catch request timeouts gracefully."""
    mock_post.side_effect = requests.exceptions.Timeout("Read timed out")

    res = client.predict("Breaking news story")
    assert res["success"] is False
    assert res["error_type"] == "TIMEOUT_ERROR"


@patch("requests.get")
def test_client_health_check_online(mock_get, client):
    """Client health check should parse active status."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "status": "healthy",
        "model_loaded": True,
        "device": "cpu",
        "model_path": "saved_models/bert_fake_news",
    }
    mock_get.return_value = mock_resp

    health = client.check_health()
    assert health["online"] is True
    assert health["model_loaded"] is True
    assert health["device"] == "cpu"


@patch("requests.get")
def test_client_health_check_offline(mock_get, client):
    """Client health check should report offline when connection fails."""
    mock_get.side_effect = requests.exceptions.ConnectionError("Offline")

    health = client.check_health()
    assert health["online"] is False
    assert health["model_loaded"] is False
    assert "Cannot connect to backend" in health["error"]
