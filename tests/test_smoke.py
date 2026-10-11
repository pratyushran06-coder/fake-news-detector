"""
Fast Smoke Test Suite for Fake News Detector.
Runs an end-to-end sanity check across all core subsystems in ~1 second:
- Preprocessing and negation preservation
- Configuration and path resolution
- FastAPI API endpoints (GET / and GET /health)
- Model inference contract (mocked)
- Streamlit API client communication (mocked)
- Scraper parsing and credential handling (mocked)

Usage:
    pytest tests/test_smoke.py
or:
    pytest -m smoke
"""
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch
import pytest

# Ensure project root is available on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config
from utils.preprocess import clean_text, combine_title_and_text
from backend.main import app
from fastapi.testclient import TestClient
from frontend.api_client import FakeNewsAPIClient
from scraper.news_fetcher import fetch_top_headlines


@pytest.mark.smoke
def test_smoke_preprocessing():
    """Smoke check: Preprocessing correctly cleans text and preserves critical negations."""
    sample = "WASHINGTON (Reuters) - The bill did NOT pass: http://example.com"
    cleaned = clean_text(sample, strip_wire_byline=True, preserve_negations=True)
    assert "not" in cleaned, "Negation 'not' was stripped in preprocessing!"
    assert "reuters" not in cleaned, "Reuters wire byline was not stripped!"
    assert "http" not in cleaned, "URL was not stripped!"

    combined = combine_title_and_text("Headline", "Body text")
    assert combined == "Headline - Body text"


@pytest.mark.smoke
def test_smoke_config_and_paths():
    """Smoke check: Project directory structure and path references exist."""
    assert config.ROOT_DIR.exists()
    assert config.DATA_DIR.exists()
    assert config.FAKE_CSV_PATH.exists()
    assert config.TRUE_CSV_PATH.exists()


@pytest.mark.smoke
def test_smoke_fastapi_endpoints():
    """Smoke check: FastAPI core endpoints respond with HTTP 200."""
    with TestClient(app) as client:
        # 1. Root
        res_root = client.get("/")
        assert res_root.status_code == 200
        assert res_root.json()["status"] == "online"

        # 2. Health
        res_health = client.get("/health")
        assert res_health.status_code == 200
        assert res_health.json()["status"] == "healthy"


@pytest.mark.smoke
def test_smoke_prediction_contract():
    """Smoke check: Prediction API returns the expected schema and 0/1 label mapping."""
    mock_payload = {
        "label": 1,
        "label_name": "Fake News",
        "confidence": 0.98,
        "probabilities": {"Real News": 0.02, "Fake News": 0.98},
        "cleaned_text": "alien invasion confirmed",
    }
    with patch("backend.main.shared_predictor.is_loaded", True), \
         patch("backend.main.shared_predictor.predict", return_value=mock_payload):
        with TestClient(app) as client:
            res = client.post("/predict", json={"text": "Alien invasion announced today."})
            assert res.status_code == 200
            data = res.json()
            assert data["label"] == 1
            assert data["label_name"] == "Fake News"
            assert data["confidence"] == 0.98


@pytest.mark.smoke
def test_smoke_frontend_api_client():
    """Smoke check: Frontend API client parses successful prediction response."""
    client = FakeNewsAPIClient()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "label": 0,
        "label_name": "Real News",
        "confidence": 0.95,
        "probabilities": {"Real News": 0.95, "Fake News": 0.05},
        "cleaned_text": "fiscal policy update",
    }
    with patch("requests.post", return_value=mock_resp):
        res = client.predict("Fiscal policy article text.")
        assert res["success"] is True
        assert res["data"]["label"] == 0
        assert res["data"]["label_name"] == "Real News"


@pytest.mark.smoke
def test_smoke_news_scraper():
    """Smoke check: News scraper parses raw API response into standardized dictionary."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "status": "ok",
        "articles": [
            {
                "title": "Smoke Test Headline",
                "description": "Smoke test summary.",
                "source": {"name": "Test Source"},
                "url": "https://example.com/smoke",
                "publishedAt": "2026-10-11T12:00:00Z",
            }
        ],
    }
    with patch("requests.get", return_value=mock_resp):
        articles = fetch_top_headlines(api_key="smoke_test_key", page_size=1)
        assert len(articles) == 1
        assert articles[0]["title"] == "Smoke Test Headline"
        assert articles[0]["source"] == "Test Source"
