"""
Integration test suite for the trained DistilBERT fake news classification model.
Tests saved_models/distilbert_fake_news/ independently without modifying the running API.
"""
from pathlib import Path
import pytest
import torch

from models.predict import FakeNewsPredictor, LABEL_MAPPING
from fastapi.testclient import TestClient
from backend.main import app

MODEL_DIR = Path(__file__).resolve().parent.parent / "saved_models" / "distilbert_fake_news"


@pytest.fixture(scope="module")
def distilbert_predictor():
    """Load DistilBERT in isolation for read-only integration verification."""
    predictor = FakeNewsPredictor(model_path=MODEL_DIR, max_length=128)
    assert predictor.load_model() is True, f"Failed to load model from {MODEL_DIR}"
    return predictor


def test_label_mapping_alignment(distilbert_predictor):
    """Confirm project standard: 0 = Real News, 1 = Fake News."""
    assert LABEL_MAPPING[0] == "Real News"
    assert LABEL_MAPPING[1] == "Fake News"
    assert distilbert_predictor.model.config.id2label[0] == "Real News"
    assert distilbert_predictor.model.config.id2label[1] == "Fake News"


def test_sample_predictions_consistency(distilbert_predictor):
    """
    Verify inference calculation consistency:
    - Probabilities sum to 1.0
    - Confidence corresponds to predicted class probability
    - Label aligns with probability threshold >= 0.5
    """
    samples = [
        ("Federal Reserve holds benchmark interest rates steady", "Economic Outlook"),
        ("Secret alien syndicate controls global climate with space mirrors", "Shock Report"),
    ]
    for text, title in samples:
        res = distilbert_predictor.predict(text=text, title=title)
        prob_real = res["probabilities"]["Real News"]
        prob_fake = res["probabilities"]["Fake News"]
        
        assert round(prob_real + prob_fake, 4) == 1.0
        expected_class = 1 if prob_fake >= 0.5 else 0
        assert res["label"] == expected_class
        assert res["confidence"] == (prob_fake if expected_class == 1 else prob_real)
        assert isinstance(res["cleaned_text"], str)


def test_empty_and_whitespace_rejection(distilbert_predictor):
    """Ensure empty or blank inputs raise ValueError."""
    for empty_val in ["", "   ", "\t\n  \r"]:
        with pytest.raises(ValueError):
            distilbert_predictor.predict(text=empty_val)


def test_long_text_truncation_handling(distilbert_predictor):
    """Ensure long articles (e.g. 20,000+ chars) do not crash and truncate gracefully."""
    long_text = "The committee held an official briefing on infrastructure developments. " * 300
    res = distilbert_predictor.predict(text=long_text, title="Long Document")
    assert res["label"] in [0, 1]
    assert 0.0 <= res["confidence"] <= 1.0


def test_api_schema_and_validation_errors():
    """Verify FastAPI endpoint schema validation and error status codes."""
    with TestClient(app) as client:
        # Blank text rejected with 422
        resp_blank = client.post("/predict", json={"text": "   "})
        assert resp_blank.status_code == 422
        assert "detail" in resp_blank.json()

        # Excessively long text rejected with 422 (>50,000 characters)
        resp_too_long = client.post("/predict", json={"text": "a" * 50001})
        assert resp_too_long.status_code == 422

        # Missing required text field rejected with 422
        resp_missing = client.post("/predict", json={"title": "Only a title"})
        assert resp_missing.status_code == 422
