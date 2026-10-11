"""
Tests for model loading, missing model handling, input formatting, and response structure.
Mocks model inference and tokenization so tests run without requiring GPU or live weights.
"""
from pathlib import Path
import sys
from unittest.mock import MagicMock
import pytest
import torch

# Ensure project root is available on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.predict import FakeNewsPredictor, ModelNotLoadedError


def test_missing_model_directory_handling(tmp_path):
    """Predictor initialized with non-existent or empty folder must report is_loaded=False."""
    empty_dir = tmp_path / "non_existent_model_dir"
    predictor = FakeNewsPredictor(model_path=empty_dir)

    # Calling load_model should return False without crashing
    assert predictor.load_model() is False
    assert predictor.is_loaded is False


def test_predict_without_loaded_model_raises_error(tmp_path):
    """Attempting inference without loaded weights must raise ModelNotLoadedError."""
    predictor = FakeNewsPredictor(model_path=tmp_path / "empty")
    predictor.is_loaded = False

    with pytest.raises(ModelNotLoadedError) as exc_info:
        predictor.predict("Some article text")
    assert "Model weights not found" in str(exc_info.value)


def test_predict_rejects_empty_input(tmp_path):
    """Predictor must reject empty, whitespace, or None text."""
    predictor = FakeNewsPredictor(model_path=tmp_path)
    predictor.is_loaded = True
    predictor.model = MagicMock()
    predictor.tokenizer = MagicMock()

    with pytest.raises(ValueError):
        predictor.predict("")

    with pytest.raises(ValueError):
        predictor.predict("   \n\t  ")


def test_prediction_response_structure_real_news(tmp_path):
    """Verify inference output format, probability summation, and Real News mapping."""
    predictor = FakeNewsPredictor(model_path=tmp_path)
    predictor.is_loaded = True
    predictor.device = torch.device("cpu")

    # Mock tokenizer
    mock_tok = MagicMock()
    mock_tok.return_value = {
        "input_ids": torch.tensor([[101, 102]]),
        "attention_mask": torch.tensor([[1, 1]]),
    }
    predictor.tokenizer = mock_tok

    # Mock model output: Logits favoring Class 0 (Real News)
    # Logits: [2.5, -2.5] -> Softmax gives ~0.99 for Real News
    mock_model = MagicMock()
    mock_outputs = MagicMock()
    mock_outputs.logits = torch.tensor([[2.5, -2.5]])
    mock_model.return_value = mock_outputs
    predictor.model = mock_model

    result = predictor.predict(
        text="The prime minister and federal cabinet announced the infrastructure package.",
        title="Infrastructure Plan",
    )

    # Structure assertions
    assert "label" in result
    assert "label_name" in result
    assert "confidence" in result
    assert "probabilities" in result
    assert "cleaned_text" in result

    # Class 0 assertions (Real News)
    assert result["label"] == 0
    assert result["label_name"] == "Real News"
    assert result["confidence"] > 0.90
    assert result["probabilities"]["Real News"] == result["confidence"]
    assert abs(result["probabilities"]["Real News"] + result["probabilities"]["Fake News"] - 1.0) < 1e-3


def test_prediction_response_structure_fake_news(tmp_path):
    """Verify inference output format and Fake News mapping (Class 1)."""
    predictor = FakeNewsPredictor(model_path=tmp_path)
    predictor.is_loaded = True
    predictor.device = torch.device("cpu")

    mock_tok = MagicMock()
    mock_tok.return_value = {
        "input_ids": torch.tensor([[101, 102]]),
        "attention_mask": torch.tensor([[1, 1]]),
    }
    predictor.tokenizer = mock_tok

    # Logits favoring Class 1 (Fake News): [-3.0, 3.0]
    mock_model = MagicMock()
    mock_outputs = MagicMock()
    mock_outputs.logits = torch.tensor([[-3.0, 3.0]])
    mock_model.return_value = mock_outputs
    predictor.model = mock_model

    result = predictor.predict(
        text="SHOCKING: Extraterrestrial beings have taken over the broadcast towers!",
        title="Alien Infiltration",
    )

    # Class 1 assertions (Fake News)
    assert result["label"] == 1
    assert result["label_name"] == "Fake News"
    assert result["confidence"] > 0.90
    assert result["probabilities"]["Fake News"] == result["confidence"]
    assert abs(result["probabilities"]["Real News"] + result["probabilities"]["Fake News"] - 1.0) < 1e-3
