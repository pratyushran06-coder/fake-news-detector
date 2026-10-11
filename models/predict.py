"""
Shared, reusable prediction module for Fake News Detector.
Used across Backend API, Streamlit Frontend, and CLI utilities.
Ensures identical preprocessing, inference logic, and label mappings.
"""
from pathlib import Path
import os
import sys
from typing import Any, Dict, Optional, Tuple

# Ensure project root is available on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Suppress Hugging Face symlink warning on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import config
from utils.preprocess import clean_text, combine_title_and_text

# Label mappings (Strict project standard: 0 = Real News, 1 = Fake News)
LABEL_MAPPING: Dict[int, str] = {0: "Real News", 1: "Fake News"}
REVERSE_LABEL_MAPPING: Dict[str, int] = {"Real News": 0, "Fake News": 1}


class ModelNotLoadedError(Exception):
    """Raised when inference is requested but the model has not been loaded or trained."""
    pass


class FakeNewsPredictor:
    """
    Thread-safe model wrapper for news verification inference.
    """

    def __init__(self, model_path: Optional[Path] = None, max_length: int = 256):
        self.model_path: Path = Path(model_path) if model_path else self._resolve_default_model_path()
        self.max_length: int = max_length
        self.model: Optional[AutoModelForSequenceClassification] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.is_loaded: bool = False
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _resolve_default_model_path(self) -> Path:
        """Find the configured or latest saved model directory."""
        configured = getattr(config, "SAVED_MODEL_PATH", None)
        if configured and Path(configured).exists() and self._is_valid_model_dir(Path(configured)):
            return Path(configured)

        # Common model locations
        candidates = [
            config.ROOT_DIR / "saved_models" / "bert_fake_news",
            config.ROOT_DIR / "saved_models" / "distilbert_fake_news",
            config.ROOT_DIR / "saved_models" / "smoke_test_model",
        ]
        for candidate in candidates:
            if self._is_valid_model_dir(candidate):
                return candidate

        return config.ROOT_DIR / "saved_models" / "bert_fake_news"

    @staticmethod
    def _is_valid_model_dir(path: Path) -> bool:
        """Verify if a directory contains necessary model artifacts."""
        if not path.is_dir():
            return False
        has_config = (path / "config.json").exists()
        has_weights = (
            (path / "model.safetensors").exists()
            or (path / "pytorch_model.bin").exists()
        )
        return has_config and has_weights

    def load_model(self) -> bool:
        """
        Load tokenizer and model weights from disk into memory.
        Returns True if successfully loaded, False otherwise.
        """
        if not self._is_valid_model_dir(self.model_path):
            self.is_loaded = False
            return False

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_path))
            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
            return True
        except Exception as exc:
            self.is_loaded = False
            self.model = None
            self.tokenizer = None
            raise RuntimeError(f"Failed to load model from {self.model_path}: {exc}") from exc

    def predict(self, text: str, title: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform prediction on news text with headline.
        
        Args:
            text: Raw body text of the article.
            title: Optional headline.
            
        Returns:
            Dictionary containing label (0 or 1), human-readable label_name,
            confidence score, and class probabilities.
            
        Raises:
            ModelNotLoadedError: If the model has not been loaded.
            ValueError: If input text is empty or invalid.
        """
        if not self.is_loaded or self.model is None or self.tokenizer is None:
            raise ModelNotLoadedError(
                f"Model weights not found or not loaded from '{self.model_path}'. "
                "Please train a model first using 'python models/train_bert.py'."
            )

        # 1. Validate input text
        if text is None or not isinstance(text, str) or not text.strip():
            raise ValueError("Input news text cannot be empty or blank.")

        # 2. Combine title and text
        combined = combine_title_and_text(title, text)

        # 3. Clean text using the standard preprocessing pipeline
        cleaned = clean_text(combined, strip_wire_byline=True, preserve_negations=True)
        if not cleaned:
            # If aggressive cleaning leaves nothing (e.g. only punctuation/links), fall back to raw stripped
            cleaned = text.strip()[:self.max_length]

        # 4. Tokenize
        inputs = self.tokenizer(
            cleaned,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 5. Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).squeeze().tolist()

        # In case of single sample or batch dimension
        if isinstance(probs, float):
            probs = [1.0 - probs, probs]

        prob_real = float(probs[0])
        prob_fake = float(probs[1])

        # Exact project label mapping: 0 = Real News, 1 = Fake News
        predicted_class = 1 if prob_fake >= 0.5 else 0
        predicted_label_name = LABEL_MAPPING[predicted_class]
        confidence = prob_fake if predicted_class == 1 else prob_real

        return {
            "label": predicted_class,
            "label_name": predicted_label_name,
            "confidence": round(confidence, 4),
            "probabilities": {
                LABEL_MAPPING[0]: round(prob_real, 4),
                LABEL_MAPPING[1]: round(prob_fake, 4),
            },
            "cleaned_text": cleaned,
        }


# Global singleton predictor instance for backend & shared usage
shared_predictor = FakeNewsPredictor()
