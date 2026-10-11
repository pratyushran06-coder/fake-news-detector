"""
API Client for communicating between Streamlit frontend and FastAPI backend.
Separates network transport and response handling from presentation logic.
"""
from pathlib import Path
import os
import sys
from typing import Any, Dict, Optional
import requests

# Ensure project root is available on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config


class APIClientError(Exception):
    """Base exception for frontend API client errors."""
    pass


class FakeNewsAPIClient:
    """
    HTTP client for the Fake News Detector FastAPI service.
    """

    def __init__(self, base_url: Optional[str] = None, timeout: float = 15.0):
        self.base_url = (base_url or getattr(config, "BACKEND_URL", "http://127.0.0.1:8000")).rstrip("/")
        self.timeout = timeout

    def check_health(self) -> Dict[str, Any]:
        """
        Check health and model readiness of backend.
        
        Returns:
            Dict with 'online', 'model_loaded', 'device', and 'message'.
        """
        url = f"{self.base_url}/health"
        try:
            resp = requests.get(url, timeout=4.0)
            if resp.status_code == 200:
                data = resp.json()
                return {
                    "online": True,
                    "model_loaded": data.get("model_loaded", False),
                    "device": data.get("device", "unknown"),
                    "model_path": data.get("model_path", ""),
                    "error": None,
                }
            return {
                "online": True,
                "model_loaded": False,
                "device": "unknown",
                "error": f"Health check returned HTTP {resp.status_code}",
            }
        except requests.exceptions.ConnectionError:
            return {
                "online": False,
                "model_loaded": False,
                "device": "unknown",
                "error": f"Cannot connect to backend at {self.base_url}. Is the FastAPI server running?",
            }
        except requests.exceptions.Timeout:
            return {
                "online": False,
                "model_loaded": False,
                "device": "unknown",
                "error": "Connection to backend timed out.",
            }
        except Exception as exc:
            return {
                "online": False,
                "model_loaded": False,
                "device": "unknown",
                "error": f"Unexpected health check error: {exc}",
            }

    def predict(self, text: str, title: Optional[str] = None) -> Dict[str, Any]:
        """
        Send news article text to the /predict endpoint.
        
        Args:
            text: Article text (must not be empty).
            title: Optional headline.
            
        Returns:
            Dict containing:
                - success: bool
                - data: dict (if successful)
                - error_type: str (if failed)
                - message: str (human-readable error description)
        """
        # 1. Local validation
        if not text or not text.strip():
            return {
                "success": False,
                "error_type": "VALIDATION_ERROR",
                "message": "Please enter or paste news article text before predicting.",
            }

        url = f"{self.base_url}/predict"
        payload = {"text": text.strip()}
        if title and title.strip():
            payload["title"] = title.strip()

        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)

            # 2. Handle HTTP 200 Success
            if resp.status_code == 200:
                return {
                    "success": True,
                    "data": resp.json(),
                    "error_type": None,
                    "message": "Prediction succeeded.",
                }

            # 3. Handle HTTP 503 Model Not Loaded
            if resp.status_code == 503:
                detail = resp.json().get("detail", "Model is not loaded or has not been trained yet.")
                return {
                    "success": False,
                    "error_type": "MODEL_UNAVAILABLE",
                    "message": detail,
                }

            # 4. Handle HTTP 422 Input Validation Error
            if resp.status_code == 422:
                detail = resp.json().get("detail", "Input validation error.")
                if isinstance(detail, list) and len(detail) > 0:
                    detail = detail[0].get("msg", str(detail))
                return {
                    "success": False,
                    "error_type": "VALIDATION_ERROR",
                    "message": f"Invalid input: {detail}",
                }

            # 5. Handle Other HTTP Errors
            return {
                "success": False,
                "error_type": "HTTP_ERROR",
                "message": f"Server error (HTTP {resp.status_code}): {resp.text}",
            }

        except requests.exceptions.ConnectionError:
            return {
                "success": False,
                "error_type": "CONNECTION_ERROR",
                "message": (
                    f"Unable to reach the backend service at {self.base_url}. "
                    "Please ensure the FastAPI backend is running (uvicorn backend.main:app)."
                ),
            }
        except requests.exceptions.Timeout:
            return {
                "success": False,
                "error_type": "TIMEOUT_ERROR",
                "message": "Inference request timed out. The model may be running on a constrained CPU.",
            }
        except Exception as exc:
            return {
                "success": False,
                "error_type": "UNKNOWN_ERROR",
                "message": f"An unexpected error occurred during prediction: {exc}",
            }
