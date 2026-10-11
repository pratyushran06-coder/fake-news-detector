"""
FastAPI backend service for Fake News Detector.
Exposes health checks and a high-performance prediction API.
Model weights and tokenizers are loaded once at startup via lifespan events.
"""
from pathlib import Path
from contextlib import asynccontextmanager
import sys
from typing import Dict, Optional

# Ensure project root is available on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

import config
from models.predict import (
    FakeNewsPredictor,
    ModelNotLoadedError,
    shared_predictor,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the trained model and tokenizer once on server startup.
    Gracefully handles missing model files by setting is_loaded=False.
    """
    print(f"[Startup] Checking for model weights at: {shared_predictor.model_path}")
    loaded = shared_predictor.load_model()
    if loaded:
        print(f"[Startup] Model loaded successfully on device: {shared_predictor.device}")
    else:
        print(f"[Startup Notice] Model files not found at '{shared_predictor.model_path}'. "
              f"API is online, but /predict will return 503 until a model is trained.")
    yield
    print("[Shutdown] Cleaning up API resources.")


app = FastAPI(
    title="Fake News Detector API",
    description="Production REST API for detecting fake news articles using deep learning.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for frontend and external clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================================
# Request / Response Models
# =====================================================================
class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=50000,
        description="The article text or claim to evaluate (1 to 50,000 characters).",
        examples=["Breaking news: The government has announced a new tax policy today."],
    )
    title: Optional[str] = Field(
        None,
        max_length=1000,
        description="Optional article headline to improve contextual prediction.",
        examples=["Government announces tax updates"],
    )

    @field_validator("text")
    @classmethod
    def validate_not_blank(cls, v: str) -> str:
        trimmed = v.strip()
        if not trimmed:
            raise ValueError("Article text cannot be empty or whitespace only.")
        return trimmed


class PredictResponse(BaseModel):
    label: int = Field(..., description="Predicted class index (0 = Real News, 1 = Fake News).")
    label_name: str = Field(..., description="Human-readable prediction ('Real News' or 'Fake News').")
    confidence: float = Field(..., description="Confidence score of the prediction (between 0.0 and 1.0).")
    probabilities: Dict[str, float] = Field(..., description="Class probabilities for Real News and Fake News.")
    cleaned_text: str = Field(..., description="Normalized text after preprocessing and byline removal.")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
    device: str
    version: str


# =====================================================================
# API Endpoints
# =====================================================================
@app.get("/", summary="Root Welcome / Info")
def home():
    """Welcome endpoint confirming API availability."""
    return {
        "message": "Fake News Detector API",
        "status": "online",
        "version": "1.0.0",
        "docs_url": "/docs",
    }


@app.get("/health", response_model=HealthResponse, summary="Service & Model Health Check")
def health_check():
    """
    Check the readiness of the service and whether model weights are loaded.
    """
    return HealthResponse(
        status="healthy",
        model_loaded=shared_predictor.is_loaded,
        model_path=str(shared_predictor.model_path),
        device=str(shared_predictor.device),
        version="1.0.0",
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Predict News Authenticity",
    responses={
        200: {"description": "Successful prediction with confidence scores."},
        422: {"description": "Validation error (empty text, invalid format, or text too long)."},
        503: {"description": "Model weights missing or not trained yet."},
        500: {"description": "Internal inference error."},
    },
)
def predict_news(request: PredictRequest):
    """
    Analyze news text and determine whether it is Real News (0) or Fake News (1).
    """
    if not shared_predictor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"Model weights not loaded from '{shared_predictor.model_path}'. "
                "Please train a model using 'python models/train_bert.py' before requesting predictions."
            ),
        )

    try:
        result = shared_predictor.predict(text=request.text, title=request.title)
        return PredictResponse(**result)
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(ve),
        )
    except ModelNotLoadedError as mne:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(mne),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference processing error: {exc}",
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=config.BACKEND_HOST, port=config.BACKEND_PORT, reload=True)