"""
Central configuration and path resolution for Fake News Detector.
All project-relative paths are anchored via pathlib to the project root directory.
"""
from pathlib import Path
import os
import sys

# Project root directory
ROOT_DIR = Path(__file__).resolve().parent

# Ensure project root is on sys.path so any subfolder or runner can import project packages
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Optional .env loading
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT_DIR / ".env")
except ImportError:
    pass

# Core directory paths
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
SAVED_MODELS_DIR = ROOT_DIR / "saved_models"
SCRAPER_DIR = ROOT_DIR / "scraper"
UTILS_DIR = ROOT_DIR / "utils"
BACKEND_DIR = ROOT_DIR / "backend"
FRONTEND_DIR = ROOT_DIR / "frontend"

# Dataset file paths
FAKE_CSV_PATH = DATA_DIR / "Fake.csv"
TRUE_CSV_PATH = DATA_DIR / "True.csv"
PROCESSED_CSV_PATH = DATA_DIR / "processed_news.csv"

# Service and API configurations
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
BACKEND_HOST = os.getenv("BACKEND_HOST", "127.0.0.1")
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8000"))
BACKEND_URL = os.getenv("BACKEND_URL", f"http://{BACKEND_HOST}:{BACKEND_PORT}")
MODEL_ARCHITECTURE = os.getenv("MODEL_ARCHITECTURE", "distilbert-base-uncased")
SAVED_MODEL_PATH = ROOT_DIR / os.getenv("SAVED_MODEL_DIR", "saved_models/distilbert_fake_news")
