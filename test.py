"""
Environment and setup validation test script.
Verifies that core modules, paths, and NLTK resources are properly installed and accessible.
"""
from pathlib import Path
import sys

# Ensure project root is available on sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

print("[1/5] Checking core data science and NLP packages...")
import pandas as pd
import torch
import transformers
import sklearn
print(f"      - pandas: {pd.__version__}")
print(f"      - torch: {torch.__version__}")
print(f"      - transformers: {transformers.__version__}")
print(f"      - scikit-learn: {sklearn.__version__}")

print("[2/5] Checking web frameworks...")
import fastapi
import streamlit
import uvicorn
print(f"      - fastapi: {fastapi.__version__}")
print(f"      - streamlit: {streamlit.__version__}")
print(f"      - uvicorn: {uvicorn.__version__}")

print("[3/5] Checking NLTK and text preprocessing...")
from utils.preprocess import clean_text, stop_words
assert len(stop_words) > 0, "Stopwords set is empty!"
print(f"      - NLTK English stopwords loaded: {len(stop_words)} words")

print("[4/5] Checking configuration and path resolution...")
import config
assert config.ROOT_DIR.exists(), f"ROOT_DIR does not exist: {config.ROOT_DIR}"
assert config.DATA_DIR.exists(), f"DATA_DIR does not exist: {config.DATA_DIR}"
print(f"      - ROOT_DIR: {config.ROOT_DIR}")
print(f"      - DATA_DIR: {config.DATA_DIR}")

print("[5/5] Testing text cleaner function...")
cleaned = clean_text("WASHINGTON (Reuters) - Breaking: Visit http://news.com now!")
assert "reuters" not in cleaned, "Reuters wire byline was not stripped!"
assert "http" not in cleaned, "URL was not stripped!"
print(f"      - Sample output: '{cleaned}'")

print("\n>>> All setup validation checks passed successfully! Everything working! <<<")