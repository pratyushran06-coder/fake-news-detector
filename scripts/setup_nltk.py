"""
Download and verify required NLTK corpora (English stopwords).
Run this script after installing dependencies:
    python scripts/setup_nltk.py
"""
import sys
from pathlib import Path

# Ensure root directory is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import nltk


def setup_nltk_corpora() -> bool:
    """Download the NLTK stopwords corpus if not already available."""
    print("Checking NLTK 'stopwords' corpus...")
    try:
        nltk.data.find("corpora/stopwords")
        print("[OK] NLTK 'stopwords' corpus is already installed.")
        return True
    except LookupError:
        print("Downloading NLTK 'stopwords' corpus...")
        success = nltk.download("stopwords", quiet=False)
        if success:
            print("[OK] Successfully installed NLTK 'stopwords' corpus.")
            return True
        else:
            print("[ERROR] Failed to download NLTK 'stopwords' corpus.", file=sys.stderr)
            return False


if __name__ == "__main__":
    ok = setup_nltk_corpora()
    sys.exit(0 if ok else 1)
