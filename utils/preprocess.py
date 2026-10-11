"""
Text preprocessing utilities for Fake News Detector.
Includes automated NLTK corpus verification, robust type handling,
HTML entity decoding, URL stripping, and negation-preserving stopword filtering.
"""
from pathlib import Path
import html
import math
import re
import sys
from typing import Any, Optional, Set

# Ensure project root is available on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import nltk

# Negation words and polarity markers that must NOT be removed in fake news classification.
# Stripping negations inverts factual truth (e.g., "was not found guilty" -> "was found guilty").
NEGATION_WORDS: Set[str] = {
    "no", "not", "nor", "neither", "never", "none", "nobody",
    "nowhere", "nothing", "cannot", "without", "against"
}


def get_stopwords(preserve_negations: bool = True) -> Set[str]:
    """
    Retrieve NLTK English stopwords with robust error handling and self-healing.
    
    Args:
        preserve_negations: If True, excludes critical negation words (e.g. 'not', 'no')
                            from the stopword set to prevent semantic inversion in fake news detection.
    
    Returns:
        Set of stopwords.
        
    Raises:
        LookupError: If NLTK corpus is missing and automated download fails.
    """
    try:
        from nltk.corpus import stopwords
        base_words = set(stopwords.words("english"))
    except (LookupError, AttributeError):
        try:
            nltk.download("stopwords", quiet=True)
            from nltk.corpus import stopwords
            base_words = set(stopwords.words("english"))
        except Exception as exc:
            raise LookupError(
                "NLTK 'stopwords' corpus is missing and could not be downloaded automatically.\n"
                "Please run:\n"
                "    python scripts/setup_nltk.py\n"
                "or:\n"
                "    python -m nltk.downloader stopwords"
            ) from exc

    if preserve_negations:
        return base_words - NEGATION_WORDS
    return base_words


# Backwards-compatible module-level stop_words set (preserves negations by default)
stop_words: Set[str] = get_stopwords(preserve_negations=True)


def clean_text(
    text: Optional[Any],
    strip_wire_byline: bool = True,
    preserve_negations: bool = True,
) -> str:
    """
    Clean and normalize raw news text safely.
    
    Handles None, NaN, numeric types, and malformed strings without crashing.
    Decodes HTML entities, removes HTML markup, removes URLs, replaces punctuation
    with spaces (preventing word concatenation), and cleans extra whitespace.
    
    Args:
        text: Raw text string or object (None, NaN, float, int handled safely).
        strip_wire_byline: If True, strips wire service headers (e.g., 'WASHINGTON (Reuters) -')
                          to prevent major dataset artifact leakage.
        preserve_negations: If True, retains negation words (e.g., 'not', 'no', 'never')
                            which are critical for factual accuracy assessment.
                            
    Returns:
        Cleaned, lowercased, whitespace-normalized string.
    """
    if text is None:
        return ""

    if not isinstance(text, str):
        if isinstance(text, float) and (math.isnan(text) or str(text) == "nan"):
            return ""
        text = str(text)

    text = text.strip()
    if not text:
        return ""

    # 1. Unescape HTML entities (&amp; -> &, &quot; -> ", etc.)
    text = html.unescape(text)

    # 2. Remove HTML tags (<p>, <br>, etc.)
    text = re.sub(r"<[^>]+>", " ", text)

    # 3. Remove wire service bylines (e.g. "WASHINGTON (Reuters) - ")
    if strip_wire_byline:
        text = re.sub(r"^.*?\((?:reuters|REUTERS)\)\s*[-—]\s*", " ", text, flags=re.IGNORECASE)

    # 4. Remove URLs (http, https, www, ftp)
    text = re.sub(r"https?://\S+|www\.\S+|ftp://\S+", " ", text)

    # 5. Lowercase
    text = text.lower()

    # 6. Replace punctuation, numbers, and non-alphabetic characters with whitespace
    # Using spaces ensures hyphenated words ("fact-checked") do not become merged ("factchecked")
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # 7. Tokenize and filter stopwords
    active_stopwords = stop_words if preserve_negations else get_stopwords(preserve_negations=False)
    words = [word for word in text.split() if word not in active_stopwords]

    # 8. Reunite with single whitespace
    return " ".join(words)


def combine_title_and_text(title: Optional[Any], text: Optional[Any]) -> str:
    """
    Combine news headline (title) and article body safely.
    
    If article body is empty or whitespace, uses title so the row remains usable.
    If headline is already prefixed in the text, avoids duplicate repetition.
    
    Args:
        title: Headline string or object.
        text: Body string or object.
        
    Returns:
        Combined text string.
    """
    clean_t = ""
    if title is not None and not (isinstance(title, float) and math.isnan(title)):
        clean_t = str(title).strip()

    clean_b = ""
    if text is not None and not (isinstance(text, float) and math.isnan(text)):
        clean_b = str(text).strip()

    if clean_t and clean_b:
        if clean_b.lower().startswith(clean_t.lower()):
            return clean_b
        return f"{clean_t} - {clean_b}"
    elif clean_b:
        return clean_b
    elif clean_t:
        return clean_t
    return ""
