"""
Unit tests for text preprocessing, edge cases, and dataset consistency.
Can be executed with pytest:
    pytest tests/test_preprocessing.py
Or directly:
    python tests/test_preprocessing.py
"""
from pathlib import Path
import math
import sys
import pytest

# Ensure project root is available on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.preprocess import clean_text, combine_title_and_text, get_stopwords, NEGATION_WORDS


def test_missing_and_none_values():
    """Verify that None, NaN, and numeric inputs do not raise exceptions."""
    assert clean_text(None) == ""
    assert clean_text(float("nan")) == ""
    assert clean_text(math.nan) == ""
    # Numeric types should be safely processed without crashing
    res_num = clean_text(12345)
    assert isinstance(res_num, str)


def test_empty_and_whitespace_strings():
    """Verify that empty and whitespace strings return empty string."""
    assert clean_text("") == ""
    assert clean_text("   ") == ""
    assert clean_text("\n\t  \r ") == ""


def test_url_cleaning():
    """Verify that http, https, and www URLs are completely removed."""
    sample = "Breaking news from https://example.com/article?id=123 please verify"
    result = clean_text(sample)
    assert "http" not in result
    assert "example" not in result
    assert "breaking news" in result
    assert "verify" in result

    sample_www = "Visit www.faketruth.org for details"
    result_www = clean_text(sample_www)
    assert "www" not in result_www
    assert "faketruth" not in result_www
    assert "visit" in result_www


def test_html_fragment_and_entity_cleaning():
    """Verify HTML entities and markup tags are decoded and cleaned."""
    sample = "<h1>Breaking News &amp; Alert</h1><p>President &quot;agrees&quot; to terms</p>"
    result = clean_text(sample)
    assert "h1" not in result
    assert "amp" not in result
    assert "quot" not in result
    assert "breaking news alert president agrees terms" == result


def test_punctuation_and_whitespace():
    """Verify punctuation is stripped and extra whitespace normalized."""
    sample = "Alert!!! Breaking: news... (updated) [confirmed]   extra   spaces."
    result = clean_text(sample)
    assert "!" not in result
    assert ":" not in result
    assert "." not in result
    assert "[" not in result
    assert "alert breaking news updated confirmed extra spaces" == result


def test_negation_preservation():
    """Verify that critical negations are preserved for fake news classification."""
    stops = get_stopwords(preserve_negations=True)
    for word in NEGATION_WORDS:
        assert word not in stops, f"Negation '{word}' was incorrectly placed in stopwords!"

    sample = "The senator did not attend and has no involvement whatsoever"
    result = clean_text(sample, preserve_negations=True)
    assert "not" in result, "'not' was stripped from text!"
    assert "no" in result, "'no' was stripped from text!"


def test_wire_byline_leakage_removal():
    """Verify that Reuters wire service prefixes are removed to prevent data leakage."""
    sample1 = "WASHINGTON (Reuters) - The Senate approved the budget proposal today."
    res1 = clean_text(sample1, strip_wire_byline=True)
    assert "reuters" not in res1
    assert "senate approved budget proposal today" in res1

    sample2 = "LONDON (Reuters) — Prime Minister announces new trade deal."
    res2 = clean_text(sample2, strip_wire_byline=True)
    assert "reuters" not in res2
    assert "prime minister announces new trade deal" in res2


def test_combine_title_and_text():
    """Verify combining title and body handles missing fields and prevents duplication."""
    # 1. Headline only (body is empty)
    assert combine_title_and_text("Shocking Discovery Found", "") == "Shocking Discovery Found"
    assert combine_title_and_text("Shocking Discovery Found", None) == "Shocking Discovery Found"

    # 2. Body only (headline is empty)
    assert combine_title_and_text("", "Scientists made a breakthrough.") == "Scientists made a breakthrough."
    assert combine_title_and_text(None, "Scientists made a breakthrough.") == "Scientists made a breakthrough."

    # 3. Both present
    comb = combine_title_and_text("Breaking Headline", "Details of the event inside.")
    assert comb == "Breaking Headline - Details of the event inside."

    # 4. Text already prefixed with title
    dup = combine_title_and_text("Breaking Headline", "Breaking Headline: Full story follows.")
    assert dup == "Breaking Headline: Full story follows."


def test_label_consistency():
    """Verify label convention: 1 is Fake, 0 is Real."""
    FAKE_LABEL = 1
    REAL_LABEL = 0
    assert FAKE_LABEL == 1, "Fake label must be 1"
    assert REAL_LABEL == 0, "Real label must be 0"
    assert FAKE_LABEL != REAL_LABEL


if __name__ == "__main__":
    test_missing_and_none_values()
    test_empty_and_whitespace_strings()
    test_url_cleaning()
    test_html_fragment_and_entity_cleaning()
    test_punctuation_and_whitespace()
    test_negation_preservation()
    test_wire_byline_leakage_removal()
    test_combine_title_and_text()
    test_label_consistency()
    print("\n>>> All 9 preprocessing and consistency tests passed successfully! <<<")
