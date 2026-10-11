"""
Unit tests for scraper/news_fetcher.py using mocked HTTP responses.
Verifies error handling, credential isolation, and parsing without calling live APIs.
"""
from pathlib import Path
import json
import sys
from unittest.mock import MagicMock, patch
import pytest
import requests

# Ensure project root is available on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scraper.news_fetcher import (
    MissingAPIKeyError,
    NewsAPIAuthError,
    NewsAPIError,
    NewsAPIRateLimitError,
    fetch_top_headlines,
    save_articles,
)


def test_missing_api_key_raises_error(monkeypatch):
    """Missing or placeholder API key must raise MissingAPIKeyError."""
    monkeypatch.delenv("NEWS_API_KEY", raising=False)
    with pytest.raises(MissingAPIKeyError) as exc_info:
        fetch_top_headlines(api_key=None)
    assert "NEWS_API_KEY is not configured" in str(exc_info.value)

    # Placeholder string should also be rejected
    with pytest.raises(MissingAPIKeyError):
        fetch_top_headlines(api_key="your_news_api_key_here")


@patch("requests.get")
def test_fetch_top_headlines_success(mock_get):
    """Successful API response must return clean, normalized list of dictionaries."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "status": "ok",
        "totalResults": 2,
        "articles": [
            {
                "source": {"id": "reuters", "name": "Reuters"},
                "author": "Jane Doe",
                "title": "Global Summit Concludes with Landmark Climate Deal",
                "description": "Delegates from 190 nations reached a consensus.",
                "url": "https://example.com/climate-deal",
                "publishedAt": "2026-10-11T12:00:00Z",
                "content": "Full article text goes here...",
            },
            {
                "source": {"id": None, "name": "Tech Daily"},
                "author": None,
                "title": "New Semiconductor Breakthrough Announced",
                "description": "Engineers develop high-efficiency microchips.",
                "url": "https://example.com/chips",
                "publishedAt": "2026-10-11T11:30:00Z",
                "content": "Chip content...",
            },
        ],
    }
    mock_get.return_value = mock_resp

    articles = fetch_top_headlines(api_key="valid_test_key", page_size=2)
    assert len(articles) == 2

    first = articles[0]
    assert first["title"] == "Global Summit Concludes with Landmark Climate Deal"
    assert first["source"] == "Reuters"
    assert first["description"] == "Delegates from 190 nations reached a consensus."
    assert first["url"] == "https://example.com/climate-deal"
    assert first["published_at"] == "2026-10-11T12:00:00Z"


@patch("requests.get")
def test_missing_title_and_description_handling(mock_get):
    """Articles with missing titles or null values must be handled safely without crashing."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "status": "ok",
        "articles": [
            # Missing title but has description
            {
                "source": {"name": None},
                "title": None,
                "description": "Only a summary is available for this event.",
                "url": "https://example.com/summary",
                "publishedAt": None,
            },
            # Missing both title and description -> should be skipped
            {
                "source": {"name": "Empty News"},
                "title": None,
                "description": None,
                "url": "https://example.com/empty",
            },
        ],
    }
    mock_get.return_value = mock_resp

    articles = fetch_top_headlines(api_key="valid_test_key")
    assert len(articles) == 1
    assert articles[0]["title"] == "Untitled Article"
    assert articles[0]["source"] == "Unknown Source"
    assert articles[0]["description"] == "Only a summary is available for this event."


@patch("requests.get")
def test_auth_error_401(mock_get):
    """HTTP 401 must raise NewsAPIAuthError."""
    mock_resp = MagicMock()
    mock_resp.status_code = 401
    mock_resp.text = "Unauthorized"
    mock_get.return_value = mock_resp

    with pytest.raises(NewsAPIAuthError) as exc_info:
        fetch_top_headlines(api_key="invalid_key")
    assert "Invalid NewsAPI credentials" in str(exc_info.value)


@patch("requests.get")
def test_rate_limit_429(mock_get):
    """HTTP 429 must raise NewsAPIRateLimitError."""
    mock_resp = MagicMock()
    mock_resp.status_code = 429
    mock_resp.text = "Too Many Requests"
    mock_get.return_value = mock_resp

    with pytest.raises(NewsAPIRateLimitError) as exc_info:
        fetch_top_headlines(api_key="test_key")
    assert "rate limit or quota exceeded" in str(exc_info.value)


@patch("requests.get")
def test_timeout_and_connection_error(mock_get):
    """Network timeouts and connection drops must raise descriptive NewsAPIErrors."""
    # Timeout
    mock_get.side_effect = requests.exceptions.Timeout("Read timeout")
    with pytest.raises(NewsAPIError) as exc_timeout:
        fetch_top_headlines(api_key="test_key")
    assert "timed out" in str(exc_timeout.value)

    # Connection Error
    mock_get.side_effect = requests.exceptions.ConnectionError("Connection dropped")
    with pytest.raises(NewsAPIError) as exc_conn:
        fetch_top_headlines(api_key="test_key")
    assert "Network connection failure" in str(exc_conn.value)


def test_save_articles_to_file(tmp_path):
    """Saving articles to separate JSON file works cleanly."""
    sample_articles = [
        {"title": "Sample 1", "description": "Desc 1", "source": "Source 1", "url": "", "published_at": "", "content": ""}
    ]
    out_file = tmp_path / "fetched.json"
    saved_path = save_articles(sample_articles, out_file)
    assert saved_path.exists()

    with open(saved_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert len(loaded) == 1
    assert loaded[0]["title"] == "Sample 1"


def test_save_articles_protects_core_datasets():
    """Attempting to overwrite Fake.csv or True.csv must raise ValueError."""
    fake_csv = Path(ROOT_DIR / "data" / "Fake.csv")
    true_csv = Path(ROOT_DIR / "data" / "True.csv")

    with pytest.raises(ValueError) as exc1:
        save_articles([], fake_csv)
    assert "Refusing to overwrite core training dataset" in str(exc1.value)

    with pytest.raises(ValueError) as exc2:
        save_articles([], true_csv)
    assert "Refusing to overwrite core training dataset" in str(exc2.value)
