"""
News fetcher module for securely retrieving live news headlines via NewsAPI.
Credentials are read from environment variables or a local .env file.
Raw responses are validated, normalized, and can optionally be exported without
overwriting training datasets.
"""
from pathlib import Path
import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional
import requests

# Ensure project root is available on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT_DIR / ".env")
except ImportError:
    pass

import config

# Protected training datasets that must never be overwritten
FORBIDDEN_OUTPUT_FILES = {
    (config.DATA_DIR / "Fake.csv").resolve(),
    (config.DATA_DIR / "True.csv").resolve(),
    (config.DATA_DIR / "processed_news.csv").resolve(),
}


class NewsAPIError(Exception):
    """Base exception for NewsAPI client errors."""
    pass


class MissingAPIKeyError(NewsAPIError):
    """Raised when NEWS_API_KEY is missing or unconfigured."""
    pass


class NewsAPIAuthError(NewsAPIError):
    """Raised when NewsAPI credentials are invalid (HTTP 401)."""
    pass


class NewsAPIRateLimitError(NewsAPIError):
    """Raised when NewsAPI rate limit or quota is exceeded (HTTP 429)."""
    pass


def get_api_key(api_key: Optional[str] = None) -> str:
    """
    Resolve API key from parameter, environment variable, or config.
    Raises MissingAPIKeyError if missing or set to placeholder.
    """
    key = api_key or os.getenv("NEWS_API_KEY") or getattr(config, "NEWS_API_KEY", "")
    key = str(key).strip()
    if not key or key == "your_news_api_key_here":
        raise MissingAPIKeyError(
            "NEWS_API_KEY is not configured.\n"
            "To fix this, create a local .env file in the project root:\n"
            "    Copy-Item .env.example .env\n"
            "And set your API key:\n"
            "    NEWS_API_KEY=your_actual_key"
        )
    return key


def fetch_top_headlines(
    country: str = "us",
    category: Optional[str] = None,
    query: Optional[str] = None,
    page_size: int = 10,
    api_key: Optional[str] = None,
    timeout: float = 10.0,
) -> List[Dict[str, str]]:
    """
    Fetch and normalize top headlines from NewsAPI.
    
    Args:
        country: Two-letter ISO 3166-1 country code (default 'us').
        category: News category (e.g. 'general', 'technology', 'business').
        query: Optional search keyword filter.
        page_size: Number of articles to retrieve (1 to 100).
        api_key: Optional explicit API key; defaults to NEWS_API_KEY environment variable.
        timeout: Network request timeout in seconds (default 10.0s).
        
    Returns:
        Clean list of article dictionaries containing:
        'title', 'description', 'content', 'source', 'url', 'published_at'.
        
    Raises:
        MissingAPIKeyError: If no API key is configured.
        NewsAPIAuthError: If the API key is rejected by NewsAPI.
        NewsAPIRateLimitError: If API rate limits are hit.
        NewsAPIError: On other network or HTTP errors.
    """
    key = get_api_key(api_key)

    url = "https://newsapi.org/v2/top-headlines"
    params: Dict[str, Any] = {
        "country": country,
        "pageSize": min(max(page_size, 1), 100),
        "apiKey": key,
    }
    if category:
        params["category"] = category
    if query:
        params["q"] = query

    try:
        response = requests.get(url, params=params, timeout=timeout)
    except requests.exceptions.Timeout as exc:
        raise NewsAPIError(f"Request to NewsAPI timed out after {timeout} seconds.") from exc
    except requests.exceptions.ConnectionError as exc:
        raise NewsAPIError("Network connection failure while connecting to NewsAPI.") from exc
    except requests.exceptions.RequestException as exc:
        raise NewsAPIError(f"HTTP request error: {exc}") from exc

    # Handle HTTP status codes
    if response.status_code == 401:
        raise NewsAPIAuthError(
            "Invalid NewsAPI credentials (HTTP 401). Please verify your NEWS_API_KEY in .env."
        )
    if response.status_code == 429:
        raise NewsAPIRateLimitError(
            "NewsAPI rate limit or quota exceeded (HTTP 429). Please wait before requesting again."
        )
    if response.status_code >= 400:
        raise NewsAPIError(
            f"NewsAPI error (HTTP {response.status_code}): {response.text}"
        )

    # Validate JSON payload
    try:
        data = response.json()
    except Exception as exc:
        raise NewsAPIError(f"Malformed JSON response from NewsAPI: {exc}") from exc

    if data.get("status") == "error":
        code = data.get("code", "")
        message = data.get("message", "Unknown error from NewsAPI")
        if code == "apiKeyInvalid":
            raise NewsAPIAuthError(f"Invalid API key: {message}")
        if code == "rateLimited":
            raise NewsAPIRateLimitError(f"Rate limited: {message}")
        raise NewsAPIError(f"NewsAPI error ({code}): {message}")

    raw_articles = data.get("articles", [])
    clean_articles: List[Dict[str, str]] = []

    for art in raw_articles:
        if not isinstance(art, dict):
            continue

        raw_title = art.get("title")
        raw_desc = art.get("description")
        raw_content = art.get("content")

        title = str(raw_title).strip() if raw_title is not None else ""
        desc = str(raw_desc).strip() if raw_desc is not None else ""
        content = str(raw_content).strip() if raw_content is not None else ""

        # Skip articles where both title and description are missing
        if not title and not desc:
            continue

        source_obj = art.get("source")
        source_name = (
            str(source_obj.get("name", "")).strip()
            if isinstance(source_obj, dict) and source_obj.get("name")
            else "Unknown Source"
        )

        clean_articles.append({
            "title": title or "Untitled Article",
            "description": desc,
            "content": content,
            "source": source_name,
            "url": str(art.get("url") or "").strip(),
            "published_at": str(art.get("publishedAt") or "").strip(),
        })

    return clean_articles


def save_articles(articles: List[Dict[str, str]], output_path: Path) -> Path:
    """
    Safely export fetched articles to JSON or CSV.
    Guards against overwriting training datasets.
    """
    resolved_path = output_path.resolve()
    if resolved_path in FORBIDDEN_OUTPUT_FILES:
        raise ValueError(
            f"Refusing to overwrite core training dataset at: {output_path}!\n"
            "Please specify a separate output file (e.g. data/fetched_news.json)."
        )

    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    if resolved_path.suffix.lower() == ".csv":
        import pandas as pd
        df = pd.DataFrame(articles)
        df.to_csv(resolved_path, index=False)
    else:
        # Default to JSON
        with open(resolved_path, "w", encoding="utf-8") as f:
            json.dump(articles, f, indent=2, ensure_ascii=False)

    return resolved_path


def main():
    parser = argparse.ArgumentParser(description="Fetch top headlines from NewsAPI safely.")
    parser.add_argument("--country", "-c", default="us", help="2-letter ISO country code (default 'us')")
    parser.add_argument("--category", help="Category: business, entertainment, general, health, science, sports, technology")
    parser.add_argument("--query", "-q", help="Search keyword query")
    parser.add_argument("--limit", "-n", type=int, default=10, help="Number of articles to fetch (default: 10)")
    parser.add_argument("--output", "-o", type=Path, help="Optional output path to save articles (.json or .csv)")
    parser.add_argument("--key", help="Explicit NewsAPI key (overrides .env)")
    args = parser.parse_args()

    try:
        articles = fetch_top_headlines(
            country=args.country,
            category=args.category,
            query=args.query,
            page_size=args.limit,
            api_key=args.key,
        )
        print(f"Successfully fetched {len(articles)} articles.\n")
        for idx, art in enumerate(articles, 1):
            print(f"[{idx}] {art['title']}")
            print(f"    Source: {art['source']} | Published: {art['published_at']}")
            if art['description']:
                print(f"    Summary: {art['description'][:120]}...")
            print()

        if args.output:
            saved_to = save_articles(articles, args.output)
            print(f"[OK] Saved {len(articles)} articles to: {saved_to}")

    except MissingAPIKeyError as err:
        print(f"\n[Configuration Error]\n{err}", file=sys.stderr)
        sys.exit(1)
    except NewsAPIAuthError as err:
        print(f"\n[Authentication Error]\n{err}", file=sys.stderr)
        sys.exit(1)
    except NewsAPIRateLimitError as err:
        print(f"\n[Rate Limit Error]\n{err}", file=sys.stderr)
        sys.exit(1)
    except NewsAPIError as err:
        print(f"\n[NewsAPI Error]\n{err}", file=sys.stderr)
        sys.exit(1)
    except Exception as err:
        print(f"\n[Error]\n{err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()