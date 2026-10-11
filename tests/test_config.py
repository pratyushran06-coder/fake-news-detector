"""
Tests for configuration settings and project-relative file paths.
"""
from pathlib import Path
import sys
import pytest

# Ensure project root is available on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config


def test_root_directory_and_structure():
    """Verify that ROOT_DIR exists and contains standard project packages."""
    assert config.ROOT_DIR.exists(), f"ROOT_DIR does not exist: {config.ROOT_DIR}"
    assert config.ROOT_DIR.is_dir()

    expected_dirs = ["backend", "frontend", "models", "scraper", "utils", "data", "tests"]
    for d in expected_dirs:
        sub = config.ROOT_DIR / d
        assert sub.exists(), f"Expected directory '{d}' missing from project root!"
        assert sub.is_dir()


def test_dataset_paths():
    """Verify that dataset path attributes resolve to the expected files."""
    assert config.DATA_DIR == config.ROOT_DIR / "data"
    assert config.FAKE_CSV_PATH == config.DATA_DIR / "Fake.csv"
    assert config.TRUE_CSV_PATH == config.DATA_DIR / "True.csv"
    assert config.PROCESSED_CSV_PATH == config.DATA_DIR / "processed_news.csv"

    # Datasets should physically exist on disk
    assert config.FAKE_CSV_PATH.exists()
    assert config.TRUE_CSV_PATH.exists()


def test_server_and_api_defaults():
    """Verify that server configurations have valid default values and formats."""
    assert isinstance(config.BACKEND_HOST, str)
    assert isinstance(config.BACKEND_PORT, int)
    assert config.BACKEND_PORT > 0
    assert config.BACKEND_URL.startswith("http://") or config.BACKEND_URL.startswith("https://")
    assert str(config.BACKEND_PORT) in config.BACKEND_URL


def test_sys_path_contains_root():
    """Verify that the repository root is present in sys.path for portable imports."""
    assert str(config.ROOT_DIR) in sys.path
