"""
Tests for dataset loading, label consistency, duplicate removal, and data leakage prevention.
Uses synthetic DataFrames and temporary CSVs so tests run instantaneously without large file I/O.
"""
from pathlib import Path
import sys
import pandas as pd
import pytest

# Ensure project root is available on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.train_bert import load_and_prepare_data, split_dataset
from utils.preprocess import combine_title_and_text


def test_label_mapping_constants():
    """Verify standard project label mapping convention: 0 = Real News, 1 = Fake News."""
    from models.predict import LABEL_MAPPING, REVERSE_LABEL_MAPPING

    assert LABEL_MAPPING[0] == "Real News"
    assert LABEL_MAPPING[1] == "Fake News"
    assert REVERSE_LABEL_MAPPING["Real News"] == 0
    assert REVERSE_LABEL_MAPPING["Fake News"] == 1


def test_dataset_loading_and_labels(tmp_path):
    """Loading raw datasets must consistently assign label 1 to Fake and label 0 to Real."""
    # Create synthetic raw CSVs
    fake_csv = tmp_path / "Fake.csv"
    true_csv = tmp_path / "True.csv"

    pd.DataFrame({
        "title": ["Fake Headline Alpha", "Fake Headline Beta"],
        "text": ["Fake body content alpha", "Fake body content beta"],
        "subject": ["News", "News"],
        "date": ["Jan 1", "Jan 2"],
    }).to_csv(fake_csv, index=False)

    pd.DataFrame({
        "title": ["Real Headline Gamma", "Real Headline Delta"],
        "text": ["Real body content gamma", "Real body content delta"],
        "subject": ["politicsNews", "politicsNews"],
        "date": ["Jan 1", "Jan 2"],
    }).to_csv(true_csv, index=False)

    df = load_and_prepare_data(fake_path=fake_csv, true_path=true_csv)

    assert "label" in df.columns
    assert "clean_text" in df.columns
    assert len(df) == 4
    # Real = 0, Fake = 1
    assert set(df[df["clean_text"].str.contains("fake")]["label"]) == {1}
    assert set(df[df["clean_text"].str.contains("real")]["label"]) == {0}


def test_headline_recovery_for_missing_body(tmp_path):
    """Articles with empty body must be preserved using their headline."""
    fake_csv = tmp_path / "Fake.csv"
    true_csv = tmp_path / "True.csv"

    pd.DataFrame({
        "title": ["Important Headline Without Body"],
        "text": ["   "],  # whitespace-only body
        "subject": ["News"],
        "date": ["Jan 1"],
    }).to_csv(fake_csv, index=False)

    pd.DataFrame({
        "title": ["Real Headline With Body"],
        "text": ["Real article body content."],
        "subject": ["worldnews"],
        "date": ["Jan 1"],
    }).to_csv(true_csv, index=False)

    df = load_and_prepare_data(fake_path=fake_csv, true_path=true_csv)

    assert len(df) == 2
    # Verify the headline-only article was retained
    retained_texts = df["clean_text"].tolist()
    assert any("important headline without body" in t for t in retained_texts)


def test_duplicate_removal_before_splitting(tmp_path):
    """Duplicate clean_text records must be purged before dataset splitting."""
    fake_csv = tmp_path / "Fake.csv"
    true_csv = tmp_path / "True.csv"

    # Insert 3 identical articles and 1 unique
    pd.DataFrame({
        "title": ["Duplicate Story", "Duplicate Story", "Unique Fake"],
        "text": ["Identical body text", "Identical body text", "Different body"],
        "subject": ["News", "News", "News"],
        "date": ["Jan 1", "Jan 1", "Jan 1"],
    }).to_csv(fake_csv, index=False)

    pd.DataFrame({
        "title": ["Real Story 1"],
        "text": ["Real body 1"],
        "subject": ["politicsNews"],
        "date": ["Jan 1"],
    }).to_csv(true_csv, index=False)

    df = load_and_prepare_data(fake_path=fake_csv, true_path=true_csv)

    # 3 fake articles had 1 duplicate pair -> should result in 2 fake + 1 real = 3 total
    assert len(df) == 3
    assert df["clean_text"].duplicated().sum() == 0


def test_data_leakage_prevention_disjoint_splits():
    """Verify that Train, Validation, and Test sets share zero overlapping texts (no data leakage)."""
    # Create synthetic dataset with 50 distinct samples
    data = []
    for i in range(50):
        data.append({"clean_text": f"unique news article content number {i}", "label": i % 2})
    df = pd.DataFrame(data)

    train_df, val_df, test_df = split_dataset(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42)

    train_texts = set(train_df["clean_text"])
    val_texts = set(val_df["clean_text"])
    test_texts = set(test_df["clean_text"])

    # Verify complete independence of partitions (no intersection)
    assert len(train_texts.intersection(val_texts)) == 0, "Train and Validation sets have overlapping samples!"
    assert len(train_texts.intersection(test_texts)) == 0, "Train and Test sets have overlapping samples!"
    assert len(val_texts.intersection(test_texts)) == 0, "Validation and Test sets have overlapping samples!"

    # Verify total partition coverage
    assert len(train_df) + len(val_df) + len(test_df) == len(df)


def test_stratified_split_preserves_class_balance():
    """Stratified splitting must preserve the Real/Fake class distribution across all splits."""
    # 60% Fake (label 1), 40% Real (label 0)
    data = []
    for i in range(60):
        data.append({"clean_text": f"fake article {i}", "label": 1})
    for i in range(40):
        data.append({"clean_text": f"real article {i}", "label": 0})
    df = pd.DataFrame(data)

    train_df, val_df, test_df = split_dataset(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42)

    # In train set: 70 samples -> should be 42 fake (60%) and 28 real (40%)
    train_fake_ratio = (train_df["label"] == 1).mean()
    val_fake_ratio = (val_df["label"] == 1).mean()
    test_fake_ratio = (test_df["label"] == 1).mean()

    assert abs(train_fake_ratio - 0.60) <= 0.05
    assert abs(val_fake_ratio - 0.60) <= 0.10
    assert abs(test_fake_ratio - 0.60) <= 0.10
