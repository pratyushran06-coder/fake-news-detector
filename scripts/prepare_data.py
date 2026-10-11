"""
Dataset preparation script for Fake News Detector.
Loads raw Fake.csv and True.csv relative to project root, combines headlines and text,
removes unusable rows and duplicates, and saves a clean, leakage-free dataset.

Usage:
    python scripts/prepare_data.py [--output data/processed_news.csv]
"""
from pathlib import Path
import argparse
import sys
import pandas as pd

# Ensure project root is available on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config
from utils.preprocess import clean_text, combine_title_and_text


def prepare_dataset(
    fake_path: Path = config.FAKE_CSV_PATH,
    true_path: Path = config.TRUE_CSV_PATH,
    output_path: Path = config.PROCESSED_CSV_PATH,
    strip_wire_byline: bool = True,
) -> pd.DataFrame:
    """
    Load, clean, normalize, and deduplicate the fake news dataset.
    
    Args:
        fake_path: Path to raw Fake.csv.
        true_path: Path to raw True.csv.
        output_path: Destination path for cleaned output CSV (does not overwrite raw inputs).
        strip_wire_byline: Whether to strip Reuters wire markers to eliminate publisher leakage.
        
    Returns:
        pd.DataFrame containing the processed, deduplicated dataset.
    """
    print("=" * 65)
    print("      Fake News Detector - Dataset Preparation Pipeline")
    print("=" * 65)

    if not fake_path.exists():
        raise FileNotFoundError(f"Fake news CSV not found at: {fake_path}")
    if not true_path.exists():
        raise FileNotFoundError(f"True news CSV not found at: {true_path}")

    # 1. Load raw datasets
    print(f"\n[1/6] Loading raw datasets...")
    print(f"      - Loading Fake news from: {fake_path}")
    df_fake = pd.read_csv(fake_path)
    fake_loaded = len(df_fake)
    print(f"        -> Loaded {fake_loaded:,} fake news articles.")

    print(f"      - Loading Real news from: {true_path}")
    df_true = pd.read_csv(true_path)
    true_loaded = len(df_true)
    print(f"        -> Loaded {true_loaded:,} real news articles.")

    total_loaded = fake_loaded + true_loaded
    print(f"      - Total articles loaded: {total_loaded:,}")

    # 2. Consistent Label Assignment (1 = Fake, 0 = Real)
    print("\n[2/6] Assigning consistent labels (1 = Fake, 0 = Real)...")
    df_fake["label"] = 1
    df_true["label"] = 0

    # 3. Concatenate
    df = pd.concat([df_fake, df_true], ignore_index=True)

    # 4. Combine Title and Article Text
    print("\n[3/6] Combining title and body text...")
    df["combined_text"] = df.apply(
        lambda row: combine_title_and_text(row.get("title"), row.get("text")),
        axis=1,
    )

    # Count how many articles had empty text but were saved by title
    text_empty_but_has_title = (
        df["text"].fillna("").str.strip().eq("") & df["title"].fillna("").str.strip().ne("")
    ).sum()
    print(f"      - Articles with missing body recovered via headline: {text_empty_but_has_title:,}")

    # 5. Clean and Normalize Text
    print(f"\n[4/6] Cleaning text (wire byline stripping={strip_wire_byline}, negations preserved)...")
    df["clean_text"] = df["combined_text"].apply(
        lambda t: clean_text(t, strip_wire_byline=strip_wire_byline, preserve_negations=True)
    )

    # 6. Filter Unusable Rows & Deduplicate
    print("\n[5/6] Filtering unusable records and removing duplicates...")
    # Unusable: empty clean_text
    empty_mask = df["clean_text"].str.strip().eq("")
    unusable_count = int(empty_mask.sum())
    df_usable = df[~empty_mask].copy()

    # Deduplication on clean_text
    clean_dupes_count = int(df_usable.duplicated(subset=["clean_text"]).sum())
    df_cleaned = df_usable.drop_duplicates(subset=["clean_text"]).copy()

    total_removed = unusable_count + clean_dupes_count
    retained_count = len(df_cleaned)

    # 7. Save Cleaned Dataset
    print(f"\n[6/6] Saving cleaned dataset to: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Ensure raw files are never overwritten
    if output_path.resolve() in (fake_path.resolve(), true_path.resolve()):
        raise ValueError("Output path cannot be the same as raw Fake.csv or True.csv!")

    df_cleaned.to_csv(output_path, index=False)
    print(f"      [OK] Saved {retained_count:,} records successfully.")

    # 8. Print Summary Statistics
    label_counts = df_cleaned["label"].value_counts().to_dict()
    fake_retained = label_counts.get(1, 0)
    true_retained = label_counts.get(0, 0)

    print("\n" + "=" * 65)
    print("                    PREPARATION SUMMARY")
    print("=" * 65)
    print(f"  Total raw records loaded:       {total_loaded:,}")
    print(f"    - Fake news (raw):             {fake_loaded:,}")
    print(f"    - Real news (raw):             {true_loaded:,}")
    print(f"  Records recovered via headline: {text_empty_but_has_title:,}")
    print(f"  Records removed:")
    print(f"    - Empty / Unusable:            {unusable_count:,}")
    print(f"    - Duplicate articles:          {clean_dupes_count:,}")
    print(f"    - Total removed:               {total_removed:,}")
    print(f"  Final records retained:         {retained_count:,}")
    print(f"  Final Label Distribution:")
    print(f"    - Real News (label 0):         {true_retained:,} ({true_retained / retained_count:.1%})")
    print(f"    - Fake News (label 1):         {fake_retained:,} ({fake_retained / retained_count:.1%})")
    print("=" * 65)

    return df_cleaned


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare and clean Fake News dataset.")
    parser.add_argument(
        "--output",
        type=Path,
        default=config.PROCESSED_CSV_PATH,
        help="Path for saving cleaned dataset (default: data/processed_news.csv)",
    )
    parser.add_argument(
        "--keep-wire-byline",
        action="store_true",
        help="Disable Reuters wire byline removal (not recommended).",
    )
    args = parser.parse_args()

    prepare_dataset(
        fake_path=config.FAKE_CSV_PATH,
        true_path=config.TRUE_CSV_PATH,
        output_path=args.output,
        strip_wire_byline=not args.keep_wire_byline,
    )
