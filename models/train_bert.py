"""
BERT / DistilBERT Sequence Classification Training Pipeline for Fake News Detection.
Reproducible training, stratified evaluation, and model artifact serialization.

Usage:
    # 1. DistilBERT training on 3,000 balanced articles (CPU-friendly, ~10-15 mins)
    python models/train_bert.py --model-name distilbert-base-uncased --sample-size 3000 --epochs 2 --batch-size 8 --output-dir saved_models/distilbert_fake_news

    # 2. Fast smoke test (validates pipeline end-to-end on 40 samples)
    python models/train_bert.py --smoke-test
"""
from pathlib import Path
import argparse
import json
import os
import random
import sys
from typing import Any, Dict, Optional, Tuple

# Ensure project root is available on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Suppress Hugging Face symlinks warning on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset

import config
from utils.preprocess import clean_text, combine_title_and_text

# Standard Project Label Mapping (0 = Real News, 1 = Fake News)
LABEL_MAPPING: Dict[int, str] = {0: "Real News", 1: "Fake News"}
REVERSE_LABEL_MAPPING: Dict[str, int] = {"Real News": 0, "Fake News": 1}


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_and_prepare_data(
    fake_path: Optional[Path] = None,
    true_path: Optional[Path] = None,
    processed_path: Optional[Path] = None,
    sample_size: Optional[int] = None,
    smoke_test: bool = False,
    smoke_samples: int = 80,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Load datasets, validate labels, combine headlines and bodies, and clean text.
    Removes duplicates before dataset splitting to prevent data leakage.
    Optionally samples a balanced stratified subset.
    """
    # 1. Load data
    # When explicit raw file paths are provided, load and process those files directly
    if fake_path is not None or true_path is not None:
        effective_fake = Path(fake_path) if fake_path is not None else config.FAKE_CSV_PATH
        effective_true = Path(true_path) if true_path is not None else config.TRUE_CSV_PATH
        print(f"Loading raw datasets:")
        print(f"  - Fake news: {effective_fake}")
        print(f"  - Real news: {effective_true}")
        if not effective_fake.exists():
            raise FileNotFoundError(f"Raw Fake news CSV not found at: {effective_fake}")
        if not effective_true.exists():
            raise FileNotFoundError(f"Raw Real news CSV not found at: {effective_true}")

        df_fake = pd.read_csv(effective_fake)
        df_true = pd.read_csv(effective_true)

        # Consistent label assignment (1 = Fake, 0 = Real)
        df_fake["label"] = 1
        df_true["label"] = 0

        df = pd.concat([df_fake, df_true], ignore_index=True)

        # Combine title and text to avoid discarding headline-only rows
        df["combined_text"] = df.apply(
            lambda r: combine_title_and_text(r.get("title"), r.get("text")),
            axis=1,
        )

        # Clean text and strip wire bylines (e.g. Reuters) to eliminate data leakage
        df["clean_text"] = df["combined_text"].apply(
            lambda t: clean_text(t, strip_wire_byline=True, preserve_negations=True)
        )
    else:
        # When no explicit raw file paths are provided, use processed dataset if available
        target_processed = processed_path or (config.PROCESSED_CSV_PATH if config.PROCESSED_CSV_PATH.exists() else None)
        if target_processed and Path(target_processed).exists():
            print(f"Loading cleaned dataset from: {target_processed}")
            df = pd.read_csv(target_processed)
            if "clean_text" not in df.columns or "label" not in df.columns:
                raise ValueError(f"Dataset at {target_processed} must contain 'clean_text' and 'label' columns.")
        else:
            effective_fake = config.FAKE_CSV_PATH
            effective_true = config.TRUE_CSV_PATH
            print(f"Loading raw datasets relative to project root:")
            print(f"  - Fake news: {effective_fake}")
            print(f"  - Real news: {effective_true}")
            if not effective_fake.exists():
                raise FileNotFoundError(f"Raw Fake news CSV not found at: {effective_fake}")
            if not effective_true.exists():
                raise FileNotFoundError(f"Raw Real news CSV not found at: {effective_true}")

            df_fake = pd.read_csv(effective_fake)
            df_true = pd.read_csv(effective_true)

            df_fake["label"] = 1
            df_true["label"] = 0

            df = pd.concat([df_fake, df_true], ignore_index=True)

            df["combined_text"] = df.apply(
                lambda r: combine_title_and_text(r.get("title"), r.get("text")),
                axis=1,
            )

            df["clean_text"] = df["combined_text"].apply(
                lambda t: clean_text(t, strip_wire_byline=True, preserve_negations=True)
            )

    # 2. Filter empty clean_text
    empty_mask = df["clean_text"].fillna("").str.strip().eq("")
    if empty_mask.sum() > 0:
        print(f"Dropping {empty_mask.sum():,} empty/unusable records...")
        df = df[~empty_mask].copy()

    # 3. Remove duplicates BEFORE splitting to avoid train/test contamination
    initial_len = len(df)
    df = df.drop_duplicates(subset=["clean_text"]).reset_index(drop=True)
    dupes_dropped = initial_len - len(df)
    if dupes_dropped > 0:
        print(f"Dropped {dupes_dropped:,} duplicate records before train/test splitting.")

    # 4. Handle smoke-test mode or stratified subset sampling
    if smoke_test:
        print(f"\n[Smoke-Test Mode] Sampling {smoke_samples} balanced rows for fast verification...")
        sample_per_class = max(10, smoke_samples // 2)
        df_real = df[df["label"] == 0].sample(n=min(sample_per_class, len(df[df["label"] == 0])), random_state=seed)
        df_fake = df[df["label"] == 1].sample(n=min(sample_per_class, len(df[df["label"] == 1])), random_state=seed)
        df = pd.concat([df_real, df_fake], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
        print(f"Smoke test dataset size: {len(df)} samples ({df['label'].value_counts().to_dict()})")

    elif sample_size is not None and sample_size < len(df):
        print(f"\n[Stratified Sampling] Extracting balanced subset of ~{sample_size:,} articles...")
        half = sample_size // 2
        avail_real = len(df[df["label"] == 0])
        avail_fake = len(df[df["label"] == 1])
        take_real = min(half, avail_real)
        take_fake = min(half, avail_fake)

        df_real = df[df["label"] == 0].sample(n=take_real, random_state=seed)
        df_fake = df[df["label"] == 1].sample(n=take_fake, random_state=seed)
        df = pd.concat([df_real, df_fake], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
        print(f"  - Sampled: {len(df):,} articles (Real: {take_real:,}, Fake: {take_fake:,})")

    return df[["clean_text", "label"]]


def split_dataset(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame into stratified Train, Validation, and Test sets.
    Enforces zero sample overlap across partitions.
    """
    total_ratio = train_ratio + val_ratio + test_ratio
    assert abs(total_ratio - 1.0) < 1e-4, "Split ratios must sum to 1.0"

    # Split train vs temp (val + test)
    temp_size = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_size,
        stratify=df["label"],
        random_state=seed,
    )

    # Split temp into val and test
    relative_test_size = test_ratio / temp_size
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_size,
        stratify=temp_df["label"],
        random_state=seed,
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Verify zero data leakage across partitions
    train_texts = set(train_df["clean_text"])
    val_texts = set(val_df["clean_text"])
    test_texts = set(test_df["clean_text"])

    assert len(train_texts.intersection(val_texts)) == 0, "Data leakage detected between train and val!"
    assert len(train_texts.intersection(test_texts)) == 0, "Data leakage detected between train and test!"
    assert len(val_texts.intersection(test_texts)) == 0, "Data leakage detected between val and test!"

    print(f"\nStratified Dataset Splits (Zero Leakage Verified):")
    print(f"  - Train Set:      {len(train_df):,} samples ({train_df['label'].value_counts().to_dict()})")
    print(f"  - Validation Set: {len(val_df):,} samples ({val_df['label'].value_counts().to_dict()})")
    print(f"  - Test Set:       {len(test_df):,} samples ({test_df['label'].value_counts().to_dict()})")

    return train_df, val_df, test_df


def compute_metrics(eval_pred) -> Dict[str, float]:
    """
    Compute evaluation metrics for Hugging Face Trainer evaluation.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, average="binary", zero_division=0)
    rec = recall_score(labels, preds, average="binary", zero_division=0)
    f1 = f1_score(labels, preds, average="binary", zero_division=0)
    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
    }


def train_bert_pipeline(
    model_name: str = "distilbert-base-uncased",
    output_dir: Path = config.ROOT_DIR / "saved_models" / "distilbert_fake_news",
    epochs: int = 2,
    batch_size: int = 8,
    max_length: int = 128,
    learning_rate: float = 2e-5,
    seed: int = 42,
    sample_size: Optional[int] = None,
    smoke_test: bool = False,
    smoke_samples: int = 80,
    data_path: Optional[Path] = None,
    fake_path: Optional[Path] = None,
    true_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Complete end-to-end training, validation, and testing pipeline.
    """
    print("=" * 70)
    print("   DistilBERT / BERT Training Pipeline for Fake News Detection")
    print("=" * 70)
    set_seed(seed)

    # Safety Guard: Never overwrite the existing working smoke test model
    smoke_path = (config.ROOT_DIR / "saved_models" / "smoke_test_model").resolve()
    if output_dir.resolve() == smoke_path and not smoke_test:
        raise ValueError(
            "SAFETY GUARD: Cannot overwrite 'saved_models/smoke_test_model' during production/subset training!\n"
            "Please specify a separate directory like 'saved_models/distilbert_fake_news'."
        )

    # 1. Device configuration & memory management
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[Environment Check]")
    print(f"  - Target device: {device.upper()}")
    if device == "cpu":
        print("  - Notice: Running on CPU. Training batch size set to CPU-friendly limits.")
    else:
        print(f"  - GPU: {torch.cuda.get_device_name(0)}")

    # 2. Data Loading & Deduplication
    print("\n[Step 1/5] Loading and preparing dataset...")
    df = load_and_prepare_data(
        fake_path=fake_path,
        true_path=true_path,
        processed_path=data_path,
        sample_size=sample_size,
        smoke_test=smoke_test,
        smoke_samples=smoke_samples,
        seed=seed,
    )

    # 3. Stratified Splitting
    print("\n[Step 2/5] Creating stratified train/validation/test splits...")
    train_df, val_df, test_df = split_dataset(df, seed=seed)

    # 4. Tokenizer & Model Initialization
    print(f"\n[Step 3/5] Initializing tokenizer & pretrained model: '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label=LABEL_MAPPING,
        label2id=REVERSE_LABEL_MAPPING,
    )

    # Tokenization wrapper
    def tokenize_fn(batch):
        return tokenizer(
            batch["clean_text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    print(f"  - Tokenizing with maximum sequence length: {max_length}")
    train_dataset = Dataset.from_pandas(train_df[["clean_text", "label"]]).map(tokenize_fn, batched=True)
    val_dataset = Dataset.from_pandas(val_df[["clean_text", "label"]]).map(tokenize_fn, batched=True)
    test_dataset = Dataset.from_pandas(test_df[["clean_text", "label"]]).map(tokenize_fn, batched=True)

    # 5. Training Arguments
    checkpoints_dir = output_dir / "checkpoints"
    effective_epochs = 1 if smoke_test else epochs
    effective_batch_size = 4 if smoke_test else batch_size

    training_args = TrainingArguments(
        output_dir=str(checkpoints_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=effective_batch_size,
        per_device_eval_batch_size=effective_batch_size,
        num_train_epochs=effective_epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=1,
        logging_steps=10 if smoke_test else 50,
        dataloader_num_workers=0,  # Safe across platforms/Windows
        report_to="none",
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # 6. Model Training
    print(f"\n[Step 4/5] Training model ({effective_epochs} epoch(s), batch_size={effective_batch_size})...")
    try:
        trainer.train()
    except (torch.cuda.OutOfMemoryError, MemoryError) as oom:
        print(f"\n[FATAL ERROR] Out of Memory: {oom}", file=sys.stderr)
        print("Suggestion: Try lowering --batch-size or --max-length.", file=sys.stderr)
        raise
    except Exception as exc:
        print(f"\n[FATAL ERROR] Training failed: {exc}", file=sys.stderr)
        raise

    # 7. Final Evaluation on Held-Out Test Set
    print("\n[Step 5/5] Evaluating on held-out test set...")
    test_predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(test_predictions.predictions, axis=1)
    y_true = test_predictions.label_ids

    acc = float(accuracy_score(y_true, y_pred))

    # Per-class metrics
    # Real News = Class 0
    prec_real = float(precision_score(y_true, y_pred, pos_label=0, zero_division=0))
    rec_real = float(recall_score(y_true, y_pred, pos_label=0, zero_division=0))
    f1_real = float(f1_score(y_true, y_pred, pos_label=0, zero_division=0))

    # Fake News = Class 1
    prec_fake = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    rec_fake = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1_fake = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))

    # Macro & Weighted metrics
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    cm = confusion_matrix(y_true, y_pred).tolist()
    clf_report = classification_report(
        y_true, y_pred, target_names=["Real News (0)", "Fake News (1)"], output_dict=True, zero_division=0
    )

    # Print Formatted Evaluation Report
    print("\n" + "=" * 70)
    print("                     TEST EVALUATION RESULTS")
    print("=" * 70)
    print(f"  Overall Accuracy:      {acc:.4f} ({acc * 100:.2f}%)")
    print(f"  Macro F1-Score:        {macro_f1:.4f}")
    print(f"  Weighted F1-Score:     {weighted_f1:.4f}")
    print("-" * 70)
    print(f"  Real News (Class 0):")
    print(f"    - Precision:         {prec_real:.4f}")
    print(f"    - Recall:            {rec_real:.4f}")
    print(f"    - F1-Score:          {f1_real:.4f}")
    print(f"  Fake News (Class 1):")
    print(f"    - Precision:         {prec_fake:.4f}")
    print(f"    - Recall:            {rec_fake:.4f}")
    print(f"    - F1-Score:          {f1_fake:.4f}")
    print("-" * 70)
    print("  Confusion Matrix:")
    print("                     Predicted REAL (0)   Predicted FAKE (1)")
    print(f"    Actual REAL (0):        {cm[0][0]:<12}         {cm[0][1]:<12}")
    print(f"    Actual FAKE (1):        {cm[1][0]:<12}         {cm[1][1]:<12}")
    print("=" * 70)

    # 8. Save Artifacts (Model, Tokenizer, Metrics, Label Mapping)
    print(f"\nSaving model and serialization artifacts to: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Save evaluation metrics
    metrics_payload = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "real_news": {
            "precision": prec_real,
            "recall": rec_real,
            "f1": f1_real,
            "support": int(np.sum(y_true == 0)),
        },
        "fake_news": {
            "precision": prec_fake,
            "recall": rec_fake,
            "f1": f1_fake,
            "support": int(np.sum(y_true == 1)),
        },
        "confusion_matrix": cm,
        "classification_report": clf_report,
        "training_config": {
            "model_name": model_name,
            "max_length": max_length,
            "epochs": effective_epochs,
            "batch_size": effective_batch_size,
            "learning_rate": learning_rate,
            "seed": seed,
            "sample_size": sample_size,
            "smoke_test": smoke_test,
        },
        "dataset_sizes": {
            "num_total_samples": len(df),
            "num_train_samples": len(train_df),
            "num_val_samples": len(val_df),
            "num_test_samples": len(test_df),
        },
        "label_mapping": LABEL_MAPPING,
    }
    with open(output_dir / "eval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    # Save label mapping
    with open(output_dir / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump(LABEL_MAPPING, f, indent=2)

    print(f"[OK] Model, tokenizer, eval_metrics.json, and label_mapping.json saved successfully.")
    return metrics_payload


def main():
    parser = argparse.ArgumentParser(description="Train DistilBERT / BERT model for Fake News Detection.")
    parser.add_argument(
        "--model-name",
        type=str,
        default="distilbert-base-uncased",
        help="Pretrained model identifier (default: 'distilbert-base-uncased')",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.ROOT_DIR / "saved_models" / "distilbert_fake_news",
        help="Directory where the trained model and artifacts will be saved",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of training epochs (default: 2)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size per device (default: 8)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Maximum token sequence length (default: 128)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Optimizer learning rate (default: 2e-5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=3000,
        help="Balanced subset sample size (e.g. 3000 for balanced fast training)",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run quick smoke test on small sub-sample (1 epoch) to verify pipeline",
    )
    parser.add_argument(
        "--smoke-samples",
        type=int,
        default=80,
        help="Number of samples to use in smoke-test mode (default: 80)",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Optional path to pre-cleaned CSV (e.g. data/processed_news.csv)",
    )
    args = parser.parse_args()

    # For smoke test, route output to smoke_test directory if default output_dir is unchanged
    output_dir = args.output_dir
    if args.smoke_test and output_dir == config.ROOT_DIR / "saved_models" / "distilbert_fake_news":
        output_dir = config.ROOT_DIR / "saved_models" / "smoke_test_model"

    train_bert_pipeline(
        model_name=args.model_name,
        output_dir=output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        seed=args.seed,
        sample_size=args.sample_size,
        smoke_test=args.smoke_test,
        smoke_samples=args.smoke_samples,
        data_path=args.data_path,
    )


if __name__ == "__main__":
    main()