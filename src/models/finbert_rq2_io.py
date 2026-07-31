"""Run the final paired FinBERT versus FinBERT-plus-affect RQ2 experiment.

The two variants use identical FinBERT encoders, pair-preserving splits,
training settings, and early-stopping rules. The only architectural difference
is that the emotion-aware variant concatenates the five standardised affective
scores to the final [CLS] representation before binary classification.

RFC-Bench is evaluated only after training and is never used for model
selection. As RFC contains only misleading examples, its principal score is
misleading-class recall (manipulation detection rate).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.models.finbert_rq2_utils import TokenizedDataset, TrainingConfig
from src.models.run_rq2_cross_validation import (
    AFFECT_COLUMNS,
    DATASETS,
    PREDICTION_DIR,
    SEEDS,
    TABLE_DIR,
    DatasetConfig,
    load_and_validate_dataset,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RFC_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "rfc_bench_with_affective_features_512.csv"
)

METRICS_PATH = TABLE_DIR / "rq2_finbert_repetition_metrics.csv"
STATISTICS_PATH = TABLE_DIR / "rq2_finbert_statistical_comparison.csv"
DIRECTIONAL_PATH = TABLE_DIR / "rq2_finbert_primary_directional_comparison.csv"
HISTORY_PATH = TABLE_DIR / "rq2_finbert_training_history.csv"
SPLITS_PATH = TABLE_DIR / "rq2_finbert_split_assignments.csv"
PREDICTIONS_PATH = PREDICTION_DIR / "rq2_finbert_predictions.csv"
RFC_METRICS_PATH = TABLE_DIR / "rq2_finbert_rfc_seed_metrics.csv"
RFC_SUMMARY_PATH = TABLE_DIR / "rq2_finbert_rfc_summary.csv"
RFC_COMPARISON_PATH = TABLE_DIR / "rq2_finbert_rfc_overall_comparison.csv"
RFC_PREDICTIONS_PATH = PREDICTION_DIR / "rq2_finbert_rfc_predictions.csv"
CONFIG_PATH = TABLE_DIR / "rq2_finbert_run_config.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[config.name for config in DATASETS],
        choices=[config.name for config in DATASETS],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--checkpoint", default="ProsusAI/finbert")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.10)
    parser.add_argument("--max-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-rfc", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() and path.stat().st_size else pd.DataFrame()


def save_rows(
    existing: pd.DataFrame,
    rows: list[dict[str, object]],
    path: Path,
    keys: list[str],
) -> pd.DataFrame:
    if not rows:
        return existing
    combined = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
    combined = combined.drop_duplicates(subset=keys, keep="last")
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)
    return combined


def load_rfc() -> pd.DataFrame:
    if not RFC_PATH.exists():
        raise FileNotFoundError(f"RFC feature dataset not found: {RFC_PATH}")
    df = pd.read_csv(RFC_PATH)
    required = [
        "rfc_id",
        "text",
        "subset",
        "manipulation_category",
        "target",
        "label",
        *AFFECT_COLUMNS,
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"RFC is missing columns: {missing}")
    if df[required].isna().any().any():
        raise ValueError("RFC contains missing required values.")
    if set(df["target"].astype(int).unique()) != {1}:
        raise ValueError("RFC must contain only misleading examples.")
    for column in AFFECT_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="raise")
    return df.reset_index(drop=True)


def make_dataset(
    df: pd.DataFrame,
    indices: np.ndarray,
    dataset_config: DatasetConfig,
    tokenizer,
    training_config: TrainingConfig,
    scaler: StandardScaler,
) -> TokenizedDataset:
    split = df.loc[indices]
    return TokenizedDataset(
        texts=split[dataset_config.text_column],
        labels=split["target"].to_numpy(dtype=int),
        affect=scaler.transform(split[AFFECT_COLUMNS]),
        row_ids=split[dataset_config.id_column],
        pair_ids=split[dataset_config.pair_column],
        tokenizer=tokenizer,
        max_length=training_config.max_length,
    )


def make_rfc_dataset(
    df: pd.DataFrame,
    tokenizer,
    training_config: TrainingConfig,
    scaler: StandardScaler,
) -> TokenizedDataset:
    return TokenizedDataset(
        texts=df["text"],
        labels=df["target"].to_numpy(dtype=int),
        affect=scaler.transform(df[AFFECT_COLUMNS]),
        row_ids=df["rfc_id"],
        pair_ids=df["rfc_id"],
        tokenizer=tokenizer,
        max_length=training_config.max_length,
    )
