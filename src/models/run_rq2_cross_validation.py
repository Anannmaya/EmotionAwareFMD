"""Run repeated pair-preserving cross-validation for RQ2.

Compares:
1. Text-only: TF-IDF + logistic regression
2. Emotion-aware: TF-IDF + five standardised affective features
   + logistic regression

The unit used for the final statistical comparison is one complete
out-of-fold evaluation per repetition, not individual CV folds.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from scipy.stats import rankdata, wilcoxon
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_DIR = PROJECT_ROOT / "data" / "features"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
PREDICTION_DIR = PROJECT_ROOT / "outputs" / "predictions"

AFFECT_COLUMNS = [
    "affect_anger",
    "affect_fear",
    "affect_joy",
    "affect_sadness",
    "affect_valence",
]

N_SPLITS = 5
SEEDS = [11, 22, 33, 44, 55, 66, 77, 88, 99, 110]

TFIDF_SETTINGS = {
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.95,
    "max_features": 20_000,
    "sublinear_tf": True,
    "strip_accents": "unicode",
}

LOGISTIC_SETTINGS = {
    "C": 1.0,
    "solver": "liblinear",
    "max_iter": 2_000,
}


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    path: Path
    id_column: str
    text_column: str
    label_column: str
    pair_column: str
    misleading_original_value: int


DATASETS = [
    DatasetConfig(
        name="liar2_finance",
        path=FEATURE_DIR / "liar2_finance_with_affective_features.csv",
        id_column="id",
        text_column="statement",
        label_column="binary_label",
        pair_column="matched_pair_id",
        misleading_original_value=0,
    ),
    DatasetConfig(
        name="factors_finance",
        path=FEATURE_DIR / "factors_with_affective_features.csv",
        id_column="id",
        text_column="text",
        label_column="label_id",
        pair_column="pair_id",
        misleading_original_value=1,
    ),
]


def load_and_validate_dataset(config: DatasetConfig) -> pd.DataFrame:
    """Load one dataset and verify the matched-pair structure."""

    if not config.path.exists():
        raise FileNotFoundError(f"Dataset not found: {config.path}")

    df = pd.read_csv(config.path)

    required_columns = [
        config.id_column,
        config.text_column,
        config.label_column,
        config.pair_column,
        *AFFECT_COLUMNS,
    ]

    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(
            f"{config.name} is missing columns: {missing_columns}"
        )

    if df[required_columns].isna().any().any():
        missing_counts = df[required_columns].isna().sum()
        missing_counts = missing_counts[missing_counts > 0]
        raise ValueError(
            f"{config.name} contains missing required values:\n"
            f"{missing_counts}"
        )

    df = df.copy()
    df[config.text_column] = df[config.text_column].astype(str)

    # Standard project convention:
    # 1 = misleading, 0 = genuine.
    df["target"] = (
        df[config.label_column].astype(int)
        == config.misleading_original_value
    ).astype(int)

    pair_sizes = df.groupby(config.pair_column).size()
    invalid_sizes = pair_sizes[pair_sizes != 2]

    if not invalid_sizes.empty:
        raise ValueError(
            f"{config.name} has pairs that do not contain exactly two rows:\n"
            f"{invalid_sizes.head(10)}"
        )

    pair_label_counts = (
        df.groupby(config.pair_column)["target"]
        .apply(lambda values: set(values.tolist()))
    )
    invalid_labels = pair_label_counts[
        pair_label_counts.apply(lambda labels: labels != {0, 1})
    ]

    if not invalid_labels.empty:
        raise ValueError(
            f"{config.name} has pairs without one genuine and one "
            f"misleading item:\n{invalid_labels.head(10)}"
        )

    for column in AFFECT_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="raise")

    print(
        f"{config.name}: "
        f"{len(df):,} rows, "
        f"{df[config.pair_column].nunique():,} valid pairs"
    )

    return df.reset_index(drop=True)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Calculate the project evaluation metrics."""

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true,
            y_pred,
            pos_label=1,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            pos_label=1,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            y_pred,
            pos_label=1,
            zero_division=0,
        ),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
    }


def create_pair_preserving_splits(
    df: pd.DataFrame,
    pair_column: str,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create shuffled folds while keeping pair members together."""

    unique_pairs = df[pair_column].drop_duplicates().to_numpy()

    splitter = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=seed,
    )

    splits: list[tuple[np.ndarray, np.ndarray]] = []

    for train_pair_indices, test_pair_indices in splitter.split(
        unique_pairs
    ):
        train_pairs = set(unique_pairs[train_pair_indices])
        test_pairs = set(unique_pairs[test_pair_indices])

        train_indices = df.index[
            df[pair_column].isin(train_pairs)
        ].to_numpy()

        test_indices = df.index[
            df[pair_column].isin(test_pairs)
        ].to_numpy()

        if train_pairs.intersection(test_pairs):
            raise RuntimeError("Pair leakage detected between folds.")

        splits.append((train_indices, test_indices))

    return splits


def fit_and_predict_fold(
    df: pd.DataFrame,
    config: DatasetConfig,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    seed: int,
) -> dict[str, dict[str, np.ndarray]]:
    """Train both model variants and predict one test fold."""

    train_df = df.loc[train_indices]
    test_df = df.loc[test_indices]

    vectorizer = TfidfVectorizer(**TFIDF_SETTINGS)

    x_train_text = vectorizer.fit_transform(
        train_df[config.text_column]
    )
    x_test_text = vectorizer.transform(
        test_df[config.text_column]
    )

    y_train = train_df["target"].to_numpy()
    y_test = test_df["target"].to_numpy()

    text_model = LogisticRegression(
        random_state=seed,
        **LOGISTIC_SETTINGS,
    )
    text_model.fit(x_train_text, y_train)

    text_predictions = text_model.predict(x_test_text)
    text_probabilities = text_model.predict_proba(x_test_text)[:, 1]

    affect_scaler = StandardScaler()

    x_train_affect = affect_scaler.fit_transform(
        train_df[AFFECT_COLUMNS]
    )
    x_test_affect = affect_scaler.transform(
        test_df[AFFECT_COLUMNS]
    )

    x_train_emotion = hstack(
        [
            x_train_text,
            csr_matrix(x_train_affect),
        ],
        format="csr",
    )
    x_test_emotion = hstack(
        [
            x_test_text,
            csr_matrix(x_test_affect),
        ],
        format="csr",
    )

    emotion_model = LogisticRegression(
        random_state=seed,
        **LOGISTIC_SETTINGS,
    )
    emotion_model.fit(x_train_emotion, y_train)

    emotion_predictions = emotion_model.predict(x_test_emotion)
    emotion_probabilities = emotion_model.predict_proba(
        x_test_emotion
    )[:, 1]

    return {
        "text_only": {
            "truth": y_test,
            "prediction": text_predictions,
            "probability": text_probabilities,
        },
        "emotion_aware": {
            "truth": y_test,
            "prediction": emotion_predictions,
            "probability": emotion_probabilities,
        },
    }


def run_dataset(
    config: DatasetConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run all repetitions and folds for one dataset."""

    df = load_and_validate_dataset(config)

    fold_metric_rows: list[dict[str, object]] = []
    repetition_metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []

    for repetition, seed in enumerate(SEEDS, start=1):
        splits = create_pair_preserving_splits(
            df=df,
            pair_column=config.pair_column,
            seed=seed,
        )

        repetition_predictions = {
            "text_only": np.full(len(df), -1, dtype=int),
            "emotion_aware": np.full(len(df), -1, dtype=int),
        }

        repetition_probabilities = {
            "text_only": np.full(len(df), np.nan, dtype=float),
            "emotion_aware": np.full(len(df), np.nan, dtype=float),
        }

        for fold, (train_indices, test_indices) in enumerate(
            splits,
            start=1,
        ):
            results = fit_and_predict_fold(
                df=df,
                config=config,
                train_indices=train_indices,
                test_indices=test_indices,
                seed=seed,
            )

            for model_name, model_results in results.items():
                y_true = model_results["truth"]
                y_pred = model_results["prediction"]
                y_probability = model_results["probability"]

                fold_metrics = calculate_metrics(y_true, y_pred)

                fold_metric_rows.append(
                    {
                        "dataset": config.name,
                        "repetition": repetition,
                        "seed": seed,
                        "fold": fold,
                        "model": model_name,
                        "train_rows": len(train_indices),
                        "test_rows": len(test_indices),
                        **fold_metrics,
                    }
                )

                repetition_predictions[model_name][
                    test_indices
                ] = y_pred

                repetition_probabilities[model_name][
                    test_indices
                ] = y_probability

                for position, row_index in enumerate(test_indices):
                    prediction_rows.append(
                        {
                            "dataset": config.name,
                            "repetition": repetition,
                            "seed": seed,
                            "fold": fold,
                            "model": model_name,
                            "row_index": int(row_index),
                            "item_id": df.loc[
                                row_index,
                                config.id_column,
                            ],
                            "pair_id": df.loc[
                                row_index,
                                config.pair_column,
                            ],
                            "y_true": int(y_true[position]),
                            "y_pred": int(y_pred[position]),
                            "probability_misleading": float(
                                y_probability[position]
                            ),
                        }
                    )

        for model_name in ["text_only", "emotion_aware"]:
            predictions = repetition_predictions[model_name]

            if np.any(predictions == -1):
                raise RuntimeError(
                    f"Incomplete OOF predictions for "
                    f"{config.name}, seed={seed}, "
                    f"model={model_name}"
                )

            repetition_metrics = calculate_metrics(
                df["target"].to_numpy(),
                predictions,
            )

            repetition_metric_rows.append(
                {
                    "dataset": config.name,
                    "repetition": repetition,
                    "seed": seed,
                    "model": model_name,
                    **repetition_metrics,
                }
            )

        print(
            f"{config.name}: completed repetition "
            f"{repetition}/{len(SEEDS)}"
        )

    return (
        pd.DataFrame(fold_metric_rows),
        pd.DataFrame(repetition_metric_rows),
        pd.DataFrame(prediction_rows),
    )


def calculate_rank_biserial(differences: np.ndarray) -> float:
    """Calculate paired rank-biserial correlation."""

    nonzero = differences[differences != 0]

    if len(nonzero) == 0:
        return 0.0

    ranks = rankdata(np.abs(nonzero))
    positive_sum = ranks[nonzero > 0].sum()
    negative_sum = ranks[nonzero < 0].sum()

    return float(
        (positive_sum - negative_sum)
        / (positive_sum + negative_sum)
    )


def holm_adjust(p_values: list[float]) -> list[float]:
    """Apply Holm's multiple-comparison correction."""

    p_array = np.asarray(p_values, dtype=float)
    order = np.argsort(p_array)
    adjusted = np.empty_like(p_array)

    running_max = 0.0
    number_of_tests = len(p_array)

    for rank, original_index in enumerate(order):
        multiplier = number_of_tests - rank
        corrected = min(1.0, multiplier * p_array[original_index])
        running_max = max(running_max, corrected)
        adjusted[original_index] = running_max

    return adjusted.tolist()


def create_statistical_summary(
    repetition_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Compare paired repetition-level scores using Wilcoxon."""

    rows: list[dict[str, object]] = []

    for dataset_name in repetition_metrics["dataset"].unique():
        dataset_results = repetition_metrics[
            repetition_metrics["dataset"] == dataset_name
        ]

        for metric in ["macro_f1", "balanced_accuracy"]:
            paired = dataset_results.pivot(
                index="repetition",
                columns="model",
                values=metric,
            ).dropna()

            text_scores = paired["text_only"].to_numpy()
            emotion_scores = paired["emotion_aware"].to_numpy()
            differences = emotion_scores - text_scores

            if np.allclose(differences, 0):
                statistic = 0.0
                p_value = 1.0
            else:
                test_result = wilcoxon(
                    emotion_scores,
                    text_scores,
                    alternative="two-sided",
                    zero_method="wilcox",
                )
                statistic = float(test_result.statistic)
                p_value = float(test_result.pvalue)

            rows.append(
                {
                    "dataset": dataset_name,
                    "metric": metric,
                    "n_repetitions": len(paired),
                    "text_only_mean": text_scores.mean(),
                    "text_only_std": text_scores.std(ddof=1),
                    "emotion_aware_mean": emotion_scores.mean(),
                    "emotion_aware_std": emotion_scores.std(ddof=1),
                    "mean_difference_emotion_minus_text": (
                        differences.mean()
                    ),
                    "median_difference_emotion_minus_text": (
                        np.median(differences)
                    ),
                    "wilcoxon_statistic": statistic,
                    "p_value": p_value,
                    "rank_biserial": calculate_rank_biserial(
                        differences
                    ),
                }
            )

    summary = pd.DataFrame(rows)
    summary["p_value_holm"] = holm_adjust(
        summary["p_value"].tolist()
    )
    summary["significant_holm_0_05"] = (
        summary["p_value_holm"] < 0.05
    )

    return summary


def main() -> None:
    """Run the complete RQ2 cross-validation experiment."""

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

    all_fold_metrics: list[pd.DataFrame] = []
    all_repetition_metrics: list[pd.DataFrame] = []
    all_predictions: list[pd.DataFrame] = []

    for config in DATASETS:
        fold_metrics, repetition_metrics, predictions = run_dataset(
            config
        )

        all_fold_metrics.append(fold_metrics)
        all_repetition_metrics.append(repetition_metrics)
        all_predictions.append(predictions)

    combined_fold_metrics = pd.concat(
        all_fold_metrics,
        ignore_index=True,
    )
    combined_repetition_metrics = pd.concat(
        all_repetition_metrics,
        ignore_index=True,
    )
    combined_predictions = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    statistical_summary = create_statistical_summary(
        combined_repetition_metrics
    )

    combined_fold_metrics.to_csv(
        TABLE_DIR / "rq2_fold_metrics.csv",
        index=False,
    )
    combined_repetition_metrics.to_csv(
        TABLE_DIR / "rq2_repetition_metrics.csv",
        index=False,
    )
    statistical_summary.to_csv(
        TABLE_DIR / "rq2_statistical_comparison.csv",
        index=False,
    )
    combined_predictions.to_csv(
        PREDICTION_DIR / "rq2_oof_predictions.csv",
        index=False,
    )

    print("\nRQ2 experiment complete.")
    print(statistical_summary.to_string(index=False))


if __name__ == "__main__":
    main()