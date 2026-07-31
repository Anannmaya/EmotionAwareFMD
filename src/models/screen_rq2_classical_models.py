"""Screen stronger classical model configurations for RQ2.

This is a development-stage model comparison, not the final reported nested
cross-validation experiment. It compares:

1. Logistic Regression and Linear SVM.
2. Word TF-IDF and combined word plus character TF-IDF.
3. Text-only and emotion-aware variants.
4. Several affective fusion weights, including zero.

The experiment uses repeated pair-preserving five-fold cross-validation. Its
purpose is to identify the strongest classical configuration to carry forward
into the final nested evaluation. RFC-Bench is deliberately not used here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src.models.run_rq2_cross_validation import (
    AFFECT_COLUMNS,
    DATASETS,
    TABLE_DIR,
    calculate_metrics,
    create_pair_preserving_splits,
    load_and_validate_dataset,
)


SCREENING_SEEDS = [11, 22, 33]
AFFECT_WEIGHTS = [0.0, 0.05, 0.10, 0.25, 0.50, 1.00]
C_VALUES = [0.1, 1.0, 10.0]

WORD_SETTINGS = {
    "analyzer": "word",
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.95,
    "max_features": 20_000,
    "sublinear_tf": True,
    "strip_accents": "unicode",
}

CHAR_SETTINGS = {
    "analyzer": "char_wb",
    "ngram_range": (3, 5),
    "min_df": 2,
    "max_features": 30_000,
    "sublinear_tf": True,
    "strip_accents": "unicode",
}


@dataclass(frozen=True)
class Candidate:
    classifier: str
    representation: str
    c_value: float
    affect_weight: float

    @property
    def model_variant(self) -> str:
        return "text_only" if self.affect_weight == 0.0 else "emotion_aware"


def build_text_features(
    train_text: pd.Series,
    test_text: pd.Series,
    representation: str,
):
    """Fit text transforms on training data and transform one test fold."""

    word_vectorizer = TfidfVectorizer(**WORD_SETTINGS)
    x_train_word = word_vectorizer.fit_transform(train_text)
    x_test_word = word_vectorizer.transform(test_text)

    if representation == "word":
        return x_train_word, x_test_word

    if representation != "word_char":
        raise ValueError(f"Unknown representation: {representation}")

    char_vectorizer = TfidfVectorizer(**CHAR_SETTINGS)
    x_train_char = char_vectorizer.fit_transform(train_text)
    x_test_char = char_vectorizer.transform(test_text)

    return (
        hstack([x_train_word, x_train_char], format="csr"),
        hstack([x_test_word, x_test_char], format="csr"),
    )


def append_affective_features(
    x_train_text,
    x_test_text,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    affect_weight: float,
):
    """Append standardised affective features using a specified weight."""

    if affect_weight == 0.0:
        return x_train_text, x_test_text

    scaler = StandardScaler()
    x_train_affect = scaler.fit_transform(train_df[AFFECT_COLUMNS])
    x_test_affect = scaler.transform(test_df[AFFECT_COLUMNS])

    return (
        hstack(
            [
                x_train_text,
                csr_matrix(x_train_affect * affect_weight),
            ],
            format="csr",
        ),
        hstack(
            [
                x_test_text,
                csr_matrix(x_test_affect * affect_weight),
            ],
            format="csr",
        ),
    )


def create_classifier(
    classifier: str,
    c_value: float,
    seed: int,
):
    """Create one linear classifier."""

    if classifier == "logistic_regression":
        return LogisticRegression(
            C=c_value,
            solver="liblinear",
            max_iter=2_000,
            random_state=seed,
        )

    if classifier == "linear_svm":
        return LinearSVC(
            C=c_value,
            random_state=seed,
        )

    raise ValueError(f"Unknown classifier: {classifier}")


def generate_candidates() -> list[Candidate]:
    """Return the compact classical screening grid."""

    candidates: list[Candidate] = []

    for classifier in ["logistic_regression", "linear_svm"]:
        for representation in ["word", "word_char"]:
            for c_value in C_VALUES:
                for affect_weight in AFFECT_WEIGHTS:
                    candidates.append(
                        Candidate(
                            classifier=classifier,
                            representation=representation,
                            c_value=c_value,
                            affect_weight=affect_weight,
                        )
                    )

    return candidates


def run_dataset_screening(config) -> pd.DataFrame:
    """Run repeated pair-preserving screening for one dataset."""

    df = load_and_validate_dataset(config)
    candidates = generate_candidates()
    metric_rows: list[dict[str, object]] = []

    for repetition, seed in enumerate(SCREENING_SEEDS, start=1):
        splits = create_pair_preserving_splits(
            df=df,
            pair_column=config.pair_column,
            seed=seed,
        )

        for fold, (train_indices, test_indices) in enumerate(
            splits,
            start=1,
        ):
            train_df = df.loc[train_indices]
            test_df = df.loc[test_indices]
            y_train = train_df["target"].to_numpy()
            y_test = test_df["target"].to_numpy()

            text_cache: dict[str, tuple[object, object]] = {}

            for representation in ["word", "word_char"]:
                text_cache[representation] = build_text_features(
                    train_text=train_df[config.text_column],
                    test_text=test_df[config.text_column],
                    representation=representation,
                )

            for candidate_index, candidate in enumerate(
                candidates,
                start=1,
            ):
                x_train_text, x_test_text = text_cache[
                    candidate.representation
                ]
                x_train, x_test = append_affective_features(
                    x_train_text=x_train_text,
                    x_test_text=x_test_text,
                    train_df=train_df,
                    test_df=test_df,
                    affect_weight=candidate.affect_weight,
                )

                model_seed = seed + fold * 1_000 + candidate_index
                model = create_classifier(
                    classifier=candidate.classifier,
                    c_value=candidate.c_value,
                    seed=model_seed,
                )
                model.fit(x_train, y_train)
                predictions = model.predict(x_test)
                metrics = calculate_metrics(y_test, predictions)

                metric_rows.append(
                    {
                        "dataset": config.name,
                        "repetition": repetition,
                        "seed": seed,
                        "fold": fold,
                        "classifier": candidate.classifier,
                        "representation": candidate.representation,
                        "C": candidate.c_value,
                        "affect_weight": candidate.affect_weight,
                        "model_variant": candidate.model_variant,
                        **metrics,
                    }
                )

            print(
                f"{config.name}: screening repetition {repetition}/"
                f"{len(SCREENING_SEEDS)}, fold {fold}/5 complete"
            )

    return pd.DataFrame(metric_rows)


def summarise_screening(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate screening scores and rank candidate configurations."""

    group_columns = [
        "dataset",
        "classifier",
        "representation",
        "C",
        "affect_weight",
        "model_variant",
    ]

    summary = (
        fold_metrics.groupby(group_columns, as_index=False)
        .agg(
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            accuracy_mean=("accuracy", "mean"),
            precision_mean=("precision", "mean"),
            recall_mean=("recall", "mean"),
            f1_mean=("f1", "mean"),
            folds=("macro_f1", "size"),
        )
        .sort_values(
            [
                "dataset",
                "macro_f1_mean",
                "balanced_accuracy_mean",
            ],
            ascending=[True, False, False],
        )
        .reset_index(drop=True)
    )

    summary["rank_within_dataset"] = (
        summary.groupby("dataset")["macro_f1_mean"]
        .rank(method="dense", ascending=False)
        .astype(int)
    )

    return summary


def main() -> None:
    """Run and save the classical model screening experiment."""

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    all_metrics = [run_dataset_screening(config) for config in DATASETS]
    fold_metrics = pd.concat(all_metrics, ignore_index=True)
    summary = summarise_screening(fold_metrics)

    fold_output = TABLE_DIR / "rq2_classical_screening_fold_metrics.csv"
    summary_output = TABLE_DIR / "rq2_classical_screening_summary.csv"

    fold_metrics.to_csv(fold_output, index=False)
    summary.to_csv(summary_output, index=False)

    print("\nClassical model screening complete.")
    print(f"Fold metrics: {fold_output}")
    print(f"Summary: {summary_output}")
    print("\nTop five configurations per dataset:")
    print(
        summary.groupby("dataset", group_keys=False)
        .head(5)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
