"""Run the final nested classical model-selection experiment for RQ2.

This experiment compares the strongest classical alternatives without using
RFC-Bench for model selection.

Outer evaluation:
- 10 repetitions
- 5 pair-preserving folds

Inner model selection:
- 3 pair-preserving folds
- Logistic Regression versus Linear SVM
- word TF-IDF versus word plus character TF-IDF
- C in {0.1, 1.0, 10.0}
- affect weight in {0.05, 0.10, 0.25, 0.50, 1.00} for the emotion-aware model

The text-only and emotion-aware variants are tuned independently on exactly
the same inner folds. The outer test fold is never used for model selection.
Existing RQ2 result files are not overwritten.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src.models.run_rq2_cross_validation import (
    AFFECT_COLUMNS,
    DATASETS,
    PREDICTION_DIR,
    SEEDS,
    TABLE_DIR,
    DatasetConfig,
    calculate_metrics,
    create_pair_preserving_splits,
    create_statistical_summary,
    load_and_validate_dataset,
)


INNER_SPLITS = 3
CLASSIFIERS = ["logistic_regression", "linear_svm"]
REPRESENTATIONS = ["word", "word_char"]
C_VALUES = [0.1, 1.0, 10.0]
AFFECT_WEIGHTS = [0.05, 0.10, 0.25, 0.50, 1.00]

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
    affect_weight: float | None

    @property
    def model_variant(self) -> str:
        return "text_only" if self.affect_weight is None else "emotion_aware"


@dataclass
class SplitFeatures:
    y_train: np.ndarray
    y_evaluation: np.ndarray
    train_text: dict[str, object]
    evaluation_text: dict[str, object]
    train_affect: np.ndarray
    evaluation_affect: np.ndarray


def generate_candidates(model_variant: str) -> list[Candidate]:
    """Return the compact candidate grid for one model variant."""

    if model_variant not in {"text_only", "emotion_aware"}:
        raise ValueError(f"Unknown model variant: {model_variant}")

    weights: list[float | None]
    if model_variant == "text_only":
        weights = [None]
    else:
        weights = AFFECT_WEIGHTS

    return [
        Candidate(
            classifier=classifier,
            representation=representation,
            c_value=c_value,
            affect_weight=affect_weight,
        )
        for classifier in CLASSIFIERS
        for representation in REPRESENTATIONS
        for c_value in C_VALUES
        for affect_weight in weights
    ]


def create_inner_pair_splits(
    df: pd.DataFrame,
    outer_train_indices: np.ndarray,
    pair_column: str,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create pair-preserving inner folds inside one outer training set."""

    outer_train_df = df.loc[outer_train_indices]
    unique_pairs = outer_train_df[pair_column].drop_duplicates().to_numpy()

    if len(unique_pairs) < INNER_SPLITS:
        raise ValueError(
            f"Cannot create {INNER_SPLITS} inner folds from "
            f"{len(unique_pairs)} pairs."
        )

    splitter = KFold(
        n_splits=INNER_SPLITS,
        shuffle=True,
        random_state=seed,
    )

    splits: list[tuple[np.ndarray, np.ndarray]] = []

    for train_positions, validation_positions in splitter.split(unique_pairs):
        train_pairs = set(unique_pairs[train_positions])
        validation_pairs = set(unique_pairs[validation_positions])

        if train_pairs.intersection(validation_pairs):
            raise RuntimeError("Pair leakage detected inside inner CV.")

        train_indices = outer_train_df.index[
            outer_train_df[pair_column].isin(train_pairs)
        ].to_numpy()
        validation_indices = outer_train_df.index[
            outer_train_df[pair_column].isin(validation_pairs)
        ].to_numpy()

        splits.append((train_indices, validation_indices))

    return splits


def prepare_split_features(
    df: pd.DataFrame,
    config: DatasetConfig,
    train_indices: np.ndarray,
    evaluation_indices: np.ndarray,
) -> SplitFeatures:
    """Fit split-specific transforms and cache all feature representations."""

    train_df = df.loc[train_indices]
    evaluation_df = df.loc[evaluation_indices]

    word_vectorizer = TfidfVectorizer(**WORD_SETTINGS)
    x_train_word = word_vectorizer.fit_transform(
        train_df[config.text_column]
    )
    x_evaluation_word = word_vectorizer.transform(
        evaluation_df[config.text_column]
    )

    char_vectorizer = TfidfVectorizer(**CHAR_SETTINGS)
    x_train_char = char_vectorizer.fit_transform(
        train_df[config.text_column]
    )
    x_evaluation_char = char_vectorizer.transform(
        evaluation_df[config.text_column]
    )

    affect_scaler = StandardScaler()
    x_train_affect = affect_scaler.fit_transform(
        train_df[AFFECT_COLUMNS]
    )
    x_evaluation_affect = affect_scaler.transform(
        evaluation_df[AFFECT_COLUMNS]
    )

    return SplitFeatures(
        y_train=train_df["target"].to_numpy(),
        y_evaluation=evaluation_df["target"].to_numpy(),
        train_text={
            "word": x_train_word,
            "word_char": hstack(
                [x_train_word, x_train_char],
                format="csr",
            ),
        },
        evaluation_text={
            "word": x_evaluation_word,
            "word_char": hstack(
                [x_evaluation_word, x_evaluation_char],
                format="csr",
            ),
        },
        train_affect=x_train_affect,
        evaluation_affect=x_evaluation_affect,
    )


def get_candidate_matrices(
    features: SplitFeatures,
    candidate: Candidate,
):
    """Return cached text features with optional weighted affective features."""

    x_train = features.train_text[candidate.representation]
    x_evaluation = features.evaluation_text[candidate.representation]

    if candidate.affect_weight is None:
        return x_train, x_evaluation

    return (
        hstack(
            [
                x_train,
                csr_matrix(
                    features.train_affect * candidate.affect_weight
                ),
            ],
            format="csr",
        ),
        hstack(
            [
                x_evaluation,
                csr_matrix(
                    features.evaluation_affect * candidate.affect_weight
                ),
            ],
            format="csr",
        ),
    )


def create_classifier(candidate: Candidate, seed: int):
    """Create one linear classifier from a candidate specification."""

    if candidate.classifier == "logistic_regression":
        return LogisticRegression(
            C=candidate.c_value,
            solver="liblinear",
            max_iter=2_000,
            random_state=seed,
        )

    if candidate.classifier == "linear_svm":
        return LinearSVC(
            C=candidate.c_value,
            random_state=seed,
        )

    raise ValueError(f"Unknown classifier: {candidate.classifier}")


def fit_and_predict(
    features: SplitFeatures,
    candidate: Candidate,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit one candidate and return predictions and continuous scores."""

    x_train, x_evaluation = get_candidate_matrices(features, candidate)
    model = create_classifier(candidate, seed)
    model.fit(x_train, features.y_train)

    predictions = model.predict(x_evaluation)

    if candidate.classifier == "logistic_regression":
        scores = model.predict_proba(x_evaluation)[:, 1]
    else:
        scores = model.decision_function(x_evaluation)

    return predictions, np.asarray(scores, dtype=float)


def candidate_sort_key(candidate: Candidate) -> tuple[int, int, float, float]:
    """Prefer simpler candidates when inner validation scores tie."""

    classifier_preference = (
        1 if candidate.classifier == "logistic_regression" else 0
    )
    representation_preference = (
        1 if candidate.representation == "word" else 0
    )
    affect_weight = (
        0.0 if candidate.affect_weight is None else candidate.affect_weight
    )

    return (
        classifier_preference,
        representation_preference,
        -candidate.c_value,
        -affect_weight,
    )


def tune_model_variant(
    inner_feature_sets: list[SplitFeatures],
    model_variant: str,
    dataset_name: str,
    repetition: int,
    outer_seed: int,
    outer_fold: int,
) -> tuple[Candidate, list[dict[str, object]]]:
    """Select one candidate using only inner validation predictions."""

    candidates = generate_candidates(model_variant)
    tuning_rows: list[dict[str, object]] = []
    best_candidate: Candidate | None = None
    best_key: tuple[float, float, int, int, float, float] | None = None

    for candidate_index, candidate in enumerate(candidates, start=1):
        truths: list[np.ndarray] = []
        predictions: list[np.ndarray] = []

        for inner_fold, features in enumerate(
            inner_feature_sets,
            start=1,
        ):
            seed = (
                outer_seed
                + outer_fold * 10_000
                + inner_fold * 100
                + candidate_index
            )
            fold_predictions, _ = fit_and_predict(
                features=features,
                candidate=candidate,
                seed=seed,
            )
            truths.append(features.y_evaluation)
            predictions.append(fold_predictions)

        pooled_truth = np.concatenate(truths)
        pooled_predictions = np.concatenate(predictions)
        metrics = calculate_metrics(pooled_truth, pooled_predictions)

        simple_key = candidate_sort_key(candidate)
        selection_key = (
            float(metrics["macro_f1"]),
            float(metrics["balanced_accuracy"]),
            *simple_key,
        )

        tuning_rows.append(
            {
                "dataset": dataset_name,
                "repetition": repetition,
                "outer_seed": outer_seed,
                "outer_fold": outer_fold,
                "model": model_variant,
                "candidate_index": candidate_index,
                "classifier": candidate.classifier,
                "representation": candidate.representation,
                "C": candidate.c_value,
                "affect_weight": (
                    candidate.affect_weight
                    if candidate.affect_weight is not None
                    else np.nan
                ),
                "inner_macro_f1": metrics["macro_f1"],
                "inner_balanced_accuracy": metrics["balanced_accuracy"],
                "selected": False,
            }
        )

        if best_key is None or selection_key > best_key:
            best_key = selection_key
            best_candidate = candidate

    if best_candidate is None:
        raise RuntimeError(
            f"No candidate selected for {dataset_name}, {model_variant}, "
            f"repetition={repetition}, outer_fold={outer_fold}."
        )

    for row in tuning_rows:
        row["selected"] = (
            row["classifier"] == best_candidate.classifier
            and row["representation"] == best_candidate.representation
            and float(row["C"]) == best_candidate.c_value
            and (
                (
                    pd.isna(row["affect_weight"])
                    and best_candidate.affect_weight is None
                )
                or (
                    not pd.isna(row["affect_weight"])
                    and best_candidate.affect_weight is not None
                    and float(row["affect_weight"])
                    == best_candidate.affect_weight
                )
            )
        )

    return best_candidate, tuning_rows


def run_nested_dataset(
    config: DatasetConfig,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Run nested repeated evaluation for one dataset."""

    df = load_and_validate_dataset(config)

    fold_rows: list[dict[str, object]] = []
    repetition_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    tuning_rows: list[dict[str, object]] = []

    for repetition, outer_seed in enumerate(SEEDS, start=1):
        outer_splits = create_pair_preserving_splits(
            df=df,
            pair_column=config.pair_column,
            seed=outer_seed,
        )

        repetition_predictions = {
            "text_only": np.full(len(df), -1, dtype=int),
            "emotion_aware": np.full(len(df), -1, dtype=int),
        }
        repetition_scores = {
            "text_only": np.full(len(df), np.nan, dtype=float),
            "emotion_aware": np.full(len(df), np.nan, dtype=float),
        }

        for outer_fold, (
            outer_train_indices,
            outer_test_indices,
        ) in enumerate(outer_splits, start=1):
            inner_splits = create_inner_pair_splits(
                df=df,
                outer_train_indices=outer_train_indices,
                pair_column=config.pair_column,
                seed=outer_seed + outer_fold * 10_000,
            )

            inner_feature_sets = [
                prepare_split_features(
                    df=df,
                    config=config,
                    train_indices=inner_train_indices,
                    evaluation_indices=inner_validation_indices,
                )
                for (
                    inner_train_indices,
                    inner_validation_indices,
                ) in inner_splits
            ]

            outer_features = prepare_split_features(
                df=df,
                config=config,
                train_indices=outer_train_indices,
                evaluation_indices=outer_test_indices,
            )

            for model_offset, model_variant in enumerate(
                ["text_only", "emotion_aware"],
                start=1,
            ):
                best_candidate, model_tuning_rows = tune_model_variant(
                    inner_feature_sets=inner_feature_sets,
                    model_variant=model_variant,
                    dataset_name=config.name,
                    repetition=repetition,
                    outer_seed=outer_seed,
                    outer_fold=outer_fold,
                )
                tuning_rows.extend(model_tuning_rows)

                selected_rows.append(
                    {
                        "dataset": config.name,
                        "repetition": repetition,
                        "outer_seed": outer_seed,
                        "outer_fold": outer_fold,
                        "model": model_variant,
                        "classifier": best_candidate.classifier,
                        "representation": best_candidate.representation,
                        "C": best_candidate.c_value,
                        "affect_weight": (
                            best_candidate.affect_weight
                            if best_candidate.affect_weight is not None
                            else np.nan
                        ),
                    }
                )

                final_seed = outer_seed + outer_fold * 1_000 + model_offset
                predictions, scores = fit_and_predict(
                    features=outer_features,
                    candidate=best_candidate,
                    seed=final_seed,
                )

                metrics = calculate_metrics(
                    outer_features.y_evaluation,
                    predictions,
                )

                fold_rows.append(
                    {
                        "dataset": config.name,
                        "repetition": repetition,
                        "outer_seed": outer_seed,
                        "outer_fold": outer_fold,
                        "model": model_variant,
                        "train_rows": len(outer_train_indices),
                        "test_rows": len(outer_test_indices),
                        "classifier": best_candidate.classifier,
                        "representation": best_candidate.representation,
                        "C": best_candidate.c_value,
                        "affect_weight": (
                            best_candidate.affect_weight
                            if best_candidate.affect_weight is not None
                            else np.nan
                        ),
                        **metrics,
                    }
                )

                repetition_predictions[model_variant][
                    outer_test_indices
                ] = predictions
                repetition_scores[model_variant][
                    outer_test_indices
                ] = scores

                for position, row_index in enumerate(outer_test_indices):
                    prediction_rows.append(
                        {
                            "dataset": config.name,
                            "repetition": repetition,
                            "outer_seed": outer_seed,
                            "outer_fold": outer_fold,
                            "model": model_variant,
                            "row_index": int(row_index),
                            "item_id": df.loc[
                                row_index,
                                config.id_column,
                            ],
                            "pair_id": df.loc[
                                row_index,
                                config.pair_column,
                            ],
                            "y_true": int(
                                outer_features.y_evaluation[position]
                            ),
                            "y_pred": int(predictions[position]),
                            "decision_score": float(scores[position]),
                            "classifier": best_candidate.classifier,
                            "representation": best_candidate.representation,
                            "C": best_candidate.c_value,
                            "affect_weight": (
                                best_candidate.affect_weight
                                if best_candidate.affect_weight is not None
                                else np.nan
                            ),
                        }
                    )

            print(
                f"{config.name}: repetition {repetition}/{len(SEEDS)}, "
                f"outer fold {outer_fold}/5 complete"
            )

        for model_variant in ["text_only", "emotion_aware"]:
            predictions = repetition_predictions[model_variant]
            scores = repetition_scores[model_variant]

            if np.any(predictions == -1):
                raise RuntimeError(
                    f"Incomplete predictions for {config.name}, "
                    f"seed={outer_seed}, model={model_variant}."
                )

            if np.isnan(scores).any():
                raise RuntimeError(
                    f"Incomplete scores for {config.name}, "
                    f"seed={outer_seed}, model={model_variant}."
                )

            metrics = calculate_metrics(
                df["target"].to_numpy(),
                predictions,
            )

            repetition_rows.append(
                {
                    "dataset": config.name,
                    "repetition": repetition,
                    "seed": outer_seed,
                    "model": model_variant,
                    **metrics,
                }
            )

        print(
            f"{config.name}: completed repetition "
            f"{repetition}/{len(SEEDS)}"
        )

    return (
        pd.DataFrame(fold_rows),
        pd.DataFrame(repetition_rows),
        pd.DataFrame(prediction_rows),
        pd.DataFrame(selected_rows),
        pd.DataFrame(tuning_rows),
    )


def create_selection_frequency(
    selected_parameters: pd.DataFrame,
) -> pd.DataFrame:
    """Count how often each candidate was selected by inner CV."""

    group_columns = [
        "dataset",
        "model",
        "classifier",
        "representation",
        "C",
        "affect_weight",
    ]

    return (
        selected_parameters.groupby(
            group_columns,
            dropna=False,
            as_index=False,
        )
        .size()
        .rename(columns={"size": "selected_outer_folds"})
        .sort_values(
            ["dataset", "model", "selected_outer_folds"],
            ascending=[True, True, False],
        )
        .reset_index(drop=True)
    )


def main() -> None:
    """Run and save the complete nested classical experiment."""

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

    outputs = [run_nested_dataset(config) for config in DATASETS]

    fold_metrics = pd.concat(
        [output[0] for output in outputs],
        ignore_index=True,
    )
    repetition_metrics = pd.concat(
        [output[1] for output in outputs],
        ignore_index=True,
    )
    predictions = pd.concat(
        [output[2] for output in outputs],
        ignore_index=True,
    )
    selected_parameters = pd.concat(
        [output[3] for output in outputs],
        ignore_index=True,
    )
    tuning_scores = pd.concat(
        [output[4] for output in outputs],
        ignore_index=True,
    )

    statistical_summary = create_statistical_summary(
        repetition_metrics
    )
    selection_frequency = create_selection_frequency(
        selected_parameters
    )

    fold_metrics.to_csv(
        TABLE_DIR / "rq2_classical_nested_fold_metrics.csv",
        index=False,
    )
    repetition_metrics.to_csv(
        TABLE_DIR / "rq2_classical_nested_repetition_metrics.csv",
        index=False,
    )
    statistical_summary.to_csv(
        TABLE_DIR / "rq2_classical_nested_statistical_comparison.csv",
        index=False,
    )
    selected_parameters.to_csv(
        TABLE_DIR / "rq2_classical_nested_selected_hyperparameters.csv",
        index=False,
    )
    tuning_scores.to_csv(
        TABLE_DIR / "rq2_classical_nested_inner_tuning_scores.csv",
        index=False,
    )
    selection_frequency.to_csv(
        TABLE_DIR / "rq2_classical_nested_selection_frequency.csv",
        index=False,
    )
    predictions.to_csv(
        PREDICTION_DIR / "rq2_classical_nested_oof_predictions.csv",
        index=False,
    )

    print("\nNested classical model-selection experiment complete.")
    print(statistical_summary.to_string(index=False))
    print("\nMost frequently selected candidates:")
    print(
        selection_frequency.groupby(
            ["dataset", "model"],
            group_keys=False,
        )
        .head(3)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
