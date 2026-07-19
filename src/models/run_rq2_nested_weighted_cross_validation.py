"""Run nested repeated pair-preserving cross-validation for RQ2.

Outer evaluation:
- 10 repetitions
- 5 pair-preserving folds

Inner hyperparameter selection:
- 3 pair-preserving folds
- macro-F1 as the primary selection metric
- balanced accuracy as the first tie-breaker

Compares:
1. Text-only: TF-IDF + logistic regression
2. Emotion-aware: TF-IDF + five standardised, weighted affective features
   + logistic regression

Each model variant is tuned separately using model-specific search grids and
the same inner folds. The emotion-aware grid additionally tunes the affective
fusion weight. The outer test fold is never used for tuning.
"""

from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

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

BASE_TFIDF_SETTINGS = {
    "max_df": 0.95,
    "max_features": 20_000,
    "sublinear_tf": True,
    "strip_accents": "unicode",
}

BASE_LOGISTIC_SETTINGS = {
    "solver": "liblinear",
    "max_iter": 2_000,
}

BASE_PARAMETER_GRID = [
    {
        "ngram_range": ngram_range,
        "min_df": min_df,
        "C": c_value,
    }
    for ngram_range, min_df, c_value in product(
        [(1, 1), (1, 2)],
        [1, 2],
        [0.1, 1.0, 10.0],
    )
]

AFFECT_WEIGHTS = [
    0.05,
    0.10,
    0.25,
    0.50,
    1.00,
]


def get_parameter_grid(
    model_name: str,
) -> list[dict[str, object]]:
    """Return the model-specific nested-CV search grid."""

    if model_name == "text_only":
        return [
            parameters.copy()
            for parameters in BASE_PARAMETER_GRID
        ]

    if model_name == "emotion_aware":
        return [
            {
                **parameters,
                "affect_weight": affect_weight,
            }
            for parameters in BASE_PARAMETER_GRID
            for affect_weight in AFFECT_WEIGHTS
        ]

    raise ValueError(
        f"Unknown model variant: {model_name}"
    )


def create_inner_pair_splits(
    df: pd.DataFrame,
    outer_train_indices: np.ndarray,
    pair_column: str,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create pair-preserving inner folds within one outer training set."""

    outer_train_df = df.loc[outer_train_indices]
    unique_pairs = outer_train_df[pair_column].drop_duplicates().to_numpy()

    if len(unique_pairs) < INNER_SPLITS:
        raise ValueError(
            f"Cannot create {INNER_SPLITS} inner folds from only "
            f"{len(unique_pairs)} pairs."
        )

    splitter = KFold(
        n_splits=INNER_SPLITS,
        shuffle=True,
        random_state=seed,
    )

    splits: list[tuple[np.ndarray, np.ndarray]] = []

    for inner_train_pair_positions, inner_validation_pair_positions in (
        splitter.split(unique_pairs)
    ):
        inner_train_pairs = set(
            unique_pairs[inner_train_pair_positions]
        )
        inner_validation_pairs = set(
            unique_pairs[inner_validation_pair_positions]
        )

        if inner_train_pairs.intersection(inner_validation_pairs):
            raise RuntimeError(
                "Pair leakage detected inside the nested CV procedure."
            )

        inner_train_indices = outer_train_df.index[
            outer_train_df[pair_column].isin(inner_train_pairs)
        ].to_numpy()

        inner_validation_indices = outer_train_df.index[
            outer_train_df[pair_column].isin(inner_validation_pairs)
        ].to_numpy()

        splits.append(
            (inner_train_indices, inner_validation_indices)
        )

    return splits


def build_feature_matrices(
    train_df: pd.DataFrame,
    evaluation_df: pd.DataFrame,
    config: DatasetConfig,
    model_name: str,
    parameters: dict[str, object],
):
    """Fit training-only feature transforms and transform evaluation data."""

    vectorizer = TfidfVectorizer(
        ngram_range=parameters["ngram_range"],
        min_df=int(parameters["min_df"]),
        **BASE_TFIDF_SETTINGS,
    )

    x_train_text = vectorizer.fit_transform(
        train_df[config.text_column]
    )
    x_evaluation_text = vectorizer.transform(
        evaluation_df[config.text_column]
    )

    if model_name == "text_only":
        return x_train_text, x_evaluation_text

    if model_name != "emotion_aware":
        raise ValueError(f"Unknown model variant: {model_name}")

    affect_scaler = StandardScaler()
    affect_weight = float(parameters["affect_weight"])

    x_train_affect = affect_scaler.fit_transform(
        train_df[AFFECT_COLUMNS]
    )
    x_evaluation_affect = affect_scaler.transform(
        evaluation_df[AFFECT_COLUMNS]
    )

    x_train = hstack(
        [
            x_train_text,
            csr_matrix(
                x_train_affect * affect_weight
            ),
        ],
        format="csr",
    )
    x_evaluation = hstack(
        [
            x_evaluation_text,
            csr_matrix(
                x_evaluation_affect * affect_weight
            ),
        ],
        format="csr",
    )

    return x_train, x_evaluation


def fit_and_predict_model(
    df: pd.DataFrame,
    config: DatasetConfig,
    train_indices: np.ndarray,
    evaluation_indices: np.ndarray,
    model_name: str,
    parameters: dict[str, object],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train one model variant and predict one held-out partition."""

    train_df = df.loc[train_indices]
    evaluation_df = df.loc[evaluation_indices]

    x_train, x_evaluation = build_feature_matrices(
        train_df=train_df,
        evaluation_df=evaluation_df,
        config=config,
        model_name=model_name,
        parameters=parameters,
    )

    y_train = train_df["target"].to_numpy()
    y_evaluation = evaluation_df["target"].to_numpy()

    model = LogisticRegression(
        C=float(parameters["C"]),
        random_state=seed,
        **BASE_LOGISTIC_SETTINGS,
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_evaluation)
    probabilities = model.predict_proba(x_evaluation)[:, 1]

    return y_evaluation, predictions, probabilities


def parameter_sort_key(
    parameters: dict[str, object],
) -> tuple[int, int, float, float]:
    """Prefer simpler settings when validation scores are exactly tied."""

    ngram_end = int(parameters["ngram_range"][1])
    min_df = int(parameters["min_df"])
    c_value = float(parameters["C"])
    affect_weight = float(parameters.get("affect_weight", 0.0))

    # Higher tuple values are preferred:
    # - unigrams before bigrams
    # - min_df=2 before min_df=1
    # - stronger regularisation (smaller C) before weaker regularisation
    # - smaller affective weight before larger weight
    return (
        -ngram_end,
        min_df,
        -c_value,
        -affect_weight,
    )


def tune_model_on_outer_training_data(
    df: pd.DataFrame,
    config: DatasetConfig,
    inner_splits: list[tuple[np.ndarray, np.ndarray]],
    model_name: str,
    repetition: int,
    outer_seed: int,
    outer_fold: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Select hyperparameters using only the outer training partition."""

    tuning_rows: list[dict[str, object]] = []
    best_parameters: dict[str, object] | None = None
    best_parameter_index: int | None = None
    best_selection_key: (
        tuple[float, float, int, int, float, float]
        | None
    ) = None

    parameter_grid = get_parameter_grid(model_name)

    for parameter_index, parameters in enumerate(
        parameter_grid,
        start=1,
    ):
        inner_truth: list[np.ndarray] = []
        inner_predictions: list[np.ndarray] = []

        for inner_fold, (
            inner_train_indices,
            inner_validation_indices,
        ) in enumerate(inner_splits, start=1):
            model_seed = (
                outer_seed
                + outer_fold * 1_000
                + inner_fold * 100
                + parameter_index
            )

            y_true, y_pred, _ = fit_and_predict_model(
                df=df,
                config=config,
                train_indices=inner_train_indices,
                evaluation_indices=inner_validation_indices,
                model_name=model_name,
                parameters=parameters,
                seed=model_seed,
            )

            inner_truth.append(y_true)
            inner_predictions.append(y_pred)

        pooled_truth = np.concatenate(inner_truth)
        pooled_predictions = np.concatenate(inner_predictions)
        inner_metrics = calculate_metrics(
            pooled_truth,
            pooled_predictions,
        )

        simple_key = parameter_sort_key(parameters)
        selection_key = (
            float(inner_metrics["macro_f1"]),
            float(inner_metrics["balanced_accuracy"]),
            *simple_key,
        )

        tuning_rows.append(
            {
                "dataset": config.name,
                "repetition": repetition,
                "outer_seed": outer_seed,
                "outer_fold": outer_fold,
                "model": model_name,
                "parameter_index": parameter_index,
                "ngram_range": (
                    f"{parameters['ngram_range'][0]}-"
                    f"{parameters['ngram_range'][1]}"
                ),
                "min_df": int(parameters["min_df"]),
                "C": float(parameters["C"]),
                "affect_weight": (
                    float(parameters["affect_weight"])
                    if "affect_weight" in parameters
                    else np.nan
                ),
                "inner_macro_f1": inner_metrics["macro_f1"],
                "inner_balanced_accuracy": (
                    inner_metrics["balanced_accuracy"]
                ),
                "selected": False,
            }
        )

        if (
            best_selection_key is None
            or selection_key > best_selection_key
        ):
            best_selection_key = selection_key
            best_parameters = parameters.copy()
            best_parameter_index = parameter_index

    if (
        best_parameters is None
        or best_parameter_index is None
    ):
        raise RuntimeError(
            f"No hyperparameters selected for {config.name}, "
            f"{model_name}, repetition={repetition}, fold={outer_fold}."
        )

    for row in tuning_rows:
        row["selected"] = (
            row["parameter_index"]
            == best_parameter_index
        )

    return best_parameters, tuning_rows


def run_nested_dataset(
    config: DatasetConfig,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Run nested repeated cross-validation for one dataset."""

    df = load_and_validate_dataset(config)

    fold_metric_rows: list[dict[str, object]] = []
    repetition_metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    selected_parameter_rows: list[dict[str, object]] = []
    tuning_score_rows: list[dict[str, object]] = []

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

        repetition_probabilities = {
            "text_only": np.full(len(df), np.nan, dtype=float),
            "emotion_aware": np.full(len(df), np.nan, dtype=float),
        }

        for outer_fold, (
            outer_train_indices,
            outer_test_indices,
        ) in enumerate(outer_splits, start=1):
            inner_seed = outer_seed + outer_fold * 10_000

            # The same inner folds are used to tune both model variants.
            inner_splits = create_inner_pair_splits(
                df=df,
                outer_train_indices=outer_train_indices,
                pair_column=config.pair_column,
                seed=inner_seed,
            )

            for model_offset, model_name in enumerate(
                ["text_only", "emotion_aware"],
                start=1,
            ):
                best_parameters, model_tuning_rows = (
                    tune_model_on_outer_training_data(
                        df=df,
                        config=config,
                        inner_splits=inner_splits,
                        model_name=model_name,
                        repetition=repetition,
                        outer_seed=outer_seed,
                        outer_fold=outer_fold,
                    )
                )
                tuning_score_rows.extend(model_tuning_rows)

                selected_parameter_rows.append(
                    {
                        "dataset": config.name,
                        "repetition": repetition,
                        "outer_seed": outer_seed,
                        "outer_fold": outer_fold,
                        "model": model_name,
                        "ngram_range": (
                            f"{best_parameters['ngram_range'][0]}-"
                            f"{best_parameters['ngram_range'][1]}"
                        ),
                        "min_df": int(best_parameters["min_df"]),
                        "C": float(best_parameters["C"]),
                        "affect_weight": (
                            float(best_parameters["affect_weight"])
                            if "affect_weight" in best_parameters
                            else np.nan
                        ),
                    }
                )

                final_model_seed = (
                    outer_seed
                    + outer_fold * 1_000
                    + model_offset
                )

                y_true, y_pred, y_probability = (
                    fit_and_predict_model(
                        df=df,
                        config=config,
                        train_indices=outer_train_indices,
                        evaluation_indices=outer_test_indices,
                        model_name=model_name,
                        parameters=best_parameters,
                        seed=final_model_seed,
                    )
                )

                fold_metrics = calculate_metrics(y_true, y_pred)

                fold_metric_rows.append(
                    {
                        "dataset": config.name,
                        "repetition": repetition,
                        "outer_seed": outer_seed,
                        "outer_fold": outer_fold,
                        "model": model_name,
                        "train_rows": len(outer_train_indices),
                        "test_rows": len(outer_test_indices),
                        "ngram_range": (
                            f"{best_parameters['ngram_range'][0]}-"
                            f"{best_parameters['ngram_range'][1]}"
                        ),
                        "min_df": int(best_parameters["min_df"]),
                        "C": float(best_parameters["C"]),
                        "affect_weight": (
                            float(best_parameters["affect_weight"])
                            if "affect_weight" in best_parameters
                            else np.nan
                        ),
                        **fold_metrics,
                    }
                )

                repetition_predictions[model_name][
                    outer_test_indices
                ] = y_pred

                repetition_probabilities[model_name][
                    outer_test_indices
                ] = y_probability

                for position, row_index in enumerate(
                    outer_test_indices
                ):
                    prediction_rows.append(
                        {
                            "dataset": config.name,
                            "repetition": repetition,
                            "outer_seed": outer_seed,
                            "outer_fold": outer_fold,
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
                            "ngram_range": (
                                f"{best_parameters['ngram_range'][0]}-"
                                f"{best_parameters['ngram_range'][1]}"
                            ),
                            "min_df": int(best_parameters["min_df"]),
                            "C": float(best_parameters["C"]),
                            "affect_weight": (
                                float(best_parameters["affect_weight"])
                                if "affect_weight" in best_parameters
                                else np.nan
                            ),
                        }
                    )

            print(
                f"{config.name}: repetition {repetition}/"
                f"{len(SEEDS)}, outer fold {outer_fold}/5 complete"
            )

        for model_name in ["text_only", "emotion_aware"]:
            predictions = repetition_predictions[model_name]
            probabilities = repetition_probabilities[model_name]

            if np.any(predictions == -1):
                raise RuntimeError(
                    f"Incomplete nested OOF predictions for "
                    f"{config.name}, seed={outer_seed}, "
                    f"model={model_name}."
                )

            if np.isnan(probabilities).any():
                raise RuntimeError(
                    f"Incomplete nested OOF probabilities for "
                    f"{config.name}, seed={outer_seed}, "
                    f"model={model_name}."
                )

            repetition_metrics = calculate_metrics(
                df["target"].to_numpy(),
                predictions,
            )

            repetition_metric_rows.append(
                {
                    "dataset": config.name,
                    "repetition": repetition,
                    "seed": outer_seed,
                    "model": model_name,
                    **repetition_metrics,
                }
            )

        print(
            f"{config.name}: completed nested repetition "
            f"{repetition}/{len(SEEDS)}"
        )

    return (
        pd.DataFrame(fold_metric_rows),
        pd.DataFrame(repetition_metric_rows),
        pd.DataFrame(prediction_rows),
        pd.DataFrame(selected_parameter_rows),
        pd.DataFrame(tuning_score_rows),
    )


def main() -> None:
    """Run the complete tuned nested-CV RQ2 experiment."""

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

    all_fold_metrics: list[pd.DataFrame] = []
    all_repetition_metrics: list[pd.DataFrame] = []
    all_predictions: list[pd.DataFrame] = []
    all_selected_parameters: list[pd.DataFrame] = []
    all_tuning_scores: list[pd.DataFrame] = []

    for config in DATASETS:
        (
            fold_metrics,
            repetition_metrics,
            predictions,
            selected_parameters,
            tuning_scores,
        ) = run_nested_dataset(config)

        all_fold_metrics.append(fold_metrics)
        all_repetition_metrics.append(repetition_metrics)
        all_predictions.append(predictions)
        all_selected_parameters.append(selected_parameters)
        all_tuning_scores.append(tuning_scores)

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
    combined_selected_parameters = pd.concat(
        all_selected_parameters,
        ignore_index=True,
    )
    combined_tuning_scores = pd.concat(
        all_tuning_scores,
        ignore_index=True,
    )

    statistical_summary = create_statistical_summary(
        combined_repetition_metrics
    )

    combined_fold_metrics.to_csv(
        TABLE_DIR / "rq2_nested_weighted_fold_metrics.csv",
        index=False,
    )
    combined_repetition_metrics.to_csv(
        TABLE_DIR / "rq2_nested_weighted_repetition_metrics.csv",
        index=False,
    )
    statistical_summary.to_csv(
        TABLE_DIR / "rq2_nested_weighted_statistical_comparison.csv",
        index=False,
    )
    combined_selected_parameters.to_csv(
        TABLE_DIR / "rq2_nested_weighted_selected_hyperparameters.csv",
        index=False,
    )
    combined_tuning_scores.to_csv(
        TABLE_DIR / "rq2_nested_weighted_inner_tuning_scores.csv",
        index=False,
    )
    combined_predictions.to_csv(
        PREDICTION_DIR / "rq2_nested_weighted_oof_predictions.csv",
        index=False,
    )

    print("\nWeighted nested RQ2 experiment complete.")
    print(statistical_summary.to_string(index=False))


if __name__ == "__main__":
    main()
