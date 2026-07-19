"""Evaluate final RQ2 models on the positive-only RFC-Bench stress test.

For each training dataset, repetition seed, and model variant:

1. Tune hyperparameters using three-fold pair-preserving CV on the complete
   training dataset.
2. Fit the selected model on the complete training dataset.
3. Evaluate it on RFC-Bench without using RFC data for training or tuning.

RFC-Bench contains only manipulated/misleading examples. Therefore, this
script reports misleading-class recall (detection rate) and predicted
misinformation confidence rather than accuracy, macro-F1, or balanced
accuracy.

Results are reported:
- across the complete public RFC release;
- separately for main and hard subsets;
- with sentiment manipulations excluded;
- by manipulation category;
- by subset and manipulation category.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.models.run_rq2_cross_validation import (
    AFFECT_COLUMNS,
    DATASETS,
    PREDICTION_DIR,
    SEEDS,
    TABLE_DIR,
    DatasetConfig,
    calculate_metrics,
    load_and_validate_dataset,
)
from src.models.run_rq2_nested_weighted_cross_validation import (
    BASE_LOGISTIC_SETTINGS,
    BASE_TFIDF_SETTINGS,
    create_inner_pair_splits,
    fit_and_predict_model,
    get_parameter_grid,
    parameter_sort_key,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RFC_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "rfc_bench_with_affective_features_512.csv"
)

RFC_ID_COLUMN = "rfc_id"
RFC_TEXT_COLUMN = "text"

MODEL_NAMES = [
    "text_only",
    "emotion_aware",
]


def load_and_validate_rfc() -> pd.DataFrame:
    """Load and validate the positive-only RFC external test set."""

    if not RFC_PATH.exists():
        raise FileNotFoundError(
            f"RFC feature dataset not found: {RFC_PATH}"
        )

    rfc_df = pd.read_csv(RFC_PATH)

    required_columns = [
        RFC_ID_COLUMN,
        RFC_TEXT_COLUMN,
        "subset",
        "manipulation_category",
        "target",
        "label",
        *AFFECT_COLUMNS,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in rfc_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"RFC dataset is missing columns: {missing_columns}"
        )

    if rfc_df[required_columns].isna().any().any():
        missing_counts = (
            rfc_df[required_columns]
            .isna()
            .sum()
        )
        missing_counts = missing_counts[
            missing_counts > 0
        ]

        raise ValueError(
            "RFC dataset contains missing required values:\n"
            f"{missing_counts}"
        )

    if not rfc_df[RFC_ID_COLUMN].is_unique:
        raise ValueError(
            "RFC identifiers are not unique."
        )

    if set(rfc_df["target"].unique()) != {1}:
        raise ValueError(
            "RFC must contain only positive misleading examples."
        )

    if set(rfc_df["label"].unique()) != {"misleading"}:
        raise ValueError(
            "RFC label column must contain only 'misleading'."
        )

    for column in AFFECT_COLUMNS:
        if not rfc_df[column].between(0, 1).all():
            raise ValueError(
                f"{column} contains values outside 0–1."
            )

    return rfc_df.reset_index(drop=True)


def tune_on_complete_training_dataset(
    df: pd.DataFrame,
    config: DatasetConfig,
    model_name: str,
    repetition: int,
    seed: int,
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
]:
    """Tune one model using pair-preserving CV on all training rows."""

    all_indices = np.arange(
        len(df),
        dtype=int,
    )

    inner_splits = create_inner_pair_splits(
        df=df,
        outer_train_indices=all_indices,
        pair_column=config.pair_column,
        seed=seed,
    )

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
        fold_truth: list[np.ndarray] = []
        fold_predictions: list[np.ndarray] = []

        for inner_fold, (
            train_indices,
            validation_indices,
        ) in enumerate(
            inner_splits,
            start=1,
        ):
            model_seed = (
                seed
                + inner_fold * 100
                + parameter_index
            )

            y_true, y_pred, _ = fit_and_predict_model(
                df=df,
                config=config,
                train_indices=train_indices,
                evaluation_indices=validation_indices,
                model_name=model_name,
                parameters=parameters,
                seed=model_seed,
            )

            fold_truth.append(y_true)
            fold_predictions.append(y_pred)

        pooled_truth = np.concatenate(fold_truth)
        pooled_predictions = np.concatenate(
            fold_predictions
        )

        metrics = calculate_metrics(
            pooled_truth,
            pooled_predictions,
        )

        simplicity_key = parameter_sort_key(
            parameters
        )

        selection_key = (
            float(metrics["macro_f1"]),
            float(metrics["balanced_accuracy"]),
            *simplicity_key,
        )

        tuning_rows.append(
            {
                "dataset": config.name,
                "repetition": repetition,
                "seed": seed,
                "model": model_name,
                "parameter_index": parameter_index,
                "ngram_range": (
                    f"{parameters['ngram_range'][0]}-"
                    f"{parameters['ngram_range'][1]}"
                ),
                "min_df": int(
                    parameters["min_df"]
                ),
                "C": float(parameters["C"]),
                "affect_weight": (
                    float(parameters["affect_weight"])
                    if model_name == "emotion_aware"
                    else np.nan
                ),
                "inner_macro_f1": float(
                    metrics["macro_f1"]
                ),
                "inner_balanced_accuracy": float(
                    metrics[
                        "balanced_accuracy"
                    ]
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
            "No hyperparameters were selected for "
            f"{config.name}, {model_name}, seed={seed}."
        )

    for row in tuning_rows:
        row["selected"] = (
            row["parameter_index"]
            == best_parameter_index
        )

    return best_parameters, tuning_rows


def fit_complete_model_and_predict_rfc(
    train_df: pd.DataFrame,
    rfc_df: pd.DataFrame,
    config: DatasetConfig,
    model_name: str,
    parameters: dict[str, object],
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit on one full training dataset and predict RFC-Bench."""

    vectorizer = TfidfVectorizer(
        ngram_range=parameters["ngram_range"],
        min_df=int(parameters["min_df"]),
        **BASE_TFIDF_SETTINGS,
    )

    x_train_text = vectorizer.fit_transform(
        train_df[config.text_column]
    )

    x_rfc_text = vectorizer.transform(
        rfc_df[RFC_TEXT_COLUMN]
    )

    if model_name == "text_only":
        x_train = x_train_text
        x_rfc = x_rfc_text

    elif model_name == "emotion_aware":
        affect_scaler = StandardScaler()
        affect_weight = float(
            parameters["affect_weight"]
        )

        x_train_affect = (
            affect_scaler.fit_transform(
                train_df[AFFECT_COLUMNS]
            )
        )

        x_rfc_affect = (
            affect_scaler.transform(
                rfc_df[AFFECT_COLUMNS]
            )
        )

        x_train = hstack(
            [
                x_train_text,
                csr_matrix(
                    x_train_affect
                    * affect_weight
                ),
            ],
            format="csr",
        )

        x_rfc = hstack(
            [
                x_rfc_text,
                csr_matrix(
                    x_rfc_affect
                    * affect_weight
                ),
            ],
            format="csr",
        )

    else:
        raise ValueError(
            f"Unknown model variant: {model_name}"
        )

    model = LogisticRegression(
        C=float(parameters["C"]),
        random_state=seed,
        **BASE_LOGISTIC_SETTINGS,
    )

    model.fit(
        x_train,
        train_df["target"].to_numpy(),
    )

    predictions = model.predict(x_rfc)

    probabilities = (
        model.predict_proba(x_rfc)[:, 1]
    )

    return predictions, probabilities


def create_evaluation_groups(
    rfc_df: pd.DataFrame,
) -> list[
    tuple[str, str, str, np.ndarray]
]:
    """Create masks for all planned RFC reporting groups."""

    groups: list[
        tuple[str, str, str, np.ndarray]
    ] = []

    all_mask = np.ones(
        len(rfc_df),
        dtype=bool,
    )

    non_sentiment_mask = (
        rfc_df["manipulation_category"]
        .ne("sentiment")
        .to_numpy()
    )

    groups.append(
        (
            "overall",
            "all",
            "all",
            all_mask,
        )
    )

    groups.append(
        (
            "overall_non_sentiment",
            "all",
            "non_sentiment",
            non_sentiment_mask,
        )
    )

    subsets = sorted(
        rfc_df["subset"].unique()
    )

    categories = sorted(
        rfc_df[
            "manipulation_category"
        ].unique()
    )

    for subset in subsets:
        subset_mask = (
            rfc_df["subset"]
            .eq(subset)
            .to_numpy()
        )

        groups.append(
            (
                "subset",
                subset,
                "all",
                subset_mask,
            )
        )

        groups.append(
            (
                "subset_non_sentiment",
                subset,
                "non_sentiment",
                subset_mask
                & non_sentiment_mask,
            )
        )

    for category in categories:
        category_mask = (
            rfc_df[
                "manipulation_category"
            ]
            .eq(category)
            .to_numpy()
        )

        groups.append(
            (
                "category",
                "all",
                category,
                category_mask,
            )
        )

    for subset in subsets:
        subset_mask = (
            rfc_df["subset"]
            .eq(subset)
            .to_numpy()
        )

        for category in categories:
            category_mask = (
                rfc_df[
                    "manipulation_category"
                ]
                .eq(category)
                .to_numpy()
            )

            combined_mask = (
                subset_mask
                & category_mask
            )

            if combined_mask.any():
                groups.append(
                    (
                        "subset_category",
                        subset,
                        category,
                        combined_mask,
                    )
                )

    return groups


def calculate_external_metrics(
    predictions: np.ndarray,
    probabilities: np.ndarray,
    groups: list[
        tuple[str, str, str, np.ndarray]
    ],
) -> list[dict[str, object]]:
    """Calculate positive-only RFC detection metrics."""

    metric_rows: list[dict[str, object]] = []

    for (
        scope,
        subset,
        category,
        mask,
    ) in groups:
        group_predictions = predictions[mask]
        group_probabilities = probabilities[mask]

        row_count = int(mask.sum())
        detected_count = int(
            group_predictions.sum()
        )

        metric_rows.append(
            {
                "scope": scope,
                "subset": subset,
                "manipulation_category": category,
                "rows": row_count,
                "detected_misleading": (
                    detected_count
                ),
                "missed_misleading": (
                    row_count
                    - detected_count
                ),
                "misleading_recall": float(
                    group_predictions.mean()
                ),
                "mean_probability_misleading": (
                    float(
                        group_probabilities.mean()
                    )
                ),
                "median_probability_misleading": (
                    float(
                        np.median(
                            group_probabilities
                        )
                    )
                ),
                "std_probability_misleading": (
                    float(
                        group_probabilities.std(
                            ddof=0
                        )
                    )
                ),
            }
        )

    return metric_rows


def build_seed_summary(
    seed_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise RFC metrics across the ten repetitions."""

    group_columns = [
        "dataset",
        "model",
        "scope",
        "subset",
        "manipulation_category",
    ]

    summary = (
        seed_metrics
        .groupby(
            group_columns,
            as_index=False,
        )
        .agg(
            repetitions=(
                "seed",
                "nunique",
            ),
            rows=(
                "rows",
                "first",
            ),
            misleading_recall_mean=(
                "misleading_recall",
                "mean",
            ),
            misleading_recall_std=(
                "misleading_recall",
                "std",
            ),
            misleading_recall_min=(
                "misleading_recall",
                "min",
            ),
            misleading_recall_max=(
                "misleading_recall",
                "max",
            ),
            mean_probability_mean=(
                "mean_probability_misleading",
                "mean",
            ),
            mean_probability_std=(
                "mean_probability_misleading",
                "std",
            ),
            median_probability_mean=(
                "median_probability_misleading",
                "mean",
            ),
        )
    )

    return summary


def build_paired_differences(
    seed_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Compare emotion-aware and text-only models by matched seed."""

    group_columns = [
        "dataset",
        "scope",
        "subset",
        "manipulation_category",
    ]

    comparison_rows: list[
        dict[str, object]
    ] = []

    for group_values, group_df in (
        seed_metrics.groupby(
            group_columns,
            sort=False,
        )
    ):
        (
            dataset,
            scope,
            subset,
            category,
        ) = group_values

        for metric in [
            "misleading_recall",
            "mean_probability_misleading",
        ]:
            pivot = group_df.pivot(
                index="seed",
                columns="model",
                values=metric,
            )

            if not {
                "text_only",
                "emotion_aware",
            }.issubset(pivot.columns):
                raise RuntimeError(
                    "Both models are required for "
                    f"{dataset}, {scope}, {metric}."
                )

            differences = (
                pivot["emotion_aware"]
                - pivot["text_only"]
            )

            tolerance = 1e-12

            comparison_rows.append(
                {
                    "dataset": dataset,
                    "scope": scope,
                    "subset": subset,
                    "manipulation_category": (
                        category
                    ),
                    "metric": metric,
                    "repetitions": len(
                        differences
                    ),
                    "text_only_mean": float(
                        pivot[
                            "text_only"
                        ].mean()
                    ),
                    "emotion_aware_mean": float(
                        pivot[
                            "emotion_aware"
                        ].mean()
                    ),
                    "mean_difference": float(
                        differences.mean()
                    ),
                    "median_difference": float(
                        differences.median()
                    ),
                    "std_difference": float(
                        differences.std(
                            ddof=1
                        )
                    ),
                    "emotion_better_repetitions": (
                        int(
                            (
                                differences
                                > tolerance
                            ).sum()
                        )
                    ),
                    "text_better_repetitions": (
                        int(
                            (
                                differences
                                < -tolerance
                            ).sum()
                        )
                    ),
                    "tied_repetitions": int(
                        (
                            differences.abs()
                            <= tolerance
                        ).sum()
                    ),
                }
            )

    return pd.DataFrame(
        comparison_rows
    )


def main() -> None:
    """Run the weighted RFC external robustness evaluation."""

    TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PREDICTION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rfc_df = load_and_validate_rfc()
    evaluation_groups = (
        create_evaluation_groups(rfc_df)
    )

    selected_parameter_rows: list[
        dict[str, object]
    ] = []

    tuning_score_rows: list[
        dict[str, object]
    ] = []

    seed_metric_rows: list[
        dict[str, object]
    ] = []

    prediction_rows: list[
        dict[str, object]
    ] = []

    for config in DATASETS:
        train_df = load_and_validate_dataset(
            config
        )

        for repetition, seed in enumerate(
            SEEDS,
            start=1,
        ):
            for model_offset, model_name in enumerate(
                MODEL_NAMES,
                start=1,
            ):
                (
                    best_parameters,
                    model_tuning_rows,
                ) = (
                    tune_on_complete_training_dataset(
                        df=train_df,
                        config=config,
                        model_name=model_name,
                        repetition=repetition,
                        seed=seed,
                    )
                )

                tuning_score_rows.extend(
                    model_tuning_rows
                )

                selected_parameter_rows.append(
                    {
                        "dataset": config.name,
                        "repetition": repetition,
                        "seed": seed,
                        "model": model_name,
                        "ngram_range": (
                            f"{best_parameters['ngram_range'][0]}-"
                            f"{best_parameters['ngram_range'][1]}"
                        ),
                        "min_df": int(
                            best_parameters["min_df"]
                        ),
                        "C": float(
                            best_parameters["C"]
                        ),
                        "affect_weight": (
                            float(
                                best_parameters[
                                    "affect_weight"
                                ]
                            )
                            if model_name
                            == "emotion_aware"
                            else np.nan
                        ),
                    }
                )

                final_model_seed = (
                    seed
                    + model_offset * 10_000
                )

                (
                    predictions,
                    probabilities,
                ) = (
                    fit_complete_model_and_predict_rfc(
                        train_df=train_df,
                        rfc_df=rfc_df,
                        config=config,
                        model_name=model_name,
                        parameters=best_parameters,
                        seed=final_model_seed,
                    )
                )

                model_metric_rows = (
                    calculate_external_metrics(
                        predictions=predictions,
                        probabilities=probabilities,
                        groups=evaluation_groups,
                    )
                )

                for metric_row in (
                    model_metric_rows
                ):
                    seed_metric_rows.append(
                        {
                            "dataset": (
                                config.name
                            ),
                            "repetition": (
                                repetition
                            ),
                            "seed": seed,
                            "model": model_name,
                            "train_rows": len(
                                train_df
                            ),
                            "ngram_range": (
                                f"{best_parameters['ngram_range'][0]}-"
                                f"{best_parameters['ngram_range'][1]}"
                            ),
                            "min_df": int(
                                best_parameters[
                                    "min_df"
                                ]
                            ),
                            "C": float(
                                best_parameters[
                                    "C"
                                ]
                            ),
                            "affect_weight": (
                                float(
                                    best_parameters[
                                        "affect_weight"
                                    ]
                                )
                                if model_name
                                == "emotion_aware"
                                else np.nan
                            ),
                            **metric_row,
                        }
                    )

                for row_index in range(
                    len(rfc_df)
                ):
                    prediction_rows.append(
                        {
                            "training_dataset": (
                                config.name
                            ),
                            "repetition": (
                                repetition
                            ),
                            "seed": seed,
                            "model": model_name,
                            "rfc_row_index": (
                                row_index
                            ),
                            "rfc_id": rfc_df.loc[
                                row_index,
                                RFC_ID_COLUMN,
                            ],
                            "subset": rfc_df.loc[
                                row_index,
                                "subset",
                            ],
                            "manipulation_category": (
                                rfc_df.loc[
                                    row_index,
                                    "manipulation_category",
                                ]
                            ),
                            "y_true": 1,
                            "y_pred": int(
                                predictions[
                                    row_index
                                ]
                            ),
                            "probability_misleading": (
                                float(
                                    probabilities[
                                        row_index
                                    ]
                                )
                            ),
                            "ngram_range": (
                                f"{best_parameters['ngram_range'][0]}-"
                                f"{best_parameters['ngram_range'][1]}"
                            ),
                            "min_df": int(
                                best_parameters[
                                    "min_df"
                                ]
                            ),
                            "C": float(
                                best_parameters[
                                    "C"
                                ]
                            ),
                            "affect_weight": (
                                float(
                                    best_parameters[
                                        "affect_weight"
                                    ]
                                )
                                if model_name
                                == "emotion_aware"
                                else np.nan
                            ),
                        }
                    )

            print(
                f"{config.name}: completed "
                f"repetition {repetition}/"
                f"{len(SEEDS)}"
            )

    selected_parameters = pd.DataFrame(
        selected_parameter_rows
    )

    tuning_scores = pd.DataFrame(
        tuning_score_rows
    )

    seed_metrics = pd.DataFrame(
        seed_metric_rows
    )

    predictions = pd.DataFrame(
        prediction_rows
    )

    summary = build_seed_summary(
        seed_metrics
    )

    paired_differences = (
        build_paired_differences(
            seed_metrics
        )
    )

    selected_parameters.to_csv(
        TABLE_DIR
        / (
            "rq2_rfc_weighted_external_"
            "selected_hyperparameters.csv"
        ),
        index=False,
    )

    tuning_scores.to_csv(
        TABLE_DIR
        / (
            "rq2_rfc_weighted_external_"
            "inner_tuning_scores.csv"
        ),
        index=False,
    )

    seed_metrics.to_csv(
        TABLE_DIR
        / "rq2_rfc_weighted_external_seed_metrics.csv",
        index=False,
    )

    summary.to_csv(
        TABLE_DIR
        / "rq2_rfc_weighted_external_summary.csv",
        index=False,
    )

    paired_differences.to_csv(
        TABLE_DIR
        / (
            "rq2_rfc_weighted_external_"
            "paired_differences.csv"
        ),
        index=False,
    )

    predictions.to_csv(
        PREDICTION_DIR
        / "rq2_rfc_weighted_external_predictions.csv",
        index=False,
    )

    print("\nWeighted RFC external evaluation complete.")
    print(
        "Training datasets:",
        len(DATASETS),
    )
    print(
        "RFC rows:",
        len(rfc_df),
    )
    print(
        "Prediction rows:",
        len(predictions),
    )
    print(
        "Seed-level metric rows:",
        len(seed_metrics),
    )
    print(
        "\nResults saved under outputs/tables "
        "and outputs/predictions."
    )


if __name__ == "__main__":
    main()
