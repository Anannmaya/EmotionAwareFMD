"""Evaluate the upgraded classical RQ2 systems on RFC-Bench.

For each source dataset, repetition seed, and model variant:

1. Tune Logistic Regression versus Linear SVM, word versus word+character
   TF-IDF, regularisation, and (for the emotion-aware model) affect weight
   using three-fold pair-preserving cross-validation on the complete source
   dataset.
2. Fit the selected model on the complete source dataset.
3. Evaluate on the untouched positive-only RFC-Bench stress test.

RFC-Bench contains only manipulated/misleading examples. The principal metric
is therefore misleading-class recall (the percentage of manipulations detected),
rather than accuracy, macro-F1, or balanced accuracy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from scipy.stats import wilcoxon
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from src.models.run_rq2_classical_nested_model_selection import (
    CHAR_SETTINGS,
    WORD_SETTINGS,
    Candidate,
    create_classifier,
    create_inner_pair_splits,
    prepare_split_features,
    tune_model_variant,
)
from src.models.run_rq2_cross_validation import (
    AFFECT_COLUMNS,
    DATASETS,
    PREDICTION_DIR,
    SEEDS,
    TABLE_DIR,
    DatasetConfig,
    calculate_rank_biserial,
    holm_adjust,
    load_and_validate_dataset,
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
MODEL_VARIANTS = ["text_only", "emotion_aware"]


def load_and_validate_rfc() -> pd.DataFrame:
    """Load the positive-only RFC external stress-test dataset."""

    if not RFC_PATH.exists():
        raise FileNotFoundError(f"RFC feature dataset not found: {RFC_PATH}")

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
        column for column in required_columns if column not in rfc_df.columns
    ]
    if missing_columns:
        raise ValueError(f"RFC dataset is missing columns: {missing_columns}")

    if rfc_df[required_columns].isna().any().any():
        missing_counts = rfc_df[required_columns].isna().sum()
        missing_counts = missing_counts[missing_counts > 0]
        raise ValueError(
            "RFC dataset contains missing required values:\n"
            f"{missing_counts}"
        )

    if not rfc_df[RFC_ID_COLUMN].is_unique:
        raise ValueError("RFC identifiers are not unique.")

    if set(rfc_df["target"].astype(int).unique()) != {1}:
        raise ValueError("RFC must contain only positive misleading examples.")

    if set(rfc_df["label"].astype(str).unique()) != {"misleading"}:
        raise ValueError(
            "RFC label column must contain only 'misleading'."
        )

    rfc_df = rfc_df.copy()
    rfc_df[RFC_TEXT_COLUMN] = rfc_df[RFC_TEXT_COLUMN].astype(str)

    for column in AFFECT_COLUMNS:
        rfc_df[column] = pd.to_numeric(rfc_df[column], errors="raise")
        if not rfc_df[column].between(0, 1).all():
            raise ValueError(f"{column} contains values outside 0-1.")

    print(
        f"RFC-Bench: {len(rfc_df):,} manipulated examples, "
        f"{rfc_df['manipulation_category'].nunique()} categories"
    )

    return rfc_df.reset_index(drop=True)


def tune_on_complete_source_dataset(
    source_df: pd.DataFrame,
    config: DatasetConfig,
    model_variant: str,
    repetition: int,
    seed: int,
) -> tuple[Candidate, list[dict[str, object]]]:
    """Tune one final system using source data only."""

    all_indices = np.arange(len(source_df), dtype=int)
    inner_splits = create_inner_pair_splits(
        df=source_df,
        outer_train_indices=all_indices,
        pair_column=config.pair_column,
        seed=seed,
    )

    inner_feature_sets = [
        prepare_split_features(
            df=source_df,
            config=config,
            train_indices=train_indices,
            evaluation_indices=validation_indices,
        )
        for train_indices, validation_indices in inner_splits
    ]

    best_candidate, tuning_rows = tune_model_variant(
        inner_feature_sets=inner_feature_sets,
        model_variant=model_variant,
        dataset_name=config.name,
        repetition=repetition,
        outer_seed=seed,
        outer_fold=0,
    )

    for row in tuning_rows:
        row["evaluation_stage"] = "rfc_complete_source_tuning"

    return best_candidate, tuning_rows


def fit_complete_model_and_predict_rfc(
    source_df: pd.DataFrame,
    rfc_df: pd.DataFrame,
    config: DatasetConfig,
    candidate: Candidate,
    seed: int,
) -> np.ndarray:
    """Fit one selected system on all source rows and predict RFC-Bench."""

    word_vectorizer = TfidfVectorizer(**WORD_SETTINGS)
    x_source_word = word_vectorizer.fit_transform(
        source_df[config.text_column]
    )
    x_rfc_word = word_vectorizer.transform(rfc_df[RFC_TEXT_COLUMN])

    if candidate.representation == "word":
        x_source_text = x_source_word
        x_rfc_text = x_rfc_word

    elif candidate.representation == "word_char":
        char_vectorizer = TfidfVectorizer(**CHAR_SETTINGS)
        x_source_char = char_vectorizer.fit_transform(
            source_df[config.text_column]
        )
        x_rfc_char = char_vectorizer.transform(rfc_df[RFC_TEXT_COLUMN])

        x_source_text = hstack(
            [x_source_word, x_source_char],
            format="csr",
        )
        x_rfc_text = hstack(
            [x_rfc_word, x_rfc_char],
            format="csr",
        )

    else:
        raise ValueError(
            f"Unknown text representation: {candidate.representation}"
        )

    if candidate.affect_weight is None:
        x_source = x_source_text
        x_rfc = x_rfc_text

    else:
        affect_scaler = StandardScaler()
        x_source_affect = affect_scaler.fit_transform(
            source_df[AFFECT_COLUMNS]
        )
        x_rfc_affect = affect_scaler.transform(rfc_df[AFFECT_COLUMNS])

        x_source = hstack(
            [
                x_source_text,
                csr_matrix(
                    x_source_affect * float(candidate.affect_weight)
                ),
            ],
            format="csr",
        )
        x_rfc = hstack(
            [
                x_rfc_text,
                csr_matrix(x_rfc_affect * float(candidate.affect_weight)),
            ],
            format="csr",
        )

    model = create_classifier(candidate=candidate, seed=seed)
    model.fit(x_source, source_df["target"].to_numpy())
    return np.asarray(model.predict(x_rfc), dtype=int)


def create_evaluation_groups(
    rfc_df: pd.DataFrame,
) -> list[tuple[str, str, str, np.ndarray]]:
    """Create the planned overall, subset, and category reporting groups."""

    groups: list[tuple[str, str, str, np.ndarray]] = []
    all_mask = np.ones(len(rfc_df), dtype=bool)
    non_sentiment_mask = (
        rfc_df["manipulation_category"].ne("sentiment").to_numpy()
    )

    groups.append(("overall", "all", "all", all_mask))
    groups.append(
        (
            "overall_non_sentiment",
            "all",
            "non_sentiment",
            non_sentiment_mask,
        )
    )

    for subset in sorted(rfc_df["subset"].unique()):
        subset_mask = rfc_df["subset"].eq(subset).to_numpy()
        groups.append(("subset", str(subset), "all", subset_mask))
        groups.append(
            (
                "subset_non_sentiment",
                str(subset),
                "non_sentiment",
                subset_mask & non_sentiment_mask,
            )
        )

    for category in sorted(
        rfc_df["manipulation_category"].unique()
    ):
        category_mask = (
            rfc_df["manipulation_category"].eq(category).to_numpy()
        )
        groups.append(("category", "all", str(category), category_mask))

    for subset in sorted(rfc_df["subset"].unique()):
        subset_mask = rfc_df["subset"].eq(subset).to_numpy()

        for category in sorted(
            rfc_df["manipulation_category"].unique()
        ):
            category_mask = (
                rfc_df["manipulation_category"].eq(category).to_numpy()
            )
            combined_mask = subset_mask & category_mask

            if combined_mask.any():
                groups.append(
                    (
                        "subset_category",
                        str(subset),
                        str(category),
                        combined_mask,
                    )
                )

    return groups


def calculate_group_metrics(
    predictions: np.ndarray,
    groups: list[tuple[str, str, str, np.ndarray]],
) -> list[dict[str, object]]:
    """Calculate positive-only detection metrics for all reporting groups."""

    rows: list[dict[str, object]] = []

    for scope, subset, category, mask in groups:
        group_predictions = predictions[mask]
        row_count = int(mask.sum())
        detected_count = int(group_predictions.sum())

        rows.append(
            {
                "scope": scope,
                "subset": subset,
                "manipulation_category": category,
                "rows": row_count,
                "detected_misleading": detected_count,
                "missed_misleading": row_count - detected_count,
                "misleading_recall": float(group_predictions.mean()),
            }
        )

    return rows


def build_seed_summary(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarise RFC detection rates across the ten repetitions."""

    group_columns = [
        "dataset",
        "model",
        "scope",
        "subset",
        "manipulation_category",
    ]

    return (
        seed_metrics.groupby(group_columns, as_index=False)
        .agg(
            repetitions=("seed", "nunique"),
            rows=("rows", "first"),
            misleading_recall_mean=("misleading_recall", "mean"),
            misleading_recall_std=("misleading_recall", "std"),
            misleading_recall_min=("misleading_recall", "min"),
            misleading_recall_max=("misleading_recall", "max"),
        )
        .sort_values(group_columns)
        .reset_index(drop=True)
    )


def build_overall_comparison(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    """Compare overall RFC recall using paired repetition-level scores."""

    overall = seed_metrics[seed_metrics["scope"] == "overall"]
    rows: list[dict[str, object]] = []

    for dataset_name in overall["dataset"].unique():
        dataset_results = overall[
            overall["dataset"] == dataset_name
        ]

        paired = dataset_results.pivot(
            index="repetition",
            columns="model",
            values="misleading_recall",
        ).dropna()

        text_scores = paired["text_only"].to_numpy()
        emotion_scores = paired["emotion_aware"].to_numpy()
        differences = emotion_scores - text_scores

        if np.allclose(differences, 0):
            statistic = 0.0
            p_value = 1.0
        else:
            result = wilcoxon(
                emotion_scores,
                text_scores,
                alternative="two-sided",
                zero_method="wilcox",
            )
            statistic = float(result.statistic)
            p_value = float(result.pvalue)

        rows.append(
            {
                "dataset": dataset_name,
                "n_repetitions": len(paired),
                "text_only_recall_mean": float(text_scores.mean()),
                "text_only_recall_std": float(text_scores.std(ddof=1)),
                "emotion_aware_recall_mean": float(
                    emotion_scores.mean()
                ),
                "emotion_aware_recall_std": float(
                    emotion_scores.std(ddof=1)
                ),
                "mean_difference_emotion_minus_text": float(
                    differences.mean()
                ),
                "median_difference_emotion_minus_text": float(
                    np.median(differences)
                ),
                "emotion_better_repetitions": int(
                    np.sum(differences > 0)
                ),
                "text_better_repetitions": int(
                    np.sum(differences < 0)
                ),
                "tied_repetitions": int(np.sum(differences == 0)),
                "wilcoxon_statistic": statistic,
                "p_value": p_value,
                "rank_biserial": calculate_rank_biserial(differences),
            }
        )

    comparison = pd.DataFrame(rows)
    comparison["p_value_holm"] = holm_adjust(
        comparison["p_value"].tolist()
    )
    comparison["significant_holm_0_05"] = (
        comparison["p_value_holm"] < 0.05
    )
    return comparison


def main() -> None:
    """Run the upgraded classical external RFC evaluation."""

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

    rfc_df = load_and_validate_rfc()
    groups = create_evaluation_groups(rfc_df)

    seed_metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    tuning_rows: list[dict[str, object]] = []

    for config in DATASETS:
        source_df = load_and_validate_dataset(config)

        for repetition, seed in enumerate(SEEDS, start=1):
            for model_offset, model_variant in enumerate(
                MODEL_VARIANTS,
                start=1,
            ):
                best_candidate, model_tuning_rows = (
                    tune_on_complete_source_dataset(
                        source_df=source_df,
                        config=config,
                        model_variant=model_variant,
                        repetition=repetition,
                        seed=seed,
                    )
                )
                tuning_rows.extend(model_tuning_rows)

                selected_rows.append(
                    {
                        "dataset": config.name,
                        "repetition": repetition,
                        "seed": seed,
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

                predictions = fit_complete_model_and_predict_rfc(
                    source_df=source_df,
                    rfc_df=rfc_df,
                    config=config,
                    candidate=best_candidate,
                    seed=seed + model_offset,
                )

                for metric_row in calculate_group_metrics(
                    predictions=predictions,
                    groups=groups,
                ):
                    seed_metric_rows.append(
                        {
                            "dataset": config.name,
                            "repetition": repetition,
                            "seed": seed,
                            "model": model_variant,
                            "classifier": best_candidate.classifier,
                            "representation": (
                                best_candidate.representation
                            ),
                            "C": best_candidate.c_value,
                            "affect_weight": (
                                best_candidate.affect_weight
                                if best_candidate.affect_weight is not None
                                else np.nan
                            ),
                            **metric_row,
                        }
                    )

                for row_index, prediction in enumerate(predictions):
                    prediction_rows.append(
                        {
                            "dataset": config.name,
                            "repetition": repetition,
                            "seed": seed,
                            "model": model_variant,
                            "rfc_id": rfc_df.loc[
                                row_index,
                                RFC_ID_COLUMN,
                            ],
                            "subset": rfc_df.loc[row_index, "subset"],
                            "manipulation_category": rfc_df.loc[
                                row_index,
                                "manipulation_category",
                            ],
                            "y_true": 1,
                            "y_pred": int(prediction),
                            "classifier": best_candidate.classifier,
                            "representation": (
                                best_candidate.representation
                            ),
                            "C": best_candidate.c_value,
                            "affect_weight": (
                                best_candidate.affect_weight
                                if best_candidate.affect_weight is not None
                                else np.nan
                            ),
                        }
                    )

            print(
                f"{config.name}: RFC repetition {repetition}/"
                f"{len(SEEDS)} complete"
            )

    seed_metrics = pd.DataFrame(seed_metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    selected_parameters = pd.DataFrame(selected_rows)
    tuning_scores = pd.DataFrame(tuning_rows)

    summary = build_seed_summary(seed_metrics)
    overall_comparison = build_overall_comparison(seed_metrics)

    seed_metrics.to_csv(
        TABLE_DIR / "rq2_classical_rfc_seed_metrics.csv",
        index=False,
    )
    summary.to_csv(
        TABLE_DIR / "rq2_classical_rfc_summary.csv",
        index=False,
    )
    overall_comparison.to_csv(
        TABLE_DIR / "rq2_classical_rfc_overall_comparison.csv",
        index=False,
    )
    selected_parameters.to_csv(
        TABLE_DIR / "rq2_classical_rfc_selected_hyperparameters.csv",
        index=False,
    )
    tuning_scores.to_csv(
        TABLE_DIR / "rq2_classical_rfc_inner_tuning_scores.csv",
        index=False,
    )
    predictions.to_csv(
        PREDICTION_DIR / "rq2_classical_rfc_predictions.csv",
        index=False,
    )

    print("\nUpgraded classical RFC evaluation complete.")
    print(overall_comparison.to_string(index=False))


if __name__ == "__main__":
    main()
