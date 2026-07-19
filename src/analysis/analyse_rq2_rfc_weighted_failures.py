"""Analyse where weighted emotion-aware models fail on RFC-Bench.

RFC-Bench contains only misleading examples. A failure therefore means that
the model predicted an RFC example as genuine.

The analysis compares matched text-only and emotion-aware predictions for the
same training dataset, repetition seed, and RFC example.
"""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RFC_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "rfc_bench_with_affective_features_512.csv"
)

PREDICTION_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "predictions"
    / "rq2_rfc_weighted_external_predictions.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
)

AFFECT_COLUMNS = [
    "affect_anger",
    "affect_fear",
    "affect_joy",
    "affect_sadness",
    "affect_valence",
]

PAIR_KEYS = [
    "training_dataset",
    "repetition",
    "seed",
    "rfc_id",
    "rfc_row_index",
    "subset",
    "manipulation_category",
]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate RFC texts and weighted model predictions."""

    if not RFC_PATH.exists():
        raise FileNotFoundError(
            f"Corrected RFC file not found: {RFC_PATH}"
        )

    if not PREDICTION_PATH.exists():
        raise FileNotFoundError(
            f"Weighted prediction file not found: {PREDICTION_PATH}"
        )

    rfc = pd.read_csv(RFC_PATH)
    predictions = pd.read_csv(PREDICTION_PATH)

    required_rfc_columns = [
        "rfc_id",
        "ticker",
        "date",
        "title",
        "link",
        "text",
        "subset",
        "manipulation_category",
        *AFFECT_COLUMNS,
    ]

    required_prediction_columns = [
        *PAIR_KEYS,
        "model",
        "y_true",
        "y_pred",
        "probability_misleading",
        "affect_weight",
    ]

    missing_rfc = [
        column
        for column in required_rfc_columns
        if column not in rfc.columns
    ]

    missing_predictions = [
        column
        for column in required_prediction_columns
        if column not in predictions.columns
    ]

    if missing_rfc:
        raise ValueError(
            f"RFC file is missing columns: {missing_rfc}"
        )

    if missing_predictions:
        raise ValueError(
            "Prediction file is missing columns: "
            f"{missing_predictions}"
        )

    if not rfc["rfc_id"].is_unique:
        raise ValueError(
            "RFC identifiers must be unique."
        )

    if set(predictions["model"].unique()) != {
        "text_only",
        "emotion_aware",
    }:
        raise ValueError(
            "Prediction file must contain both model variants."
        )

    if set(predictions["y_true"].unique()) != {1}:
        raise ValueError(
            "RFC predictions must contain only misleading targets."
        )

    duplicate_count = predictions.duplicated(
        subset=[*PAIR_KEYS, "model"]
    ).sum()

    if duplicate_count:
        raise ValueError(
            f"Found {duplicate_count} duplicate prediction rows."
        )

    return rfc, predictions


def create_matched_predictions(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Pair text-only and emotion-aware predictions exactly."""

    text = predictions.loc[
        predictions["model"] == "text_only",
        [
            *PAIR_KEYS,
            "y_pred",
            "probability_misleading",
        ],
    ].rename(
        columns={
            "y_pred": "text_y_pred",
            "probability_misleading": (
                "text_probability_misleading"
            ),
        }
    )

    emotion = predictions.loc[
        predictions["model"] == "emotion_aware",
        [
            *PAIR_KEYS,
            "y_pred",
            "probability_misleading",
            "affect_weight",
        ],
    ].rename(
        columns={
            "y_pred": "emotion_y_pred",
            "probability_misleading": (
                "emotion_probability_misleading"
            ),
            "affect_weight": "selected_affect_weight",
        }
    )

    paired = text.merge(
        emotion,
        on=PAIR_KEYS,
        how="inner",
        validate="one_to_one",
    )

    expected_pairs = len(predictions) // 2

    if len(paired) != expected_pairs:
        raise RuntimeError(
            f"Expected {expected_pairs} paired predictions, "
            f"but created {len(paired)}."
        )

    paired["text_correct"] = (
        paired["text_y_pred"] == 1
    ).astype(int)

    paired["emotion_correct"] = (
        paired["emotion_y_pred"] == 1
    ).astype(int)

    paired["text_correct_emotion_wrong"] = (
        (paired["text_y_pred"] == 1)
        & (paired["emotion_y_pred"] == 0)
    ).astype(int)

    paired["emotion_correct_text_wrong"] = (
        (paired["emotion_y_pred"] == 1)
        & (paired["text_y_pred"] == 0)
    ).astype(int)

    paired["probability_change_emotion_minus_text"] = (
        paired["emotion_probability_misleading"]
        - paired["text_probability_misleading"]
    )

    return paired


def create_example_summary(
    paired: pd.DataFrame,
    rfc: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise matched outcomes for every individual RFC example."""

    group_columns = [
        "training_dataset",
        "rfc_id",
        "rfc_row_index",
        "subset",
        "manipulation_category",
    ]

    summary = (
        paired.groupby(
            group_columns,
            as_index=False,
        )
        .agg(
            repetitions=("seed", "nunique"),
            text_correct_count=("text_correct", "sum"),
            emotion_correct_count=("emotion_correct", "sum"),
            text_correct_emotion_wrong_count=(
                "text_correct_emotion_wrong",
                "sum",
            ),
            emotion_correct_text_wrong_count=(
                "emotion_correct_text_wrong",
                "sum",
            ),
            text_probability_mean=(
                "text_probability_misleading",
                "mean",
            ),
            emotion_probability_mean=(
                "emotion_probability_misleading",
                "mean",
            ),
            probability_change_mean=(
                "probability_change_emotion_minus_text",
                "mean",
            ),
            probability_change_median=(
                "probability_change_emotion_minus_text",
                "median",
            ),
            selected_affect_weight_mean=(
                "selected_affect_weight",
                "mean",
            ),
            selected_affect_weight_median=(
                "selected_affect_weight",
                "median",
            ),
        )
    )

    count_columns = [
        "repetitions",
        "text_correct_count",
        "emotion_correct_count",
        "text_correct_emotion_wrong_count",
        "emotion_correct_text_wrong_count",
    ]

    summary[count_columns] = (
        summary[count_columns].astype(int)
    )

    summary["text_correct_rate"] = (
        summary["text_correct_count"]
        / summary["repetitions"]
    )

    summary["emotion_correct_rate"] = (
        summary["emotion_correct_count"]
        / summary["repetitions"]
    )

    summary["recall_change_emotion_minus_text"] = (
        summary["emotion_correct_rate"]
        - summary["text_correct_rate"]
    )

    summary["persistent_text_success_emotion_failure"] = (
        summary["text_correct_emotion_wrong_count"]
        >= np.ceil(
            summary["repetitions"] * 0.60
        )
    )

    summary["unanimous_text_success_emotion_failure"] = (
        summary["text_correct_emotion_wrong_count"]
        == summary["repetitions"]
    )

    metadata_columns = [
        "rfc_id",
        "ticker",
        "date",
        "title",
        "link",
        "text",
        *AFFECT_COLUMNS,
    ]

    summary = summary.merge(
        rfc[metadata_columns],
        on="rfc_id",
        how="left",
        validate="many_to_one",
    )

    return summary


def create_category_summary(
    paired: pd.DataFrame,
    example_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise failures by training dataset and manipulation category."""

    category_columns = [
        "training_dataset",
        "manipulation_category",
    ]

    category_summary = (
        paired.groupby(
            category_columns,
            as_index=False,
        )
        .agg(
            examples=("rfc_id", "nunique"),
            paired_predictions=("rfc_id", "size"),
            text_recall=("text_correct", "mean"),
            emotion_recall=("emotion_correct", "mean"),
            text_correct_emotion_wrong_predictions=(
                "text_correct_emotion_wrong",
                "sum",
            ),
            emotion_correct_text_wrong_predictions=(
                "emotion_correct_text_wrong",
                "sum",
            ),
            text_probability_mean=(
                "text_probability_misleading",
                "mean",
            ),
            emotion_probability_mean=(
                "emotion_probability_misleading",
                "mean",
            ),
            probability_change_mean=(
                "probability_change_emotion_minus_text",
                "mean",
            ),
        )
    )

    category_summary[
        "recall_change_emotion_minus_text"
    ] = (
        category_summary["emotion_recall"]
        - category_summary["text_recall"]
    )

    persistent_counts = (
        example_summary.groupby(
            category_columns,
            as_index=False,
        )
        .agg(
            persistent_failure_examples=(
                "persistent_text_success_emotion_failure",
                "sum",
            ),
            unanimous_failure_examples=(
                "unanimous_text_success_emotion_failure",
                "sum",
            ),
        )
    )

    category_summary = category_summary.merge(
        persistent_counts,
        on=category_columns,
        how="left",
        validate="one_to_one",
    )

    return category_summary


def create_review_sample(
    example_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Select 24 representative failures for qualitative review."""

    failures = example_summary.loc[
        example_summary[
            "text_correct_emotion_wrong_count"
        ] > 0
    ].copy()

    failures = failures.sort_values(
        [
            "training_dataset",
            "manipulation_category",
            "text_correct_emotion_wrong_count",
            "probability_change_mean",
            "text_correct_count",
        ],
        ascending=[
            True,
            True,
            False,
            True,
            False,
        ],
    )

    review_sample = (
        failures.groupby(
            [
                "training_dataset",
                "manipulation_category",
            ],
            group_keys=False,
        )
        .head(3)
        .copy()
    )

    review_sample[
        "review_rank_within_category"
    ] = (
        review_sample.groupby(
            [
                "training_dataset",
                "manipulation_category",
            ]
        )
        .cumcount()
        + 1
    )

    preferred_columns = [
        "training_dataset",
        "manipulation_category",
        "review_rank_within_category",
        "rfc_id",
        "subset",
        "ticker",
        "date",
        "title",
        "text",
        "link",
        "repetitions",
        "text_correct_count",
        "emotion_correct_count",
        "text_correct_emotion_wrong_count",
        "text_correct_rate",
        "emotion_correct_rate",
        "recall_change_emotion_minus_text",
        "text_probability_mean",
        "emotion_probability_mean",
        "probability_change_mean",
        "selected_affect_weight_mean",
        *AFFECT_COLUMNS,
    ]

    return review_sample[preferred_columns]


def main() -> None:
    """Run the complete RFC failure analysis."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rfc, predictions = load_data()

    paired = create_matched_predictions(
        predictions
    )

    example_summary = create_example_summary(
        paired=paired,
        rfc=rfc,
    )

    category_summary = create_category_summary(
        paired=paired,
        example_summary=example_summary,
    )

    review_sample = create_review_sample(
        example_summary
    )

    example_path = (
        OUTPUT_DIR
        / "rq2_rfc_weighted_failure_analysis_by_example.csv"
    )

    category_path = (
        OUTPUT_DIR
        / "rq2_rfc_weighted_failure_analysis_by_category.csv"
    )

    review_path = (
        OUTPUT_DIR
        / "rq2_rfc_weighted_failure_examples_for_review.csv"
    )

    example_summary.to_csv(
        example_path,
        index=False,
    )

    category_summary.to_csv(
        category_path,
        index=False,
    )

    review_sample.to_csv(
        review_path,
        index=False,
    )

    print("\nRFC weighted failure analysis complete.")

    print("\nCategory summary:")
    print(
        category_summary[
            [
                "training_dataset",
                "manipulation_category",
                "examples",
                "text_recall",
                "emotion_recall",
                "recall_change_emotion_minus_text",
                "persistent_failure_examples",
                "unanimous_failure_examples",
                "probability_change_mean",
            ]
        ]
        .sort_values(
            [
                "training_dataset",
                "recall_change_emotion_minus_text",
            ]
        )
        .to_string(index=False)
    )

    print("\nReview examples selected:")
    print(len(review_sample))

    print("\nSaved:")
    print(example_path)
    print(category_path)
    print(review_path)


if __name__ == "__main__":
    main()
