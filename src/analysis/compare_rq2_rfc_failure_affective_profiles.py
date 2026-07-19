"""Compare affective profiles of RFC failures and training classes.

The analysis compares four groups for each training dataset:

1. Genuine training examples
2. Misleading training examples
3. Persistent RFC failures:
   text-only succeeds but emotion-aware fails in at least 60% of repetitions
4. Jointly detected RFC examples:
   both models succeed in at least 60% of repetitions

The analysis is descriptive and does not prove that emotion caused individual
errors.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FAILURE_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "rq2_rfc_weighted_failure_analysis_by_example.csv"
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

SUCCESS_THRESHOLD = 0.60

TRAINING_CONFIGS = {
    "liar2_finance": {
        "path": (
            PROJECT_ROOT
            / "data"
            / "features"
            / "liar2_finance_with_affective_features.csv"
        ),
        "label_column": "binary_label_name",
    },
    "factors_finance": {
        "path": (
            PROJECT_ROOT
            / "data"
            / "features"
            / "factors_with_affective_features.csv"
        ),
        "label_column": "label",
    },
}


def holm_adjust(
    p_values: list[float],
) -> list[float]:
    """Apply Holm correction to one family of p-values."""

    p_array = np.asarray(
        p_values,
        dtype=float,
    )

    order = np.argsort(p_array)
    adjusted = np.empty_like(p_array)

    running_maximum = 0.0
    number_of_tests = len(p_array)

    for rank, original_index in enumerate(order):
        multiplier = number_of_tests - rank

        adjusted_value = min(
            1.0,
            p_array[original_index] * multiplier,
        )

        running_maximum = max(
            running_maximum,
            adjusted_value,
        )

        adjusted[original_index] = running_maximum

    return adjusted.tolist()


def load_training_data(
    dataset_name: str,
) -> pd.DataFrame:
    """Load and validate one affectively annotated training dataset."""

    config = TRAINING_CONFIGS[dataset_name]
    path = config["path"]
    label_column = config["label_column"]

    if not path.exists():
        raise FileNotFoundError(
            f"Training file not found: {path}"
        )

    df = pd.read_csv(path)

    required_columns = [
        label_column,
        *AFFECT_COLUMNS,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{dataset_name} is missing columns: "
            f"{missing_columns}"
        )

    df = df.copy()

    df["analysis_label"] = (
        df[label_column]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    observed_labels = set(
        df["analysis_label"].unique()
    )

    expected_labels = {
        "genuine",
        "misleading",
    }

    if observed_labels != expected_labels:
        raise ValueError(
            f"Unexpected labels in {dataset_name}: "
            f"{sorted(observed_labels)}"
        )

    if df[AFFECT_COLUMNS].isna().any().any():
        raise ValueError(
            f"{dataset_name} contains missing affective values."
        )

    return df


def load_rfc_example_analysis() -> pd.DataFrame:
    """Load the previously generated RFC example-level analysis."""

    if not FAILURE_PATH.exists():
        raise FileNotFoundError(
            f"RFC failure analysis not found: {FAILURE_PATH}"
        )

    df = pd.read_csv(FAILURE_PATH)

    required_columns = [
        "training_dataset",
        "rfc_id",
        "text_correct_rate",
        "emotion_correct_rate",
        "persistent_text_success_emotion_failure",
        *AFFECT_COLUMNS,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "RFC failure analysis is missing columns: "
            f"{missing_columns}"
        )

    df = df.copy()

    df["persistent_failure"] = (
        df[
            "persistent_text_success_emotion_failure"
        ]
        .astype(str)
        .str.lower()
        .eq("true")
    )

    df["jointly_detected"] = (
        (
            df["text_correct_rate"]
            >= SUCCESS_THRESHOLD
        )
        & (
            df["emotion_correct_rate"]
            >= SUCCESS_THRESHOLD
        )
    )

    return df


def create_descriptive_statistics(
    dataset_name: str,
    training_df: pd.DataFrame,
    rfc_df: pd.DataFrame,
) -> pd.DataFrame:
    """Create affective descriptive statistics for all comparison groups."""

    group_frames = {
        "training_genuine": training_df.loc[
            training_df["analysis_label"]
            == "genuine"
        ],
        "training_misleading": training_df.loc[
            training_df["analysis_label"]
            == "misleading"
        ],
        "rfc_persistent_failure": rfc_df.loc[
            rfc_df["persistent_failure"]
        ],
        "rfc_jointly_detected": rfc_df.loc[
            rfc_df["jointly_detected"]
        ],
    }

    rows: list[dict[str, object]] = []

    for group_name, group_df in group_frames.items():
        for feature in AFFECT_COLUMNS:
            values = (
                group_df[feature]
                .astype(float)
                .to_numpy()
            )

            rows.append(
                {
                    "dataset": dataset_name,
                    "group": group_name,
                    "feature": feature,
                    "n_examples": len(values),
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "std": float(
                        np.std(values, ddof=1)
                    )
                    if len(values) > 1
                    else np.nan,
                    "minimum": float(np.min(values)),
                    "maximum": float(np.max(values)),
                }
            )

    return pd.DataFrame(rows)


def create_profile_distances(
    dataset_name: str,
    descriptive: pd.DataFrame,
) -> pd.DataFrame:
    """Measure whether RFC groups are closer to genuine or misleading means."""

    mean_table = descriptive.pivot(
        index="group",
        columns="feature",
        values="mean",
    )

    genuine_profile = mean_table.loc[
        "training_genuine",
        AFFECT_COLUMNS,
    ].to_numpy(dtype=float)

    misleading_profile = mean_table.loc[
        "training_misleading",
        AFFECT_COLUMNS,
    ].to_numpy(dtype=float)

    rows: list[dict[str, object]] = []

    for group_name in [
        "rfc_persistent_failure",
        "rfc_jointly_detected",
    ]:
        profile = mean_table.loc[
            group_name,
            AFFECT_COLUMNS,
        ].to_numpy(dtype=float)

        distance_to_genuine = float(
            np.linalg.norm(
                profile - genuine_profile
            )
        )

        distance_to_misleading = float(
            np.linalg.norm(
                profile - misleading_profile
            )
        )

        closer_profile = (
            "genuine"
            if distance_to_genuine
            < distance_to_misleading
            else "misleading"
        )

        rows.append(
            {
                "dataset": dataset_name,
                "rfc_group": group_name,
                "distance_to_training_genuine": (
                    distance_to_genuine
                ),
                "distance_to_training_misleading": (
                    distance_to_misleading
                ),
                "closer_training_profile": (
                    closer_profile
                ),
                "distance_difference_genuine_minus_misleading": (
                    distance_to_genuine
                    - distance_to_misleading
                ),
            }
        )

    return pd.DataFrame(rows)


def create_failure_tests(
    dataset_name: str,
    rfc_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compare persistent failures with jointly detected RFC examples."""

    failure_df = rfc_df.loc[
        rfc_df["persistent_failure"]
    ]

    detected_df = rfc_df.loc[
        rfc_df["jointly_detected"]
    ]

    if failure_df.empty:
        raise ValueError(
            f"No persistent failures found for {dataset_name}."
        )

    if detected_df.empty:
        raise ValueError(
            f"No jointly detected examples found for {dataset_name}."
        )

    rows: list[dict[str, object]] = []

    for feature in AFFECT_COLUMNS:
        failure_values = (
            failure_df[feature]
            .astype(float)
            .to_numpy()
        )

        detected_values = (
            detected_df[feature]
            .astype(float)
            .to_numpy()
        )

        statistic, p_value = mannwhitneyu(
            failure_values,
            detected_values,
            alternative="two-sided",
        )

        rank_biserial = (
            2.0
            * statistic
            / (
                len(failure_values)
                * len(detected_values)
            )
            - 1.0
        )

        rows.append(
            {
                "dataset": dataset_name,
                "feature": feature,
                "persistent_failure_n": (
                    len(failure_values)
                ),
                "jointly_detected_n": (
                    len(detected_values)
                ),
                "persistent_failure_mean": float(
                    np.mean(failure_values)
                ),
                "jointly_detected_mean": float(
                    np.mean(detected_values)
                ),
                "mean_difference_failure_minus_detected": (
                    float(
                        np.mean(failure_values)
                        - np.mean(detected_values)
                    )
                ),
                "mann_whitney_u": float(statistic),
                "p_value": float(p_value),
                "rank_biserial": float(
                    rank_biserial
                ),
            }
        )

    adjusted_values = holm_adjust(
        [
            float(row["p_value"])
            for row in rows
        ]
    )

    for row, adjusted_value in zip(
        rows,
        adjusted_values,
    ):
        row["p_value_holm"] = adjusted_value
        row["significant_holm_0_05"] = (
            adjusted_value < 0.05
        )

    return pd.DataFrame(rows)


def main() -> None:
    """Run the complete affective-profile comparison."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rfc_analysis = (
        load_rfc_example_analysis()
    )

    descriptive_frames: list[pd.DataFrame] = []
    distance_frames: list[pd.DataFrame] = []
    test_frames: list[pd.DataFrame] = []

    for dataset_name in TRAINING_CONFIGS:
        training_df = load_training_data(
            dataset_name
        )

        dataset_rfc = rfc_analysis.loc[
            rfc_analysis["training_dataset"]
            == dataset_name
        ].copy()

        if dataset_rfc.empty:
            raise ValueError(
                f"No RFC results found for {dataset_name}."
            )

        descriptive = (
            create_descriptive_statistics(
                dataset_name=dataset_name,
                training_df=training_df,
                rfc_df=dataset_rfc,
            )
        )

        distances = create_profile_distances(
            dataset_name=dataset_name,
            descriptive=descriptive,
        )

        tests = create_failure_tests(
            dataset_name=dataset_name,
            rfc_df=dataset_rfc,
        )

        descriptive_frames.append(
            descriptive
        )

        distance_frames.append(
            distances
        )

        test_frames.append(
            tests
        )

    descriptive_results = pd.concat(
        descriptive_frames,
        ignore_index=True,
    )

    distance_results = pd.concat(
        distance_frames,
        ignore_index=True,
    )

    test_results = pd.concat(
        test_frames,
        ignore_index=True,
    )

    descriptive_path = (
        OUTPUT_DIR
        / "rq2_rfc_affective_profile_descriptive_statistics.csv"
    )

    distance_path = (
        OUTPUT_DIR
        / "rq2_rfc_affective_profile_distances.csv"
    )

    test_path = (
        OUTPUT_DIR
        / "rq2_rfc_failure_vs_detected_affective_tests.csv"
    )

    descriptive_results.to_csv(
        descriptive_path,
        index=False,
    )

    distance_results.to_csv(
        distance_path,
        index=False,
    )

    test_results.to_csv(
        test_path,
        index=False,
    )

    print(
        "\nRFC affective-profile comparison complete."
    )

    print("\nGroup sizes:")
    print(
        descriptive_results[
            [
                "dataset",
                "group",
                "n_examples",
            ]
        ]
        .drop_duplicates()
        .to_string(index=False)
    )

    print("\nProfile distances:")
    print(
        distance_results.to_string(
            index=False
        )
    )

    print(
        "\nPersistent failures versus "
        "jointly detected examples:"
    )

    print(
        test_results[
            [
                "dataset",
                "feature",
                "persistent_failure_mean",
                "jointly_detected_mean",
                "mean_difference_failure_minus_detected",
                "rank_biserial",
                "p_value_holm",
                "significant_holm_0_05",
            ]
        ].to_string(index=False)
    )

    print("\nSaved:")
    print(descriptive_path)
    print(distance_path)
    print(test_path)


if __name__ == "__main__":
    main()
