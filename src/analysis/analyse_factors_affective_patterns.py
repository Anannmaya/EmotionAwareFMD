# PERMANENT RQ1 ANALYSIS SCRIPT — KEEP THIS FILE

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon


INPUT_PATH = Path(
    "data/features/factors_with_affective_features.csv"
)

DESCRIPTIVE_OUTPUT = Path(
    "outputs/tables/factors_affective_descriptive_statistics.csv"
)

TEST_OUTPUT = Path(
    "outputs/tables/factors_affective_paired_tests.csv"
)

LABEL_COLUMN = "label"
PAIR_COLUMN = "pair_id"
EXPECTED_ROWS = 522
EXPECTED_PAIRS = 261

FEATURE_COLUMNS = [
    "affect_anger",
    "affect_fear",
    "affect_joy",
    "affect_sadness",
    "affect_valence",
]


def holm_adjust(p_values: list[float]) -> np.ndarray:
    """Apply Holm correction for multiple hypothesis tests."""
    p_values_array = np.asarray(p_values, dtype=float)
    number_of_tests = len(p_values_array)

    sorted_indices = np.argsort(p_values_array)
    adjusted_sorted = np.empty(number_of_tests, dtype=float)

    previous_adjusted = 0.0

    for position, index in enumerate(sorted_indices):
        adjusted_value = (
            number_of_tests - position
        ) * p_values_array[index]

        adjusted_value = max(previous_adjusted, adjusted_value)
        adjusted_sorted[position] = min(adjusted_value, 1.0)
        previous_adjusted = adjusted_sorted[position]

    adjusted = np.empty(number_of_tests, dtype=float)

    for position, index in enumerate(sorted_indices):
        adjusted[index] = adjusted_sorted[position]

    return adjusted


def paired_rank_biserial(differences: pd.Series) -> float:
    """
    Calculate paired rank-biserial correlation.

    Positive values indicate higher scores in misleading texts.
    Negative values indicate higher scores in genuine texts.
    """
    nonzero_differences = differences[
        differences != 0
    ].to_numpy()

    if len(nonzero_differences) == 0:
        return 0.0

    ranks = rankdata(np.abs(nonzero_differences))

    positive_rank_sum = ranks[
        nonzero_differences > 0
    ].sum()

    negative_rank_sum = ranks[
        nonzero_differences < 0
    ].sum()

    total_rank_sum = positive_rank_sum + negative_rank_sum

    return (
        positive_rank_sum - negative_rank_sum
    ) / total_rank_sum


def validate_dataset(dataframe: pd.DataFrame) -> None:
    required_columns = {
        LABEL_COLUMN,
        PAIR_COLUMN,
        *FEATURE_COLUMNS,
    }

    missing_columns = required_columns.difference(
        dataframe.columns
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if len(dataframe) != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS:,} rows, "
            f"found {len(dataframe):,}."
        )

    if dataframe[FEATURE_COLUMNS].isna().any().any():
        raise ValueError(
            "Missing affective feature values detected."
        )

    if set(dataframe[LABEL_COLUMN].unique()) != {
        "genuine",
        "misleading",
    }:
        raise ValueError(
            "Unexpected label values detected."
        )

    pair_check = (
        dataframe.groupby(PAIR_COLUMN)
        .agg(
            rows=(LABEL_COLUMN, "size"),
            labels=(LABEL_COLUMN, "nunique"),
        )
    )

    if len(pair_check) != EXPECTED_PAIRS:
        raise ValueError(
            f"Expected {EXPECTED_PAIRS} matched pairs, "
            f"found {len(pair_check)}."
        )

    if not (pair_check["rows"] == 2).all():
        raise ValueError(
            "Every matched pair must contain exactly two rows."
        )

    if not (pair_check["labels"] == 2).all():
        raise ValueError(
            "Every matched pair must contain one genuine "
            "and one misleading text."
        )


def create_descriptive_statistics(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    results = []

    for feature in FEATURE_COLUMNS:
        for label in ["genuine", "misleading"]:
            values = dataframe.loc[
                dataframe[LABEL_COLUMN] == label,
                feature,
            ]

            results.append(
                {
                    "feature": feature,
                    "label": label,
                    "n": len(values),
                    "mean": values.mean(),
                    "standard_deviation": values.std(ddof=1),
                    "median": values.median(),
                    "q1": values.quantile(0.25),
                    "q3": values.quantile(0.75),
                }
            )

    return pd.DataFrame(results)


def run_paired_tests(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    results = []

    for feature in FEATURE_COLUMNS:
        paired_values = dataframe.pivot(
            index=PAIR_COLUMN,
            columns=LABEL_COLUMN,
            values=feature,
        )

        genuine = paired_values["genuine"]
        misleading = paired_values["misleading"]

        differences = misleading - genuine

        test_result = wilcoxon(
            differences,
            alternative="two-sided",
            zero_method="wilcox",
            method="auto",
        )

        results.append(
            {
                "feature": feature,
                "pairs": len(differences),
                "genuine_mean": genuine.mean(),
                "misleading_mean": misleading.mean(),
                "mean_difference_misleading_minus_genuine": (
                    differences.mean()
                ),
                "median_difference_misleading_minus_genuine": (
                    differences.median()
                ),
                "wilcoxon_statistic": test_result.statistic,
                "p_value": test_result.pvalue,
                "rank_biserial_correlation": (
                    paired_rank_biserial(differences)
                ),
            }
        )

    results_dataframe = pd.DataFrame(results)

    results_dataframe["holm_adjusted_p_value"] = holm_adjust(
        results_dataframe["p_value"].tolist()
    )

    results_dataframe["significant_after_holm_0_05"] = (
        results_dataframe["holm_adjusted_p_value"] < 0.05
    )

    return results_dataframe


def main() -> None:
    dataframe = pd.read_csv(INPUT_PATH)

    validate_dataset(dataframe)

    descriptive_statistics = create_descriptive_statistics(
        dataframe
    )

    paired_tests = run_paired_tests(dataframe)

    DESCRIPTIVE_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    descriptive_statistics.to_csv(
        DESCRIPTIVE_OUTPUT,
        index=False,
    )

    paired_tests.to_csv(
        TEST_OUTPUT,
        index=False,
    )

    print("FACTors RQ1 analysis completed.")
    print(
        f"Matched pairs: "
        f"{dataframe[PAIR_COLUMN].nunique():,}"
    )

    print("\nPaired Wilcoxon results:")
    print(
        paired_tests[
            [
                "feature",
                "mean_difference_misleading_minus_genuine",
                "rank_biserial_correlation",
                "holm_adjusted_p_value",
                "significant_after_holm_0_05",
            ]
        ].to_string(index=False)
    )

    print("\nSaved:")
    print(DESCRIPTIVE_OUTPUT)
    print(TEST_OUTPUT)


if __name__ == "__main__":
    main()