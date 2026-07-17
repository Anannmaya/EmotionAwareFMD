# PERMANENT RQ1 SUMMARY SCRIPT — KEEP THIS FILE

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LIAR2_INPUT = Path("outputs/tables/liar2_affective_paired_tests.csv")
FACTORS_INPUT = Path("outputs/tables/factors_affective_paired_tests.csv")

COMBINED_LONG_OUTPUT = Path(
    "outputs/tables/rq1_affective_comparison_long.csv"
)

COMBINED_WIDE_OUTPUT = Path(
    "outputs/tables/rq1_affective_comparison_wide.csv"
)

FIGURE_OUTPUT = Path(
    "outputs/figures/rq1_affective_mean_difference_comparison.png"
)

FEATURE_ORDER = [
    "affect_anger",
    "affect_fear",
    "affect_joy",
    "affect_sadness",
    "affect_valence",
]

FEATURE_LABELS = {
    "affect_anger": "Anger",
    "affect_fear": "Fear",
    "affect_joy": "Joy",
    "affect_sadness": "Sadness",
    "affect_valence": "Valence",
}


def load_results(path: Path, dataset_name: str) -> pd.DataFrame:
    dataframe = pd.read_csv(path).copy()
    dataframe["dataset"] = dataset_name
    return dataframe


def build_combined_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    liar2 = load_results(LIAR2_INPUT, "LIAR2-Finance")
    factors = load_results(FACTORS_INPUT, "FACTors")

    combined = pd.concat([liar2, factors], ignore_index=True)

    combined["feature"] = pd.Categorical(
        combined["feature"],
        categories=FEATURE_ORDER,
        ordered=True,
    )

    combined = combined.sort_values(["feature", "dataset"]).reset_index(drop=True)

    combined_long = combined[
        [
            "dataset",
            "feature",
            "pairs",
            "mean_difference_misleading_minus_genuine",
            "rank_biserial_correlation",
            "holm_adjusted_p_value",
            "significant_after_holm_0_05",
        ]
    ].copy()

    combined_long["feature_label"] = combined_long["feature"].map(FEATURE_LABELS)

    combined_wide = combined_long.pivot(
        index="feature_label",
        columns="dataset",
        values=[
            "mean_difference_misleading_minus_genuine",
            "rank_biserial_correlation",
            "holm_adjusted_p_value",
            "significant_after_holm_0_05",
        ],
    )

    combined_wide = combined_wide.sort_index()

    return combined_long, combined_wide


def save_tables(combined_long: pd.DataFrame, combined_wide: pd.DataFrame) -> None:
    COMBINED_LONG_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    combined_long.to_csv(COMBINED_LONG_OUTPUT, index=False)
    combined_wide.to_csv(COMBINED_WIDE_OUTPUT)


def create_figure(combined_long: pd.DataFrame) -> None:
    figure_df = combined_long.copy()

    liar2 = (
        figure_df[figure_df["dataset"] == "LIAR2-Finance"]
        .set_index("feature")
        .reindex(FEATURE_ORDER)
    )

    factors = (
        figure_df[figure_df["dataset"] == "FACTors"]
        .set_index("feature")
        .reindex(FEATURE_ORDER)
    )

    x = np.arange(len(FEATURE_ORDER))
    width = 0.36

    plt.figure(figsize=(10, 6))

    plt.bar(
        x - width / 2,
        liar2["mean_difference_misleading_minus_genuine"],
        width=width,
        label="LIAR2-Finance",
    )

    plt.bar(
        x + width / 2,
        factors["mean_difference_misleading_minus_genuine"],
        width=width,
        label="FACTors",
    )

    plt.axhline(0, linewidth=1)

    plt.xticks(
        ticks=x,
        labels=[FEATURE_LABELS[feature] for feature in FEATURE_ORDER],
    )

    plt.ylabel("Mean difference (misleading - genuine)")
    plt.xlabel("Affective feature")
    plt.title("RQ1: Affective differences across financial misinformation datasets")
    plt.legend()
    plt.tight_layout()

    FIGURE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURE_OUTPUT, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    combined_long, combined_wide = build_combined_tables()
    save_tables(combined_long, combined_wide)
    create_figure(combined_long)

    print("RQ1 combined summary created.")
    print("\nSaved tables:")
    print(COMBINED_LONG_OUTPUT)
    print(COMBINED_WIDE_OUTPUT)

    print("\nSaved figure:")
    print(FIGURE_OUTPUT)

    print("\nCombined table preview:")
    print(
        combined_long[
            [
                "dataset",
                "feature_label",
                "mean_difference_misleading_minus_genuine",
                "rank_biserial_correlation",
                "holm_adjusted_p_value",
                "significant_after_holm_0_05",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()