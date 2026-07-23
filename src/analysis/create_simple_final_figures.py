"""Create simple, easy-to-read dissertation figures for RQ1 and RQ2."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

DATASET_LABELS = {
    "liar2_finance": "LIAR2-Finance",
    "factors_finance": "FACTors-Finance",
    "LIAR2-Finance": "LIAR2-Finance",
    "FACTors": "FACTors-Finance",
    "FACTors-Finance": "FACTors-Finance",
}


def load_table(filename: str) -> pd.DataFrame:
    """Load one required results table."""

    path = TABLE_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Required table not found: {path}"
        )

    return pd.read_csv(path)


def save_figure(
    figure: plt.Figure,
    filename: str,
) -> None:
    """Save PNG and PDF copies."""

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    png_path = FIGURE_DIR / filename
    pdf_path = png_path.with_suffix(".pdf")

    figure.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    figure.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.close(figure)


def clean_axis(axis: plt.Axes) -> None:
    """Apply a simple academic layout."""

    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(
        axis="y",
        alpha=0.25,
    )
    axis.set_axisbelow(True)


def add_bar_labels(
    axis: plt.Axes,
    bars,
    decimals: int = 3,
) -> None:
    """Write values above positive bars and below negative bars."""

    for bar in bars:
        value = bar.get_height()

        vertical_alignment = (
            "bottom"
            if value >= 0
            else "top"
        )

        offset = (
            3
            if value >= 0
            else -3
        )

        axis.annotate(
            f"{value:.{decimals}f}",
            xy=(
                bar.get_x()
                + bar.get_width() / 2,
                value,
            ),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va=vertical_alignment,
            fontsize=9,
        )


def create_rq1_figure() -> None:
    """Create the simple RQ1 grouped bar chart."""

    results = load_table(
        "rq1_affective_comparison_long.csv"
    )

    results["dataset_label"] = (
        results["dataset"]
        .map(DATASET_LABELS)
    )

    feature_order = [
        "affect_anger",
        "affect_fear",
        "affect_joy",
        "affect_sadness",
        "affect_valence",
    ]

    feature_labels = [
        "Anger",
        "Fear",
        "Joy",
        "Sadness",
        "Valence",
    ]

    pivot = results.pivot(
        index="feature",
        columns="dataset_label",
        values=(
            "mean_difference_misleading_minus_genuine"
        ),
    ).reindex(feature_order)

    x_positions = np.arange(
        len(feature_order)
    )

    width = 0.36

    figure, axis = plt.subplots(
        figsize=(8.5, 4.8)
    )

    liar2_bars = axis.bar(
        x_positions - width / 2,
        pivot["LIAR2-Finance"],
        width,
        label="LIAR2-Finance",
    )

    factors_bars = axis.bar(
        x_positions + width / 2,
        pivot["FACTors-Finance"],
        width,
        label="FACTors-Finance",
    )

    axis.axhline(
        0,
        linewidth=1,
    )

    axis.set_xticks(
        x_positions,
        feature_labels,
    )

    axis.set_ylabel(
        "Mean difference\n(misleading − genuine)"
    )

    axis.set_title(
        "Emotional differences between misleading and genuine text"
    )

    axis.legend(
        frameon=False,
    )

    add_bar_labels(
        axis,
        liar2_bars,
    )

    add_bar_labels(
        axis,
        factors_bars,
    )

    clean_axis(axis)
    figure.tight_layout()

    save_figure(
        figure,
        "rq1_affective_mean_difference_comparison.png",
    )


def create_in_domain_figure() -> None:
    """Create the simple in-domain macro-F1 chart."""

    results = load_table(
        "rq2_final_in_domain_results.csv"
    )

    results = results.loc[
        results["metric"] == "macro_f1"
    ].copy()

    results["dataset_label"] = (
        results["dataset"]
        .map(DATASET_LABELS)
    )

    dataset_order = [
        "LIAR2-Finance",
        "FACTors-Finance",
    ]

    results = (
        results.set_index("dataset_label")
        .reindex(dataset_order)
    )

    x_positions = np.arange(
        len(dataset_order)
    )

    width = 0.36

    figure, axis = plt.subplots(
        figsize=(7.5, 4.8)
    )

    text_bars = axis.bar(
        x_positions - width / 2,
        results["text_only_mean"],
        width,
        yerr=results["text_only_std"],
        capsize=4,
        label="Text-only",
    )

    emotion_bars = axis.bar(
        x_positions + width / 2,
        results["emotion_aware_mean"],
        width,
        yerr=results["emotion_aware_std"],
        capsize=4,
        label="Emotion-aware",
    )

    axis.set_xticks(
        x_positions,
        dataset_order,
    )

    axis.set_ylim(
        0,
        0.75,
    )

    axis.set_ylabel(
        "Macro-F1"
    )

    axis.set_title(
        "In-domain misinformation detection performance"
    )

    axis.legend(
        frameon=False,
    )

    add_bar_labels(
        axis,
        text_bars,
    )

    add_bar_labels(
        axis,
        emotion_bars,
    )

    clean_axis(axis)
    figure.tight_layout()

    save_figure(
        figure,
        "rq2_final_in_domain_macro_f1.png",
    )


def create_rfc_overall_figure() -> None:
    """Create the simple overall RFC recall chart."""

    results = load_table(
        "rq2_final_rfc_overall_results.csv"
    )

    results = results.loc[
        (
            results["scope"] == "overall"
        )
        & (
            results["metric"]
            == "misleading_recall"
        )
    ].copy()

    results["dataset_label"] = (
        results["dataset"]
        .map(DATASET_LABELS)
    )

    dataset_order = [
        "LIAR2-Finance",
        "FACTors-Finance",
    ]

    results = (
        results.set_index("dataset_label")
        .reindex(dataset_order)
    )

    x_positions = np.arange(
        len(dataset_order)
    )

    width = 0.36

    figure, axis = plt.subplots(
        figsize=(7.5, 4.8)
    )

    text_bars = axis.bar(
        x_positions - width / 2,
        results["text_only_mean"],
        width,
        label="Text-only",
    )

    emotion_bars = axis.bar(
        x_positions + width / 2,
        results["emotion_aware_mean"],
        width,
        label="Emotion-aware",
    )

    axis.set_xticks(
        x_positions,
        dataset_order,
    )

    axis.set_ylim(
        0,
        0.5,
    )

    axis.set_ylabel(
        "Misleading recall"
    )

    axis.set_xlabel(
        "Training dataset"
    )

    axis.set_title(
        "External performance on RFC-Bench"
    )

    axis.legend(
        frameon=False,
    )

    add_bar_labels(
        axis,
        text_bars,
    )

    add_bar_labels(
        axis,
        emotion_bars,
    )

    clean_axis(axis)
    figure.tight_layout()

    save_figure(
        figure,
        "rq2_final_rfc_overall_recall.png",
    )


def create_rfc_category_figure() -> None:
    """Create the simple RFC category comparison chart."""

    results = load_table(
        "rq2_final_rfc_category_results.csv"
    )

    results["dataset_label"] = (
        results["dataset"]
        .map(DATASET_LABELS)
    )

    category_order = [
        "causal",
        "numerical",
        "flipping",
        "sentiment",
    ]

    category_labels = [
        "Causal",
        "Numerical",
        "Flipping",
        "Sentiment",
    ]

    pivot = results.pivot(
        index="manipulation_category",
        columns="dataset_label",
        values="recall_change_percentage_points",
    ).reindex(category_order)

    x_positions = np.arange(
        len(category_order)
    )

    width = 0.36

    figure, axis = plt.subplots(
        figsize=(8.5, 4.8)
    )

    liar2_bars = axis.bar(
        x_positions - width / 2,
        pivot["LIAR2-Finance"],
        width,
        label="LIAR2-Finance",
    )

    factors_bars = axis.bar(
        x_positions + width / 2,
        pivot["FACTors-Finance"],
        width,
        label="FACTors-Finance",
    )

    axis.axhline(
        0,
        linewidth=1,
    )

    axis.set_xticks(
        x_positions,
        category_labels,
    )

    axis.set_ylabel(
        "Recall change\n(percentage points)"
    )

    axis.set_xlabel(
        "Manipulation category"
    )

    axis.set_title(
        "Effect of emotion features on RFC-Bench recall"
    )

    axis.legend(
        frameon=False,
    )

    add_bar_labels(
        axis,
        liar2_bars,
        decimals=1,
    )

    add_bar_labels(
        axis,
        factors_bars,
        decimals=1,
    )

    clean_axis(axis)
    figure.tight_layout()

    save_figure(
        figure,
        "rq2_final_rfc_category_recall_change.png",
    )


def main() -> None:
    """Create all four simple figures."""

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
        }
    )

    create_rq1_figure()
    create_in_domain_figure()
    create_rfc_overall_figure()
    create_rfc_category_figure()

    print("\nSimple final figures created.")

    for filename in [
        "rq1_affective_mean_difference_comparison.png",
        "rq2_final_in_domain_macro_f1.png",
        "rq2_final_rfc_overall_recall.png",
        "rq2_final_rfc_category_recall_change.png",
    ]:
        print(FIGURE_DIR / filename)


if __name__ == "__main__":
    main()
