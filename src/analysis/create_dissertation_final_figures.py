"""Create polished final dissertation figures for RQ1 and RQ2."""

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


def load_table(filename: str) -> pd.DataFrame:
    """Load one required results table."""

    path = TABLE_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Required table not found: {path}"
        )

    return pd.read_csv(path)


def prepare_axis(axis: plt.Axes) -> None:
    """Apply a clean academic layout."""

    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.set_axisbelow(True)


def save_figure(
    figure: plt.Figure,
    filename: str,
) -> None:
    """Save both high-resolution PNG and vector PDF."""

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


def create_rq1_figure() -> None:
    """Create a horizontal dot plot for RQ1 affective differences."""

    results = load_table(
        "rq1_affective_comparison_long.csv"
    )

    results["dataset_label"] = (
        results["dataset"]
        .map(DATASET_LABELS)
    )

    colours = (
        plt.rcParams[
            "axes.prop_cycle"
        ]
        .by_key()["color"]
    )

    datasets = [
        ("LIAR2-Finance", "o", colours[0], -0.12),
        ("FACTors-Finance", "s", colours[1], 0.12),
    ]

    figure, axis = plt.subplots(
        figsize=(8.0, 4.8),
        constrained_layout=True,
    )

    y_positions = np.arange(
        len(FEATURE_ORDER)
    )

    axis.axvline(
        0,
        linewidth=1.1,
    )

    for (
        dataset,
        marker,
        colour,
        offset,
    ) in datasets:
        dataset_results = (
            results.loc[
                results["dataset_label"]
                == dataset
            ]
            .set_index("feature")
            .reindex(FEATURE_ORDER)
        )

        values = dataset_results[
            "mean_difference_misleading_minus_genuine"
        ].to_numpy()

        significance = dataset_results[
            "significant_after_holm_0_05"
        ].astype(bool).to_numpy()

        axis.scatter(
            values,
            y_positions + offset,
            marker=marker,
            s=75,
            label=dataset,
            zorder=3,
            color=colour,
        )

        for y_value, value, significant in zip(
            y_positions + offset,
            values,
            significance,
        ):
            label = (
                f"{value:+.3f}"
                + ("*" if significant else "")
            )

            horizontal_alignment = (
                "left"
                if value >= 0
                else "right"
            )

            x_offset = (
                5
                if value >= 0
                else -5
            )

            axis.annotate(
                label,
                xy=(value, y_value),
                xytext=(x_offset, 0),
                textcoords="offset points",
                va="center",
                ha=horizontal_alignment,
                fontsize=9,
            )

    axis.set_yticks(
        y_positions,
        [
            FEATURE_LABELS[feature]
            for feature in FEATURE_ORDER
        ],
    )

    axis.invert_yaxis()

    axis.set_xlabel(
        "Mean difference (misleading − genuine)"
    )

    axis.set_title(
        "Affective differences between misleading and genuine text",
        pad=12,
    )

    axis.grid(
        axis="x",
        alpha=0.25,
    )

    axis.legend(
        frameon=False,
        loc="lower right",
    )

    axis.text(
        0.99,
        0.02,
        "* Holm-adjusted p < .05",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
    )

    prepare_axis(axis)

    save_figure(
        figure,
        "rq1_affective_mean_difference_comparison.png",
    )


def create_in_domain_figure() -> None:
    """Create a compact dumbbell plot for in-domain RQ2 results."""

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
        .reset_index()
    )

    colours = (
        plt.rcParams[
            "axes.prop_cycle"
        ]
        .by_key()["color"]
    )

    figure, axis = plt.subplots(
        figsize=(8.0, 4.4),
        constrained_layout=True,
    )

    y_positions = np.arange(
        len(results)
    )

    for row_index, row in results.iterrows():
        y_value = y_positions[row_index]

        text_mean = row["text_only_mean"]
        emotion_mean = row["emotion_aware_mean"]

        axis.plot(
            [text_mean, emotion_mean],
            [y_value, y_value],
            linewidth=2,
            alpha=0.7,
            zorder=1,
        )

        axis.errorbar(
            text_mean,
            y_value,
            xerr=row["text_only_std"],
            fmt="o",
            markersize=8,
            capsize=4,
            linewidth=1.5,
            label=(
                "Text-only"
                if row_index == 0
                else None
            ),
            color=colours[0],
            zorder=3,
        )

        axis.errorbar(
            emotion_mean,
            y_value,
            xerr=row["emotion_aware_std"],
            fmt="s",
            markersize=8,
            capsize=4,
            linewidth=1.5,
            label=(
                "Emotion-aware"
                if row_index == 0
                else None
            ),
            color=colours[1],
            zorder=3,
        )

        difference = (
            row[
                "difference_percentage_points"
            ]
        )

        significance = (
            "p = .023"
            if bool(row["significant"])
            else "not significant"
        )

        annotation_x = max(
            text_mean + row["text_only_std"],
            emotion_mean
            + row["emotion_aware_std"],
        ) + 0.002

        axis.text(
            annotation_x,
            y_value,
            (
                f"{difference:+.2f} pp\n"
                f"{significance}"
            ),
            va="center",
            fontsize=9,
        )

    axis.set_yticks(
        y_positions,
        dataset_order,
    )

    axis.invert_yaxis()

    axis.set_xlabel(
        "Macro-F1"
    )

    axis.set_title(
        "In-domain effect of affective fusion",
        pad=12,
    )

    axis.set_xlim(
        0.60,
        0.68,
    )

    axis.grid(
        axis="x",
        alpha=0.25,
    )

    axis.legend(
        frameon=False,
        loc="lower right",
    )

    prepare_axis(axis)

    save_figure(
        figure,
        "rq2_final_in_domain_macro_f1.png",
    )


def create_rfc_overall_figure() -> None:
    """Create a dumbbell plot for overall RFC misleading recall."""

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
        .reset_index()
    )

    colours = (
        plt.rcParams[
            "axes.prop_cycle"
        ]
        .by_key()["color"]
    )

    figure, axis = plt.subplots(
        figsize=(8.0, 4.2),
        constrained_layout=True,
    )

    y_positions = np.arange(
        len(results)
    )

    for row_index, row in results.iterrows():
        y_value = y_positions[row_index]

        text_value = row["text_only_mean"]
        emotion_value = row["emotion_aware_mean"]

        axis.plot(
            [emotion_value, text_value],
            [y_value, y_value],
            linewidth=2,
            alpha=0.7,
            zorder=1,
        )

        axis.scatter(
            text_value,
            y_value,
            marker="o",
            s=80,
            label=(
                "Text-only"
                if row_index == 0
                else None
            ),
            color=colours[0],
            zorder=3,
        )

        axis.scatter(
            emotion_value,
            y_value,
            marker="s",
            s=80,
            label=(
                "Emotion-aware"
                if row_index == 0
                else None
            ),
            color=colours[1],
            zorder=3,
        )

        axis.text(
            text_value + 0.008,
            y_value,
            f"{text_value:.3f}",
            va="center",
            fontsize=9,
        )

        axis.text(
            emotion_value - 0.008,
            y_value,
            f"{emotion_value:.3f}",
            va="center",
            ha="right",
            fontsize=9,
        )

        midpoint = (
            text_value + emotion_value
        ) / 2

        axis.annotate(
            (
                f"{row['difference_percentage_points']:+.2f} pp"
            ),
            xy=(midpoint, y_value),
            xytext=(0, -18),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )

    axis.set_yticks(
        y_positions,
        dataset_order,
    )

    axis.invert_yaxis()

    axis.set_xlabel(
        "Misleading recall on RFC-Bench"
    )

    axis.set_title(
        "External robustness of the trained models",
        pad=12,
    )

    axis.set_xlim(
        0.27,
        0.44,
    )

    axis.grid(
        axis="x",
        alpha=0.25,
    )

    axis.legend(
        frameon=False,
        loc="lower right",
    )

    prepare_axis(axis)

    save_figure(
        figure,
        "rq2_final_rfc_overall_recall.png",
    )


def create_rfc_category_figure() -> None:
    """Create a category-level horizontal change plot."""

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

    datasets = [
        "LIAR2-Finance",
        "FACTors-Finance",
    ]

    colours = (
        plt.rcParams[
            "axes.prop_cycle"
        ]
        .by_key()["color"]
    )

    settings = [
        (
            "LIAR2-Finance",
            "o",
            colours[0],
            -0.13,
        ),
        (
            "FACTors-Finance",
            "s",
            colours[1],
            0.13,
        ),
    ]

    figure, axis = plt.subplots(
        figsize=(8.2, 4.8),
        constrained_layout=True,
    )

    y_positions = np.arange(
        len(category_order)
    )

    axis.axvline(
        0,
        linewidth=1.1,
    )

    for (
        dataset,
        marker,
        colour,
        offset,
    ) in settings:
        dataset_results = (
            results.loc[
                results["dataset_label"]
                == dataset
            ]
            .set_index(
                "manipulation_category"
            )
            .reindex(category_order)
        )

        values = dataset_results[
            "recall_change_percentage_points"
        ].to_numpy()

        adjusted_y = (
            y_positions + offset
        )

        for y_value, value in zip(
            adjusted_y,
            values,
        ):
            axis.plot(
                [0, value],
                [y_value, y_value],
                linewidth=1.5,
                alpha=0.5,
                color=colour,
                zorder=1,
            )

        axis.scatter(
            values,
            adjusted_y,
            marker=marker,
            s=75,
            color=colour,
            label=dataset,
            zorder=3,
        )

        for y_value, value in zip(
            adjusted_y,
            values,
        ):
            axis.annotate(
                f"{value:.1f}",
                xy=(value, y_value),
                xytext=(-6, 0),
                textcoords="offset points",
                ha="right",
                va="center",
                fontsize=9,
            )

    axis.set_yticks(
        y_positions,
        [
            category.capitalize()
            for category in category_order
        ],
    )

    axis.invert_yaxis()

    axis.set_xlabel(
        "Recall change: emotion-aware − text-only "
        "(percentage points)"
    )

    axis.set_title(
        "External effect by RFC-Bench manipulation type",
        pad=12,
    )

    axis.set_xlim(
        -18,
        1,
    )

    axis.grid(
        axis="x",
        alpha=0.25,
    )

    axis.legend(
        frameon=False,
        loc="lower right",
    )

    prepare_axis(axis)

    save_figure(
        figure,
        "rq2_final_rfc_category_recall_change.png",
    )


def main() -> None:
    """Create all final dissertation figures."""

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

    print("\nFinal dissertation figures created:")

    for filename in [
        "rq1_affective_mean_difference_comparison.png",
        "rq2_final_in_domain_macro_f1.png",
        "rq2_final_rfc_overall_recall.png",
        "rq2_final_rfc_category_recall_change.png",
    ]:
        print(FIGURE_DIR / filename)


if __name__ == "__main__":
    main()
