"""Create dissertation-ready figures for the final RQ2 results."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"


def load_table(filename: str) -> pd.DataFrame:
    """Load one required final RQ2 table."""

    path = TABLE_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Required table not found: {path}"
        )

    return pd.read_csv(path)


def display_dataset_name(name: str) -> str:
    """Convert internal dataset names into dissertation labels."""

    mapping = {
        "liar2_finance": "LIAR2-Finance",
        "factors_finance": "FACTors-Finance",
    }

    return mapping.get(name, name)


def create_in_domain_macro_f1_figure() -> Path:
    """Compare text-only and emotion-aware macro-F1."""

    results = load_table(
        "rq2_final_in_domain_results.csv"
    )

    results = results.loc[
        results["metric"] == "macro_f1"
    ].copy()

    results["dataset_label"] = (
        results["dataset"]
        .map(display_dataset_name)
    )

    figure, axis = plt.subplots(
        figsize=(7.5, 5.0)
    )

    x_positions = np.arange(
        len(results)
    )

    offsets = {
        "Text-only": -0.06,
        "Emotion-aware": 0.06,
    }

    model_settings = [
        (
            "Text-only",
            "text_only_mean",
            "text_only_std",
            "o",
        ),
        (
            "Emotion-aware",
            "emotion_aware_mean",
            "emotion_aware_std",
            "s",
        ),
    ]

    for (
        model_label,
        mean_column,
        std_column,
        marker,
    ) in model_settings:
        axis.errorbar(
            x_positions + offsets[model_label],
            results[mean_column],
            yerr=results[std_column],
            fmt=marker,
            capsize=5,
            markersize=8,
            linewidth=1.5,
            label=model_label,
        )

    axis.set_ylabel("Macro-F1")
    axis.set_xlabel("Dataset")
    axis.set_title(
        "In-domain RQ2 performance"
    )
    axis.set_xticks(
        x_positions,
        results["dataset_label"],
    )
    axis.set_ylim(0.58, 0.69)
    axis.legend()
    axis.grid(
        axis="y",
        alpha=0.3,
    )

    figure.tight_layout()

    output_path = (
        FIGURE_DIR
        / "rq2_final_in_domain_macro_f1.png"
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    return output_path

def create_rfc_overall_recall_figure() -> Path:
    """Compare RFC misleading recall across model variants."""

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
        .map(display_dataset_name)
    )

    x_positions = np.arange(len(results))
    width = 0.34

    figure, axis = plt.subplots(
        figsize=(7.5, 5.0)
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

    axis.set_ylabel(
        "Misleading recall"
    )
    axis.set_xlabel(
        "Training dataset"
    )
    axis.set_title(
        "External robustness on RFC-Bench"
    )
    axis.set_xticks(
        x_positions,
        results["dataset_label"],
    )
    axis.set_ylim(0.0, 0.48)
    axis.legend()
    axis.grid(
        axis="y",
        alpha=0.3,
    )

    axis.bar_label(
        text_bars,
        fmt="%.3f",
        padding=3,
        fontsize=9,
    )

    axis.bar_label(
        emotion_bars,
        fmt="%.3f",
        padding=3,
        fontsize=9,
    )

    figure.tight_layout()

    output_path = (
        FIGURE_DIR
        / "rq2_final_rfc_overall_recall.png"
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    return output_path


def create_rfc_category_change_figure() -> Path:
    """Show category-level RFC recall changes."""

    results = load_table(
        "rq2_final_rfc_category_results.csv"
    )

    results["dataset_label"] = (
        results["dataset"]
        .map(display_dataset_name)
    )

    category_order = [
        "causal",
        "numerical",
        "flipping",
        "sentiment",
    ]

    pivot = results.pivot(
        index="manipulation_category",
        columns="dataset_label",
        values="recall_change_percentage_points",
    )

    pivot = pivot.reindex(
        category_order
    )

    figure, axis = plt.subplots(
        figsize=(8.0, 5.3)
    )

    pivot.plot(
        kind="bar",
        ax=axis,
    )

    axis.axhline(
        0,
        linewidth=1,
    )

    axis.set_ylabel(
        "Recall change: emotion-aware minus text-only\n"
        "(percentage points)"
    )
    axis.set_xlabel(
        "Manipulation category"
    )
    axis.set_title(
        "RFC-Bench category-level effect of affective fusion"
    )
    axis.set_xticklabels(
        [
            category.capitalize()
            for category in category_order
        ],
        rotation=0,
    )
    axis.legend(
        title="Training dataset"
    )
    axis.grid(
        axis="y",
        alpha=0.3,
    )

    figure.tight_layout()

    output_path = (
        FIGURE_DIR
        / "rq2_final_rfc_category_recall_change.png"
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    return output_path


def main() -> None:
    """Create all final RQ2 figures."""

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = [
        create_in_domain_macro_f1_figure(),
        create_rfc_overall_recall_figure(),
        create_rfc_category_change_figure(),
    ]

    print("\nFinal RQ2 figures created.")

    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
