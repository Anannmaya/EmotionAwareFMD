"""Create concise dissertation-ready tables for RQ2."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"


def require_file(filename: str) -> Path:
    """Return an existing result path or raise a clear error."""

    path = TABLE_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Required result file not found: {path}"
        )

    return path


def create_in_domain_table() -> pd.DataFrame:
    """Create the main nested-CV comparison table."""

    results = pd.read_csv(
        require_file(
            "rq2_nested_weighted_statistical_comparison.csv"
        )
    )

    required_columns = {
        "dataset",
        "metric",
        "n_repetitions",
        "text_only_mean",
        "text_only_std",
        "emotion_aware_mean",
        "emotion_aware_std",
        "mean_difference_emotion_minus_text",
        "rank_biserial",
        "p_value_holm",
        "significant_holm_0_05",
    }

    missing = required_columns.difference(results.columns)

    if missing:
        raise ValueError(
            f"Nested-CV results are missing columns: {sorted(missing)}"
        )

    final = results[
        [
            "dataset",
            "metric",
            "n_repetitions",
            "text_only_mean",
            "text_only_std",
            "emotion_aware_mean",
            "emotion_aware_std",
            "mean_difference_emotion_minus_text",
            "rank_biserial",
            "p_value_holm",
            "significant_holm_0_05",
        ]
    ].copy()

    final["difference_percentage_points"] = (
        final["mean_difference_emotion_minus_text"]
        * 100
    )

    final = final.rename(
        columns={
            "n_repetitions": "repetitions",
            "mean_difference_emotion_minus_text": (
                "difference_emotion_minus_text"
            ),
            "p_value_holm": "holm_adjusted_p",
            "significant_holm_0_05": "significant",
        }
    )

    return final


def create_rfc_overall_table() -> pd.DataFrame:
    """Create the overall RFC external-evaluation table."""

    results = pd.read_csv(
        require_file(
            "rq2_rfc_weighted_external_paired_differences.csv"
        )
    )

    final = results.loc[
        (
            results["scope"].isin(
                [
                    "overall",
                    "overall_non_sentiment",
                ]
            )
        )
        & (
            results["metric"].isin(
                [
                    "misleading_recall",
                    "mean_probability_misleading",
                ]
            )
        ),
        [
            "dataset",
            "scope",
            "metric",
            "repetitions",
            "text_only_mean",
            "emotion_aware_mean",
            "mean_difference",
            "emotion_better_repetitions",
            "text_better_repetitions",
            "tied_repetitions",
        ],
    ].copy()

    final["difference_percentage_points"] = (
        final["mean_difference"] * 100
    )

    return final


def create_rfc_category_table() -> pd.DataFrame:
    """Create category-level RFC failure results."""

    results = pd.read_csv(
        require_file(
            "rq2_rfc_weighted_failure_analysis_by_category.csv"
        )
    )

    final = results[
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
    ].copy()

    final["recall_change_percentage_points"] = (
        final["recall_change_emotion_minus_text"]
        * 100
    )

    final = final.rename(
        columns={
            "training_dataset": "dataset",
        }
    )

    return final


def create_weight_summary() -> pd.DataFrame:
    """Summarise affective weights selected in outer nested CV."""

    selected = pd.read_csv(
        require_file(
            "rq2_nested_weighted_selected_hyperparameters.csv"
        )
    )

    emotion = selected.loc[
        selected["model"] == "emotion_aware"
    ].copy()

    if emotion.empty:
        raise ValueError(
            "No emotion-aware hyperparameter selections found."
        )

    counts = (
        emotion.groupby(
            [
                "dataset",
                "affect_weight",
            ],
            as_index=False,
        )
        .size()
        .rename(
            columns={
                "size": "selected_outer_folds",
            }
        )
    )

    totals = (
        emotion.groupby(
            "dataset"
        )
        .size()
        .rename("total_outer_folds")
    )

    counts = counts.merge(
        totals,
        on="dataset",
        how="left",
        validate="many_to_one",
    )

    counts["selection_percentage"] = (
        counts["selected_outer_folds"]
        / counts["total_outer_folds"]
        * 100
    )

    return counts


def main() -> None:
    """Generate and save all final RQ2 tables."""

    TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    in_domain = create_in_domain_table()
    rfc_overall = create_rfc_overall_table()
    rfc_category = create_rfc_category_table()
    weight_summary = create_weight_summary()

    output_tables = {
        "rq2_final_in_domain_results.csv": in_domain,
        "rq2_final_rfc_overall_results.csv": rfc_overall,
        "rq2_final_rfc_category_results.csv": rfc_category,
        "rq2_final_affect_weight_summary.csv": weight_summary,
    }

    for filename, dataframe in output_tables.items():
        dataframe.to_csv(
            TABLE_DIR / filename,
            index=False,
        )

    print("\nFinal RQ2 tables created.")

    print("\nIn-domain results:")
    print(
        in_domain.round(4).to_string(
            index=False
        )
    )

    print("\nRFC overall misleading recall:")
    print(
        rfc_overall.loc[
            rfc_overall["metric"]
            == "misleading_recall"
        ]
        .round(4)
        .to_string(index=False)
    )

    print("\nSaved:")
    for filename in output_tables:
        print(TABLE_DIR / filename)


if __name__ == "__main__":
    main()
