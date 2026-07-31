"""Statistical and RFC summaries for the final FinBERT RQ2 experiment."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.finbert_rq2_utils import (
    PRIMARY_METRIC,
    SECONDARY_METRIC,
    holm,
    paired_test,
)


def comparison_table(metrics: pd.DataFrame, directional: bool) -> pd.DataFrame:
    rows = []
    metrics_to_test = [PRIMARY_METRIC] if directional else [
        PRIMARY_METRIC,
        SECONDARY_METRIC,
    ]
    for dataset in sorted(metrics["dataset"].unique()):
        dataset_metrics = metrics[metrics["dataset"] == dataset]
        for metric in metrics_to_test:
            pivot = dataset_metrics.pivot(
                index="seed",
                columns="model",
                values=metric,
            ).dropna()
            text = pivot["text_only"].to_numpy(dtype=float)
            emotion = pivot["emotion_aware"].to_numpy(dtype=float)
            difference = emotion - text
            alternative = "greater" if directional else "two-sided"
            statistic, p_value, effect = paired_test(
                emotion,
                text,
                alternative,
            )
            rows.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "alternative": alternative,
                    "repetitions": len(pivot),
                    "text_only_mean": text.mean(),
                    "emotion_aware_mean": emotion.mean(),
                    "mean_difference": difference.mean(),
                    "relative_improvement_percent": (
                        100 * difference.mean() / text.mean()
                    ),
                    "emotion_better_repetitions": int((difference > 0).sum()),
                    "text_better_repetitions": int((difference < 0).sum()),
                    "ties": int((difference == 0).sum()),
                    "wilcoxon_statistic": statistic,
                    "p_value": p_value,
                    "rank_biserial": effect,
                }
            )
    result = pd.DataFrame(rows)
    result["p_value_holm"] = holm(result["p_value"].tolist())
    result["significant_holm_0_05"] = result["p_value_holm"] < 0.05
    return result


def rfc_groups(df: pd.DataFrame):
    yield "overall", "all", "all", np.ones(len(df), dtype=bool)
    non_sentiment = df["manipulation_category"].ne("sentiment").to_numpy()
    yield "overall_non_sentiment", "all", "non_sentiment", non_sentiment
    for subset in sorted(df["subset"].unique()):
        yield "subset", str(subset), "all", df["subset"].eq(subset).to_numpy()
    for category in sorted(df["manipulation_category"].unique()):
        yield (
            "category",
            "all",
            str(category),
            df["manipulation_category"].eq(category).to_numpy(),
        )


def rfc_metric_rows(result: dict[str, object], df: pd.DataFrame):
    predictions = np.asarray(result["y_pred"], dtype=int)
    probabilities = np.asarray(result["probability"], dtype=float)
    rows = []
    for scope, subset, category, mask in rfc_groups(df):
        rows.append(
            {
                "scope": scope,
                "subset": subset,
                "manipulation_category": category,
                "rows": int(mask.sum()),
                "misleading_recall": predictions[mask].mean(),
                "mean_probability_misleading": probabilities[mask].mean(),
            }
        )
    return rows


def summarise_rfc(seed_metrics: pd.DataFrame):
    groups = [
        "dataset",
        "model",
        "scope",
        "subset",
        "manipulation_category",
    ]
    summary = (
        seed_metrics.groupby(groups, as_index=False)
        .agg(
            repetitions=("seed", "nunique"),
            rows=("rows", "first"),
            misleading_recall_mean=("misleading_recall", "mean"),
            misleading_recall_std=("misleading_recall", "std"),
            mean_probability_mean=("mean_probability_misleading", "mean"),
        )
    )
    overall = seed_metrics[
        (seed_metrics["scope"] == "overall")
        & (seed_metrics["subset"] == "all")
        & (seed_metrics["manipulation_category"] == "all")
    ]
    rows = []
    for dataset in sorted(overall["dataset"].unique()):
        pivot = overall[overall["dataset"] == dataset].pivot(
            index="seed",
            columns="model",
            values="misleading_recall",
        ).dropna()
        text = pivot["text_only"].to_numpy(dtype=float)
        emotion = pivot["emotion_aware"].to_numpy(dtype=float)
        difference = emotion - text
        statistic, p_value, effect = paired_test(
            emotion,
            text,
            "two-sided",
        )
        rows.append(
            {
                "dataset": dataset,
                "metric": "misleading_recall",
                "repetitions": len(pivot),
                "text_only_mean": text.mean(),
                "emotion_aware_mean": emotion.mean(),
                "mean_difference": difference.mean(),
                "emotion_better_repetitions": int((difference > 0).sum()),
                "text_better_repetitions": int((difference < 0).sum()),
                "ties": int((difference == 0).sum()),
                "wilcoxon_statistic": statistic,
                "p_value": p_value,
                "rank_biserial": effect,
            }
        )
    comparison = pd.DataFrame(rows)
    comparison["p_value_holm"] = holm(comparison["p_value"].tolist())
    comparison["significant_holm_0_05"] = comparison["p_value_holm"] < 0.05
    return summary, comparison
