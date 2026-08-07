"""Controlled same-split comparison of classical models and FinBERT.

Purpose
-------
Re-evaluate the classical text-only and emotion-aware model families on the
exact 10 pair-preserving 80/10/10 splits already used by the final FinBERT
experiment. This makes the classical-vs-FinBERT comparison directly comparable
without retraining FinBERT or changing its existing results.

For each dataset and seed:
1. Reuse the saved FinBERT train/validation/test pair assignments.
2. Select the best classical configuration using only the training and
   validation partitions.
3. Fit the selected classical configuration on the training partition only.
4. Evaluate once on the untouched test partition.
5. Compare the resulting classical test score with the already-saved FinBERT
   score from the same seed and test examples.

Existing result files are never overwritten.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

from src.models.run_rq2_cross_validation import (
    DATASETS,
    TABLE_DIR,
    calculate_metrics,
    load_and_validate_dataset,
)
from src.models.run_rq2_classical_nested_model_selection import (
    candidate_sort_key,
    fit_and_predict,
    generate_candidates,
    prepare_split_features,
)


SPLIT_ASSIGNMENTS_PATH = TABLE_DIR / "rq2_finbert_split_assignments.csv"
FINBERT_METRICS_PATH = TABLE_DIR / "rq2_finbert_repetition_metrics.csv"

CLASSICAL_METRICS_PATH = (
    TABLE_DIR / "rq2_same_split_classical_metrics.csv"
)
SUMMARY_PATH = (
    TABLE_DIR / "rq2_same_split_model_family_summary.csv"
)
COMPARISON_PATH = (
    TABLE_DIR / "rq2_same_split_model_family_comparison.csv"
)
SELECTED_CONFIGS_PATH = (
    TABLE_DIR / "rq2_same_split_classical_selected_configs.csv"
)

MODEL_VARIANTS = ["text_only", "emotion_aware"]


def _split_indices(
    df: pd.DataFrame,
    pair_column: str,
    assignments: pd.DataFrame,
    dataset_name: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return row indices for the exact saved FinBERT split."""

    current = assignments[
        (assignments["dataset"] == dataset_name)
        & (assignments["seed"].astype(int) == int(seed))
    ].copy()

    if current.empty:
        raise ValueError(
            f"No saved FinBERT split found for {dataset_name}, seed={seed}."
        )

    if current["pair_id"].duplicated().any():
        duplicates = current.loc[
            current["pair_id"].duplicated(), "pair_id"
        ].tolist()[:10]
        raise ValueError(
            f"Duplicate pair assignments for {dataset_name}, seed={seed}: "
            f"{duplicates}"
        )

    valid_splits = {"train", "validation", "test"}
    observed_splits = set(current["split"].astype(str))
    if observed_splits != valid_splits:
        raise ValueError(
            f"Unexpected split labels for {dataset_name}, seed={seed}: "
            f"{sorted(observed_splits)}"
        )

    pair_values = df[pair_column].astype(str)
    current["pair_id"] = current["pair_id"].astype(str)

    assigned_pairs = set(current["pair_id"])
    dataset_pairs = set(pair_values.unique())

    missing_pairs = dataset_pairs - assigned_pairs
    extra_pairs = assigned_pairs - dataset_pairs
    if missing_pairs or extra_pairs:
        raise ValueError(
            f"Pair mismatch for {dataset_name}, seed={seed}. "
            f"Missing={len(missing_pairs)}, extra={len(extra_pairs)}."
        )

    def indices_for(split_name: str) -> np.ndarray:
        selected_pairs = set(
            current.loc[current["split"] == split_name, "pair_id"]
        )
        return df.index[pair_values.isin(selected_pairs)].to_numpy()

    train_idx = indices_for("train")
    validation_idx = indices_for("validation")
    test_idx = indices_for("test")

    covered = np.concatenate([train_idx, validation_idx, test_idx])
    if len(np.unique(covered)) != len(df):
        raise RuntimeError(
            f"Split coverage problem for {dataset_name}, seed={seed}."
        )

    return train_idx, validation_idx, test_idx


def _select_candidate(
    df: pd.DataFrame,
    config,
    train_idx: np.ndarray,
    validation_idx: np.ndarray,
    model_variant: str,
    seed: int,
):
    """Select a classical candidate using only the saved validation split."""

    validation_features = prepare_split_features(
        df=df,
        config=config,
        train_indices=train_idx,
        evaluation_indices=validation_idx,
    )

    best_candidate = None
    best_key = None
    candidate_rows: list[dict[str, object]] = []

    for candidate_index, candidate in enumerate(
        generate_candidates(model_variant),
        start=1,
    ):
        predictions, _ = fit_and_predict(
            features=validation_features,
            candidate=candidate,
            seed=seed + candidate_index,
        )
        metrics = calculate_metrics(
            validation_features.y_evaluation,
            predictions,
        )

        selection_key = (
            float(metrics["macro_f1"]),
            float(metrics["balanced_accuracy"]),
            *candidate_sort_key(candidate),
        )

        candidate_rows.append(
            {
                "candidate_index": candidate_index,
                "classifier": candidate.classifier,
                "representation": candidate.representation,
                "C": candidate.c_value,
                "affect_weight": (
                    candidate.affect_weight
                    if candidate.affect_weight is not None
                    else np.nan
                ),
                "validation_macro_f1": metrics["macro_f1"],
                "validation_balanced_accuracy": metrics[
                    "balanced_accuracy"
                ],
            }
        )

        if best_key is None or selection_key > best_key:
            best_key = selection_key
            best_candidate = candidate

    if best_candidate is None:
        raise RuntimeError(
            f"No candidate selected for {config.name}, "
            f"{model_variant}, seed={seed}."
        )

    return best_candidate, candidate_rows


def _rank_biserial(differences: np.ndarray) -> float:
    """Paired rank-biserial correlation for FinBERT minus classical."""

    nonzero = differences[np.abs(differences) > 0]
    if len(nonzero) == 0:
        return 0.0

    ranks = rankdata(np.abs(nonzero))
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    denominator = positive + negative

    if denominator == 0:
        return 0.0
    return (positive - negative) / denominator


def _holm_adjust(p_values: list[float]) -> list[float]:
    """Holm-adjust a family of p-values."""

    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m, dtype=float)

    running_max = 0.0
    for rank, index in enumerate(order):
        candidate = (m - rank) * float(p_values[index])
        running_max = max(running_max, candidate)
        adjusted[index] = min(1.0, running_max)

    return adjusted.tolist()


def _paired_model_family_comparisons(
    combined: pd.DataFrame,
) -> pd.DataFrame:
    """Compare FinBERT and classical scores on identical seeds."""

    rows: list[dict[str, object]] = []

    for dataset_name in sorted(combined["dataset"].unique()):
        for model_variant in MODEL_VARIANTS:
            subset = combined[
                (combined["dataset"] == dataset_name)
                & (combined["model"] == model_variant)
            ]

            pivot = subset.pivot(
                index="seed",
                columns="model_family",
                values="macro_f1",
            ).dropna()

            if set(pivot.columns) != {"classical", "finbert"}:
                raise ValueError(
                    f"Missing paired family scores for {dataset_name}, "
                    f"{model_variant}."
                )

            differences = (
                pivot["finbert"].to_numpy()
                - pivot["classical"].to_numpy()
            )

            if np.allclose(differences, 0):
                statistic = 0.0
                p_value = 1.0
            else:
                test = wilcoxon(
                    differences,
                    alternative="two-sided",
                    zero_method="wilcox",
                    method="auto",
                )
                statistic = float(test.statistic)
                p_value = float(test.pvalue)

            rows.append(
                {
                    "dataset": dataset_name,
                    "model": model_variant,
                    "repetitions": len(pivot),
                    "classical_mean_macro_f1": float(
                        pivot["classical"].mean()
                    ),
                    "finbert_mean_macro_f1": float(
                        pivot["finbert"].mean()
                    ),
                    "mean_difference_finbert_minus_classical": float(
                        differences.mean()
                    ),
                    "finbert_better_seeds": int(
                        np.sum(differences > 0)
                    ),
                    "classical_better_seeds": int(
                        np.sum(differences < 0)
                    ),
                    "ties": int(np.sum(differences == 0)),
                    "wilcoxon_statistic": statistic,
                    "p_value": p_value,
                    "rank_biserial": _rank_biserial(differences),
                }
            )

    result = pd.DataFrame(rows)
    result["p_value_holm"] = _holm_adjust(
        result["p_value"].tolist()
    )
    result["significant_holm_0_05"] = (
        result["p_value_holm"] < 0.05
    )
    return result


def main() -> None:
    if not SPLIT_ASSIGNMENTS_PATH.exists():
        raise FileNotFoundError(
            f"Missing saved FinBERT split assignments: "
            f"{SPLIT_ASSIGNMENTS_PATH}"
        )
    if not FINBERT_METRICS_PATH.exists():
        raise FileNotFoundError(
            f"Missing saved FinBERT metrics: {FINBERT_METRICS_PATH}"
        )

    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    assignments = pd.read_csv(SPLIT_ASSIGNMENTS_PATH)
    finbert_metrics = pd.read_csv(FINBERT_METRICS_PATH)

    required_assignment_columns = {
        "pair_id",
        "split",
        "dataset",
        "repetition",
        "seed",
    }
    missing_assignment_columns = (
        required_assignment_columns - set(assignments.columns)
    )
    if missing_assignment_columns:
        raise ValueError(
            "Saved FinBERT split file is missing columns: "
            f"{sorted(missing_assignment_columns)}"
        )

    required_finbert_columns = {
        "dataset",
        "repetition",
        "seed",
        "model",
        "macro_f1",
        "balanced_accuracy",
    }
    missing_finbert_columns = (
        required_finbert_columns - set(finbert_metrics.columns)
    )
    if missing_finbert_columns:
        raise ValueError(
            "Saved FinBERT metrics file is missing columns: "
            f"{sorted(missing_finbert_columns)}"
        )

    classical_rows: list[dict[str, object]] = []
    selected_config_rows: list[dict[str, object]] = []

    for config in DATASETS:
        df = load_and_validate_dataset(config)

        dataset_assignments = assignments[
            assignments["dataset"] == config.name
        ]
        seeds = sorted(
            dataset_assignments["seed"].astype(int).unique().tolist()
        )

        if len(seeds) != 10:
            raise ValueError(
                f"Expected 10 FinBERT seeds for {config.name}; "
                f"found {len(seeds)}: {seeds}"
            )

        print(f"\n=== {config.name} ===")

        for repetition, seed in enumerate(seeds, start=1):
            train_idx, validation_idx, test_idx = _split_indices(
                df=df,
                pair_column=config.pair_column,
                assignments=assignments,
                dataset_name=config.name,
                seed=seed,
            )

            print(
                f"Seed {seed:>3} ({repetition}/10): "
                f"train={len(train_idx)}, "
                f"validation={len(validation_idx)}, "
                f"test={len(test_idx)}"
            )

            for model_offset, model_variant in enumerate(
                MODEL_VARIANTS,
                start=1,
            ):
                best_candidate, _ = _select_candidate(
                    df=df,
                    config=config,
                    train_idx=train_idx,
                    validation_idx=validation_idx,
                    model_variant=model_variant,
                    seed=seed,
                )

                test_features = prepare_split_features(
                    df=df,
                    config=config,
                    train_indices=train_idx,
                    evaluation_indices=test_idx,
                )

                predictions, _ = fit_and_predict(
                    features=test_features,
                    candidate=best_candidate,
                    seed=seed + 10_000 + model_offset,
                )
                metrics = calculate_metrics(
                    test_features.y_evaluation,
                    predictions,
                )

                classical_rows.append(
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
                        **metrics,
                    }
                )

                selected_config_rows.append(
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

                print(
                    f"  {model_variant:<13} "
                    f"{best_candidate.classifier:<20} "
                    f"{best_candidate.representation:<9} "
                    f"macro-F1={metrics['macro_f1']:.4f}"
                )

    classical_metrics = pd.DataFrame(classical_rows)
    selected_configs = pd.DataFrame(selected_config_rows)

    classical_metrics.to_csv(CLASSICAL_METRICS_PATH, index=False)
    selected_configs.to_csv(SELECTED_CONFIGS_PATH, index=False)

    classical_for_comparison = classical_metrics[
        [
            "dataset",
            "repetition",
            "seed",
            "model",
            "macro_f1",
            "balanced_accuracy",
        ]
    ].copy()
    classical_for_comparison["model_family"] = "classical"

    finbert_for_comparison = finbert_metrics[
        [
            "dataset",
            "repetition",
            "seed",
            "model",
            "macro_f1",
            "balanced_accuracy",
        ]
    ].copy()
    finbert_for_comparison["model_family"] = "finbert"

    finbert_for_comparison = finbert_for_comparison[
        finbert_for_comparison["model"].isin(MODEL_VARIANTS)
    ]

    combined = pd.concat(
        [classical_for_comparison, finbert_for_comparison],
        ignore_index=True,
    )

    summary = (
        combined.groupby(
            ["dataset", "model", "model_family"],
            as_index=False,
        )
        .agg(
            repetitions=("seed", "nunique"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
        )
        .sort_values(["dataset", "model", "model_family"])
    )
    summary.to_csv(SUMMARY_PATH, index=False)

    comparisons = _paired_model_family_comparisons(combined)
    comparisons.to_csv(COMPARISON_PATH, index=False)

    print("\n=== SAME-SPLIT MODEL FAMILY SUMMARY ===")
    print(summary.to_string(index=False))

    print("\n=== FINBERT MINUS CLASSICAL: PAIRED MACRO-F1 COMPARISON ===")
    print(comparisons.to_string(index=False))

    print("\nSaved:")
    print(f"  {CLASSICAL_METRICS_PATH}")
    print(f"  {SELECTED_CONFIGS_PATH}")
    print(f"  {SUMMARY_PATH}")
    print(f"  {COMPARISON_PATH}")


if __name__ == "__main__":
    main()
