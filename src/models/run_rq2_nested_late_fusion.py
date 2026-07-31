"""Nested repeated late-fusion experiment for RQ2.

A tuned text-only Logistic Regression model is compared with the same text
model combined with a separate emotion-only Logistic Regression model:

    p_fused = (1 - alpha) * p_text + alpha * p_emotion

All text settings, emotion-model regularisation, and alpha are selected only
inside three-fold pair-preserving inner cross-validation. Final evaluation uses
10 repetitions of five-fold pair-preserving outer cross-validation. RFC-Bench
is not used during selection, and existing RQ2 outputs are not overwritten.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression

from src.models.run_rq2_classical_nested_model_selection import (
    SplitFeatures,
    create_inner_pair_splits,
    prepare_split_features,
)
from src.models.run_rq2_cross_validation import (
    DATASETS,
    PREDICTION_DIR,
    SEEDS,
    TABLE_DIR,
    DatasetConfig,
    calculate_metrics,
    calculate_rank_biserial,
    create_pair_preserving_splits,
    holm_adjust,
    load_and_validate_dataset,
)


TEXT_REPRESENTATIONS = ["word", "word_char"]
TEXT_C_VALUES = [0.1, 1.0, 10.0]
EMOTION_C_VALUES = [0.1, 1.0, 10.0]
FUSION_ALPHAS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
LOGISTIC_SETTINGS = {"solver": "liblinear", "max_iter": 2_000}


@dataclass(frozen=True)
class TextCandidate:
    representation: str
    c_value: float


@dataclass(frozen=True)
class FusionCandidate:
    emotion_c: float
    alpha: float


def create_model(c_value: float, seed: int) -> LogisticRegression:
    return LogisticRegression(
        C=c_value,
        random_state=seed,
        **LOGISTIC_SETTINGS,
    )


def to_predictions(probabilities: np.ndarray) -> np.ndarray:
    return (probabilities >= 0.5).astype(int)


def text_probabilities(
    features: SplitFeatures,
    candidate: TextCandidate,
    seed: int,
) -> np.ndarray:
    model = create_model(candidate.c_value, seed)
    model.fit(
        features.train_text[candidate.representation],
        features.y_train,
    )
    return model.predict_proba(
        features.evaluation_text[candidate.representation]
    )[:, 1]


def emotion_probabilities(
    features: SplitFeatures,
    c_value: float,
    seed: int,
) -> np.ndarray:
    model = create_model(c_value, seed)
    model.fit(features.train_affect, features.y_train)
    return model.predict_proba(features.evaluation_affect)[:, 1]


def tune_text(
    feature_sets: list[SplitFeatures],
    dataset: str,
    repetition: int,
    outer_seed: int,
    outer_fold: int,
) -> tuple[TextCandidate, list[dict[str, object]]]:
    candidates = [
        TextCandidate(representation, c_value)
        for representation in TEXT_REPRESENTATIONS
        for c_value in TEXT_C_VALUES
    ]
    rows: list[dict[str, object]] = []
    best_candidate: TextCandidate | None = None
    best_key: tuple[float, float, int, float] | None = None

    for candidate_index, candidate in enumerate(candidates, start=1):
        truth: list[np.ndarray] = []
        predictions: list[np.ndarray] = []
        for inner_fold, features in enumerate(feature_sets, start=1):
            seed = (
                outer_seed
                + outer_fold * 10_000
                + inner_fold * 100
                + candidate_index
            )
            probabilities = text_probabilities(features, candidate, seed)
            truth.append(features.y_evaluation)
            predictions.append(to_predictions(probabilities))

        metrics = calculate_metrics(
            np.concatenate(truth),
            np.concatenate(predictions),
        )
        key = (
            float(metrics["macro_f1"]),
            float(metrics["balanced_accuracy"]),
            1 if candidate.representation == "word" else 0,
            -candidate.c_value,
        )
        rows.append(
            {
                "dataset": dataset,
                "repetition": repetition,
                "outer_seed": outer_seed,
                "outer_fold": outer_fold,
                "stage": "text_selection",
                "representation": candidate.representation,
                "text_C": candidate.c_value,
                "emotion_C": np.nan,
                "alpha": np.nan,
                "inner_macro_f1": metrics["macro_f1"],
                "inner_balanced_accuracy": metrics["balanced_accuracy"],
                "selected": False,
            }
        )
        if best_key is None or key > best_key:
            best_key = key
            best_candidate = candidate

    if best_candidate is None:
        raise RuntimeError("No text candidate selected.")
    for row in rows:
        row["selected"] = (
            row["representation"] == best_candidate.representation
            and float(row["text_C"]) == best_candidate.c_value
        )
    return best_candidate, rows


def pooled_text_probabilities(
    feature_sets: list[SplitFeatures],
    candidate: TextCandidate,
    outer_seed: int,
    outer_fold: int,
) -> tuple[np.ndarray, np.ndarray]:
    truth: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    for inner_fold, features in enumerate(feature_sets, start=1):
        truth.append(features.y_evaluation)
        probabilities.append(
            text_probabilities(
                features,
                candidate,
                outer_seed + outer_fold * 20_000 + inner_fold,
            )
        )
    return np.concatenate(truth), np.concatenate(probabilities)


def tune_fusion(
    feature_sets: list[SplitFeatures],
    truth: np.ndarray,
    text_probs: np.ndarray,
    text_candidate: TextCandidate,
    dataset: str,
    repetition: int,
    outer_seed: int,
    outer_fold: int,
) -> tuple[FusionCandidate, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    best_candidate: FusionCandidate | None = None
    best_key: tuple[float, float, float, float] | None = None

    for emotion_index, emotion_c in enumerate(EMOTION_C_VALUES, start=1):
        emotion_probs = np.concatenate(
            [
                emotion_probabilities(
                    features,
                    emotion_c,
                    outer_seed
                    + outer_fold * 30_000
                    + inner_fold * 100
                    + emotion_index,
                )
                for inner_fold, features in enumerate(feature_sets, start=1)
            ]
        )
        if len(emotion_probs) != len(text_probs):
            raise RuntimeError("Inner fusion predictions are misaligned.")

        for alpha in FUSION_ALPHAS:
            fused = (1.0 - alpha) * text_probs + alpha * emotion_probs
            metrics = calculate_metrics(truth, to_predictions(fused))
            key = (
                float(metrics["macro_f1"]),
                float(metrics["balanced_accuracy"]),
                -alpha,
                -emotion_c,
            )
            rows.append(
                {
                    "dataset": dataset,
                    "repetition": repetition,
                    "outer_seed": outer_seed,
                    "outer_fold": outer_fold,
                    "stage": "fusion_selection",
                    "representation": text_candidate.representation,
                    "text_C": text_candidate.c_value,
                    "emotion_C": emotion_c,
                    "alpha": alpha,
                    "inner_macro_f1": metrics["macro_f1"],
                    "inner_balanced_accuracy": metrics[
                        "balanced_accuracy"
                    ],
                    "selected": False,
                }
            )
            if best_key is None or key > best_key:
                best_key = key
                best_candidate = FusionCandidate(emotion_c, alpha)

    if best_candidate is None:
        raise RuntimeError("No fusion candidate selected.")
    for row in rows:
        row["selected"] = (
            float(row["emotion_C"]) == best_candidate.emotion_c
            and float(row["alpha"]) == best_candidate.alpha
        )
    return best_candidate, rows


def statistical_summary(repetition_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset in repetition_metrics["dataset"].unique():
        subset = repetition_metrics[repetition_metrics["dataset"] == dataset]
        for metric in ["macro_f1", "balanced_accuracy"]:
            paired = subset.pivot(
                index="repetition",
                columns="model",
                values=metric,
            ).dropna()
            text = paired["text_only"].to_numpy()
            fusion = paired["late_fusion"].to_numpy()
            differences = fusion - text
            if np.allclose(differences, 0):
                statistic, p_value = 0.0, 1.0
            else:
                result = wilcoxon(
                    fusion,
                    text,
                    alternative="two-sided",
                    zero_method="wilcox",
                )
                statistic, p_value = (
                    float(result.statistic),
                    float(result.pvalue),
                )
            rows.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "n_repetitions": len(paired),
                    "text_only_mean": text.mean(),
                    "text_only_std": text.std(ddof=1),
                    "late_fusion_mean": fusion.mean(),
                    "late_fusion_std": fusion.std(ddof=1),
                    "mean_difference_fusion_minus_text": differences.mean(),
                    "median_difference_fusion_minus_text": np.median(
                        differences
                    ),
                    "wilcoxon_statistic": statistic,
                    "p_value": p_value,
                    "rank_biserial": calculate_rank_biserial(differences),
                }
            )
    summary = pd.DataFrame(rows)
    summary["p_value_holm"] = holm_adjust(summary["p_value"].tolist())
    summary["significant_holm_0_05"] = summary["p_value_holm"] < 0.05
    return summary


def run_dataset(
    config: DatasetConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = load_and_validate_dataset(config)
    fold_rows: list[dict[str, object]] = []
    repetition_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    tuning_rows: list[dict[str, object]] = []

    for repetition, outer_seed in enumerate(SEEDS, start=1):
        outer_splits = create_pair_preserving_splits(
            df,
            config.pair_column,
            outer_seed,
        )
        repetition_predictions = {
            name: np.full(len(df), -1, dtype=int)
            for name in ["text_only", "late_fusion"]
        }
        repetition_probabilities = {
            name: np.full(len(df), np.nan, dtype=float)
            for name in ["text_only", "late_fusion"]
        }

        for outer_fold, (train_indices, test_indices) in enumerate(
            outer_splits,
            start=1,
        ):
            inner_splits = create_inner_pair_splits(
                df,
                train_indices,
                config.pair_column,
                outer_seed + outer_fold * 10_000,
            )
            feature_sets = [
                prepare_split_features(df, config, inner_train, inner_valid)
                for inner_train, inner_valid in inner_splits
            ]
            best_text, text_rows = tune_text(
                feature_sets,
                config.name,
                repetition,
                outer_seed,
                outer_fold,
            )
            tuning_rows.extend(text_rows)
            inner_truth, inner_text_probs = pooled_text_probabilities(
                feature_sets,
                best_text,
                outer_seed,
                outer_fold,
            )
            best_fusion, fusion_rows = tune_fusion(
                feature_sets,
                inner_truth,
                inner_text_probs,
                best_text,
                config.name,
                repetition,
                outer_seed,
                outer_fold,
            )
            tuning_rows.extend(fusion_rows)
            selected_rows.append(
                {
                    "dataset": config.name,
                    "repetition": repetition,
                    "outer_seed": outer_seed,
                    "outer_fold": outer_fold,
                    "text_representation": best_text.representation,
                    "text_C": best_text.c_value,
                    "emotion_C": best_fusion.emotion_c,
                    "alpha": best_fusion.alpha,
                }
            )

            outer_features = prepare_split_features(
                df,
                config,
                train_indices,
                test_indices,
            )
            text_probs = text_probabilities(
                outer_features,
                best_text,
                outer_seed + outer_fold * 1_000 + 1,
            )
            emotion_probs = emotion_probabilities(
                outer_features,
                best_fusion.emotion_c,
                outer_seed + outer_fold * 1_000 + 2,
            )
            fused_probs = (
                (1.0 - best_fusion.alpha) * text_probs
                + best_fusion.alpha * emotion_probs
            )

            for model_name, probabilities in {
                "text_only": text_probs,
                "late_fusion": fused_probs,
            }.items():
                predictions = to_predictions(probabilities)
                metrics = calculate_metrics(
                    outer_features.y_evaluation,
                    predictions,
                )
                fold_rows.append(
                    {
                        "dataset": config.name,
                        "repetition": repetition,
                        "outer_seed": outer_seed,
                        "outer_fold": outer_fold,
                        "model": model_name,
                        "train_rows": len(train_indices),
                        "test_rows": len(test_indices),
                        "text_representation": best_text.representation,
                        "text_C": best_text.c_value,
                        "emotion_C": (
                            best_fusion.emotion_c
                            if model_name == "late_fusion"
                            else np.nan
                        ),
                        "alpha": (
                            best_fusion.alpha
                            if model_name == "late_fusion"
                            else 0.0
                        ),
                        **metrics,
                    }
                )
                repetition_predictions[model_name][test_indices] = predictions
                repetition_probabilities[model_name][test_indices] = probabilities
                for position, row_index in enumerate(test_indices):
                    prediction_rows.append(
                        {
                            "dataset": config.name,
                            "repetition": repetition,
                            "outer_seed": outer_seed,
                            "outer_fold": outer_fold,
                            "model": model_name,
                            "row_index": int(row_index),
                            "item_id": df.loc[row_index, config.id_column],
                            "pair_id": df.loc[row_index, config.pair_column],
                            "y_true": int(
                                outer_features.y_evaluation[position]
                            ),
                            "y_pred": int(predictions[position]),
                            "probability_misleading": float(
                                probabilities[position]
                            ),
                            "alpha": (
                                best_fusion.alpha
                                if model_name == "late_fusion"
                                else 0.0
                            ),
                        }
                    )

            print(
                f"{config.name}: late fusion repetition {repetition}/"
                f"{len(SEEDS)}, outer fold {outer_fold}/5 complete"
            )

        for model_name in ["text_only", "late_fusion"]:
            predictions = repetition_predictions[model_name]
            probabilities = repetition_probabilities[model_name]
            if np.any(predictions == -1) or np.isnan(probabilities).any():
                raise RuntimeError(
                    f"Incomplete {model_name} outputs for {config.name}, "
                    f"repetition={repetition}."
                )
            repetition_rows.append(
                {
                    "dataset": config.name,
                    "repetition": repetition,
                    "seed": outer_seed,
                    "model": model_name,
                    **calculate_metrics(df["target"].to_numpy(), predictions),
                }
            )

    return (
        pd.DataFrame(fold_rows),
        pd.DataFrame(repetition_rows),
        pd.DataFrame(prediction_rows),
        pd.DataFrame(selected_rows),
        pd.DataFrame(tuning_rows),
    )


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

    results = [run_dataset(config) for config in DATASETS]
    fold_metrics = pd.concat([result[0] for result in results], ignore_index=True)
    repetition_metrics = pd.concat(
        [result[1] for result in results],
        ignore_index=True,
    )
    predictions = pd.concat([result[2] for result in results], ignore_index=True)
    selected = pd.concat([result[3] for result in results], ignore_index=True)
    tuning = pd.concat([result[4] for result in results], ignore_index=True)
    statistics = statistical_summary(repetition_metrics)
    frequency = (
        selected.groupby(
            [
                "dataset",
                "text_representation",
                "text_C",
                "emotion_C",
                "alpha",
            ],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "selected_outer_folds"})
        .sort_values(
            ["dataset", "selected_outer_folds"],
            ascending=[True, False],
        )
    )

    fold_metrics.to_csv(
        TABLE_DIR / "rq2_late_fusion_fold_metrics.csv",
        index=False,
    )
    repetition_metrics.to_csv(
        TABLE_DIR / "rq2_late_fusion_repetition_metrics.csv",
        index=False,
    )
    statistics.to_csv(
        TABLE_DIR / "rq2_late_fusion_statistical_comparison.csv",
        index=False,
    )
    selected.to_csv(
        TABLE_DIR / "rq2_late_fusion_selected_hyperparameters.csv",
        index=False,
    )
    frequency.to_csv(
        TABLE_DIR / "rq2_late_fusion_selection_frequency.csv",
        index=False,
    )
    tuning.to_csv(
        TABLE_DIR / "rq2_late_fusion_inner_tuning_scores.csv",
        index=False,
    )
    predictions.to_csv(
        PREDICTION_DIR / "rq2_late_fusion_oof_predictions.csv",
        index=False,
    )

    print("\nNested late-fusion experiment complete.")
    print(statistics.to_string(index=False))


if __name__ == "__main__":
    main()
