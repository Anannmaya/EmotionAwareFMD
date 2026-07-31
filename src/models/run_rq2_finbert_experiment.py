"""Run the final paired FinBERT versus FinBERT-plus-affect RQ2 experiment."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict

import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer

from src.models.finbert_rq2_analysis import (
    comparison_table,
    rfc_metric_rows,
    summarise_rfc,
)
from src.models.finbert_rq2_io import (
    CONFIG_PATH,
    DIRECTIONAL_PATH,
    HISTORY_PATH,
    METRICS_PATH,
    PREDICTIONS_PATH,
    RFC_COMPARISON_PATH,
    RFC_METRICS_PATH,
    RFC_PREDICTIONS_PATH,
    RFC_SUMMARY_PATH,
    SPLITS_PATH,
    STATISTICS_PATH,
    load_rfc,
    make_dataset,
    make_rfc_dataset,
    parse_args,
    read_csv,
    save_rows,
)
from src.models.finbert_rq2_utils import (
    MODEL_VARIANTS,
    TrainingConfig,
    choose_device,
    evaluate,
    make_loader,
    pair_preserving_split,
    train,
)
from src.models.run_rq2_cross_validation import (
    AFFECT_COLUMNS,
    DATASETS,
    PREDICTION_DIR,
    TABLE_DIR,
    load_and_validate_dataset,
)


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    training_config = TrainingConfig(
        checkpoint=args.checkpoint,
        max_length=args.max_length,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_epochs=args.max_epochs,
        patience=args.patience,
        dropout=args.dropout,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        num_workers=args.num_workers,
    )
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(
        json.dumps(
            {
                "training_config": asdict(training_config),
                "datasets": args.datasets,
                "seeds": args.seeds,
                "split": "pair-preserving 80/10/10",
                "rfc_enabled": not args.no_rfc,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    device = choose_device()
    tokenizer = AutoTokenizer.from_pretrained(training_config.checkpoint)
    print(f"Device: {device}")
    print(f"FinBERT checkpoint: {training_config.checkpoint}")

    rfc = None if args.no_rfc else load_rfc()
    metrics = read_csv(METRICS_PATH)
    history = read_csv(HISTORY_PATH)
    splits = read_csv(SPLITS_PATH)
    predictions = read_csv(PREDICTIONS_PATH)
    rfc_metrics = read_csv(RFC_METRICS_PATH)
    rfc_predictions = read_csv(RFC_PREDICTIONS_PATH)

    completed = set()
    if args.resume and not metrics.empty:
        completed = {
            (str(row.dataset), int(row.seed), str(row.model))
            for row in metrics.itertuples()
        }
        if rfc is not None:
            rfc_completed = {
                (str(row.dataset), int(row.seed), str(row.model))
                for row in rfc_metrics[
                    rfc_metrics["scope"] == "overall"
                ].itertuples()
            }
            completed &= rfc_completed

    selected_configs = [
        config for config in DATASETS if config.name in set(args.datasets)
    ]
    for dataset_config in selected_configs:
        df = load_and_validate_dataset(dataset_config)
        for repetition, seed in enumerate(args.seeds, start=1):
            train_idx, validation_idx, test_idx, assignment = (
                pair_preserving_split(df, dataset_config.pair_column, seed)
            )
            splits = save_rows(
                splits,
                assignment.assign(
                    dataset=dataset_config.name,
                    repetition=repetition,
                    seed=seed,
                ).to_dict("records"),
                SPLITS_PATH,
                ["dataset", "seed", "pair_id"],
            )
            scaler = StandardScaler().fit(df.loc[train_idx, AFFECT_COLUMNS])
            train_data = make_dataset(
                df,
                train_idx,
                dataset_config,
                tokenizer,
                training_config,
                scaler,
            )
            validation_data = make_dataset(
                df,
                validation_idx,
                dataset_config,
                tokenizer,
                training_config,
                scaler,
            )
            test_data = make_dataset(
                df,
                test_idx,
                dataset_config,
                tokenizer,
                training_config,
                scaler,
            )
            external_data = (
                make_rfc_dataset(rfc, tokenizer, training_config, scaler)
                if rfc is not None
                else None
            )

            for model_variant in MODEL_VARIANTS:
                key = (dataset_config.name, seed, model_variant)
                if key in completed:
                    print(f"Skipping completed run: {key}")
                    continue
                print(
                    f"\n{dataset_config.name} | repetition "
                    f"{repetition}/{len(args.seeds)} | {model_variant}"
                )
                started = time.time()
                model, epoch_history, best_epoch = train(
                    train_data,
                    validation_data,
                    model_variant,
                    training_config,
                    seed,
                    device,
                )
                test_result = evaluate(
                    model,
                    make_loader(test_data, training_config, False, seed),
                    device,
                )
                metrics = save_rows(
                    metrics,
                    [
                        {
                            "dataset": dataset_config.name,
                            "repetition": repetition,
                            "seed": seed,
                            "model": model_variant,
                            "best_epoch": best_epoch,
                            "duration_seconds": time.time() - started,
                            **test_result["metrics"],
                        }
                    ],
                    METRICS_PATH,
                    ["dataset", "seed", "model"],
                )
                history = save_rows(
                    history,
                    [
                        {
                            "dataset": dataset_config.name,
                            "repetition": repetition,
                            "seed": seed,
                            "model": model_variant,
                            "best_epoch": best_epoch,
                            **row,
                        }
                        for row in epoch_history
                    ],
                    HISTORY_PATH,
                    ["dataset", "seed", "model", "epoch"],
                )
                predictions = save_rows(
                    predictions,
                    [
                        {
                            "dataset": dataset_config.name,
                            "repetition": repetition,
                            "seed": seed,
                            "model": model_variant,
                            "row_id": row_id,
                            "pair_id": pair_id,
                            "y_true": int(y_true),
                            "y_pred": int(y_pred),
                            "probability_misleading": float(probability),
                        }
                        for row_id, pair_id, y_true, y_pred, probability in zip(
                            test_result["row_ids"],
                            test_result["pair_ids"],
                            test_result["y_true"],
                            test_result["y_pred"],
                            test_result["probability"],
                        )
                    ],
                    PREDICTIONS_PATH,
                    ["dataset", "seed", "model", "row_id"],
                )

                if external_data is not None and rfc is not None:
                    external_result = evaluate(
                        model,
                        make_loader(
                            external_data,
                            training_config,
                            False,
                            seed,
                        ),
                        device,
                    )
                    rfc_metrics = save_rows(
                        rfc_metrics,
                        [
                            {
                                "dataset": dataset_config.name,
                                "repetition": repetition,
                                "seed": seed,
                                "model": model_variant,
                                **row,
                            }
                            for row in rfc_metric_rows(external_result, rfc)
                        ],
                        RFC_METRICS_PATH,
                        [
                            "dataset",
                            "seed",
                            "model",
                            "scope",
                            "subset",
                            "manipulation_category",
                        ],
                    )
                    rfc_predictions = save_rows(
                        rfc_predictions,
                        [
                            {
                                "dataset": dataset_config.name,
                                "repetition": repetition,
                                "seed": seed,
                                "model": model_variant,
                                "rfc_id": row_id,
                                "subset": str(rfc.iloc[index]["subset"]),
                                "manipulation_category": str(
                                    rfc.iloc[index]["manipulation_category"]
                                ),
                                "y_pred": int(y_pred),
                                "probability_misleading": float(probability),
                            }
                            for index, (row_id, y_pred, probability) in enumerate(
                                zip(
                                    external_result["row_ids"],
                                    external_result["y_pred"],
                                    external_result["probability"],
                                )
                            )
                        ],
                        RFC_PREDICTIONS_PATH,
                        ["dataset", "seed", "model", "rfc_id"],
                    )

                print(
                    f"Test macro-F1={test_result['metrics']['macro_f1']:.4f}; "
                    f"best epoch={best_epoch}"
                )
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    two_sided = comparison_table(metrics, directional=False)
    directional = comparison_table(metrics, directional=True)
    two_sided.to_csv(STATISTICS_PATH, index=False)
    directional.to_csv(DIRECTIONAL_PATH, index=False)
    print("\nTwo-sided comparison:")
    print(two_sided.to_string(index=False))

    if not rfc_metrics.empty:
        rfc_summary, rfc_comparison = summarise_rfc(rfc_metrics)
        rfc_summary.to_csv(RFC_SUMMARY_PATH, index=False)
        rfc_comparison.to_csv(RFC_COMPARISON_PATH, index=False)
        print("\nRFC comparison:")
        print(rfc_comparison.to_string(index=False))

    print("\nFinBERT experiment complete.")


if __name__ == "__main__":
    main()
