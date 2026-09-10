# Final FinBERT RQ2 Experiment

This repository contains the final paired neural experiment for RQ2:

- **Text-only:** end-to-end fine-tuned `ProsusAI/finbert`.
- **Emotion-aware:** the identical FinBERT encoder with the five standardised affective scores concatenated to the final `[CLS]` representation before the binary classification layer.

## Evaluation design

- LIAR2 Finance and FACTors Finance are evaluated separately.
- Ten fixed seeds are used.
- Each seed creates a pair-preserving 80/10/10 train, validation and test split.
- Both variants receive the exact same split and training settings.
- Validation macro-F1 controls early stopping.
- Test macro-F1 is the primary outcome; balanced accuracy is secondary.
- The script reports a two-sided paired Wilcoxon analysis and a separate directional primary analysis.
- Every trained model is also evaluated on untouched RFC-Bench. RFC is never used for training, early stopping or tuning.

## Run

Use a Colab GPU or another CUDA GPU. From the repository root:

```bash
python -m src.models.run_rq2_finbert_experiment
```

The command can be resumed after interruption:

```bash
python -m src.models.run_rq2_finbert_experiment --resume
```

For a CUDA out-of-memory error, preserve the effective batch size with:

```bash
python -m src.models.run_rq2_finbert_experiment \
  --batch-size 8 \
  --gradient-accumulation-steps 2 \
  --resume
```

Do not combine outputs produced with different training settings. Delete the partial `rq2_finbert_*` output files before changing settings and restarting the final experiment.

## Main outputs

In-domain results:

- `outputs/tables/rq2_finbert_repetition_metrics.csv`
- `outputs/tables/rq2_finbert_statistical_comparison.csv`
- `outputs/tables/rq2_finbert_primary_directional_comparison.csv`
- `outputs/tables/rq2_finbert_training_history.csv`
- `outputs/predictions/rq2_finbert_predictions.csv`

RFC-Bench results:

- `outputs/tables/rq2_finbert_rfc_seed_metrics.csv`
- `outputs/tables/rq2_finbert_rfc_summary.csv`
- `outputs/tables/rq2_finbert_rfc_overall_comparison.csv`
- `outputs/predictions/rq2_finbert_rfc_predictions.csv`

The exact settings are recorded in `outputs/tables/rq2_finbert_run_config.json`.
