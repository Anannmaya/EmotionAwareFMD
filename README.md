# Emotion-Aware Financial Misinformation Detection

MSc Artificial Intelligence dissertation project investigating whether emotional information can help detect financial misinformation.

The project studies two questions:

1. **RQ1:** Do genuine and misleading financial texts exhibit different affective patterns?
2. **RQ2:** Can affective information improve financial misinformation detection compared with text-only models?

## Overview

Financial misinformation can influence investment decisions, market behaviour and public trust. This project investigates whether affective signals provide useful information beyond textual content alone.

Five affective dimensions are used:

- Anger
- Fear
- Joy
- Sadness
- Valence

The main experiments compare text-only models with emotion-aware models that combine TF-IDF representations with these affective features.

## Datasets

### LIAR2-Finance

A finance-focused subset constructed from LIAR2 containing 2,048 claims organised into 1,024 genuine–misleading pairs.

### FACTors-Finance

A manually audited finance-focused paired dataset containing 261 genuine–misleading pairs.

### RFC-Bench

Used as an external robustness test. The public RFC-Bench release contains manipulated financial articles across several manipulation categories and is not used during model training or hyperparameter selection.

## Methodology

### RQ1 — Affective analysis

Affective scores are compared between genuine and misleading members of each pair.

Statistical testing uses paired non-parametric tests with Holm correction for multiple comparisons.

### RQ2 — Misinformation detection

The primary classical comparison is:

**Text-only**

TF-IDF + Logistic Regression

**Emotion-aware**

TF-IDF + five standardised affective features + Logistic Regression

Evaluation uses repeated nested pair-preserving cross-validation:

- 10 repetitions
- 5 outer folds
- 3 inner folds for hyperparameter selection
- Macro-F1 as the primary evaluation metric
- Balanced accuracy as a secondary metric
- Wilcoxon signed-rank testing with Holm correction

Pair-preserving splitting ensures that the two members of a genuine–misleading pair never appear on opposite sides of a train/test split.

The affective contribution is also weighted and selected within the inner cross-validation procedure.

## Main Findings

### RQ1

On **LIAR2-Finance**, misleading claims showed significantly:

- higher anger
- higher fear
- higher sadness
- lower joy
- lower valence

On **FACTors-Finance**, anger and fear were significantly higher in misleading text, while differences in joy, sadness and valence were not statistically significant after correction.

### RQ2

For **LIAR2-Finance**, adding affective information improved mean macro-F1:

- Text-only: **0.6528**
- Emotion-aware: **0.6610**
- Difference: **+0.0082**
- Holm-adjusted p-value: **0.0234**

For **FACTors-Finance**, the emotion-aware model did not outperform the text-only model:

- Text-only: **0.6246**
- Emotion-aware: **0.6224**

External evaluation on RFC-Bench showed that the affective augmentation did not generalise reliably outside the training domains and reduced misleading-class recall for models trained on both source datasets.

Overall, the results suggest that affective information can provide useful complementary information in some in-domain settings, but the benefit is dataset-dependent and does not automatically transfer across domains.

## Repository Structure

```text
EmotionAwareFMD/
├── data/
│   ├── features/       # Datasets with extracted affective features
│   ├── interim/        # Required intermediate dataset construction file
│   └── processed/      # Processed and model-ready datasets
│
├── notebooks/
│   └── Affective Features Extraction.ipynb
│
├── outputs/
│   ├── figures/        # Dissertation figures
│   ├── predictions/    # Prediction-level experiment outputs
│   └── tables/         # Statistical tests and experiment results
│
├── src/
│   ├── analysis/       # RQ1, failure analysis and figure generation
│   ├── data/           # Dataset preparation scripts
│   └── models/         # Classical and FinBERT experiments
│
├── FINBERT_EXPERIMENT.md
├── requirements.txt
└── README.md

```

## Installation

Clone the repository and create a Python environment:

```bash
git clone https://github.com/Anannmaya/EmotionAwareFMD.git
cd EmotionAwareFMD

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

## Running the Main Experiments

### RQ1

```bash
python -m src.analysis.analyse_liar2_affective_patterns
python -m src.analysis.analyse_factors_affective_patterns
python -m src.analysis.summarise_rq1_across_datasets
```

### RQ2 — Main nested experiment

```bash
python -m src.models.run_rq2_nested_weighted_cross_validation
```

### RFC-Bench external evaluation

```bash
python -m src.models.run_rq2_rfc_weighted_external_evaluation
```

### RFC-Bench failure analysis

```bash
python -m src.analysis.analyse_rq2_rfc_weighted_failures
python -m src.analysis.compare_rq2_rfc_failure_affective_profiles
```

### Generate final result tables and figures

```bash
python -m src.analysis.create_rq2_final_tables
python -m src.analysis.create_dissertation_final_figures
```

## FinBERT Experiment

A supplementary paired FinBERT experiment compares:

- fine-tuned `ProsusAI/finbert`
- the same FinBERT encoder augmented with the five affective features

The experiment uses fixed pair-preserving train, validation and test splits.

Run with:

```bash
python -m src.models.run_rq2_finbert_experiment
```

A controlled same-split comparison with the classical models is available through:

```bash
python -m src.models.run_rq2_same_split_classical_finbert_comparison
```

See [`FINBERT_EXPERIMENT.md`](FINBERT_EXPERIMENT.md) for full details.

## Outputs

The main dissertation-ready results are stored in:

```text
outputs/tables/rq1_affective_comparison_long.csv
outputs/tables/rq2_final_in_domain_results.csv
outputs/tables/rq2_final_rfc_overall_results.csv
outputs/tables/rq2_final_rfc_category_results.csv
```

Final figures are available under:

```text
outputs/figures/
```

Prediction-level outputs are retained under `outputs/predictions/` to support reproducibility and error analysis.

## Reproducibility

The experiments use fixed random seeds and pair-preserving evaluation procedures. Hyperparameter selection is performed only within training data, while outer test folds and RFC-Bench remain excluded from model selection.

Detailed experiment outputs, selected hyperparameters, statistical comparisons and prediction-level results are included in the repository.
