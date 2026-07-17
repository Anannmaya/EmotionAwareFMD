from __future__ import annotations

from pathlib import Path
import pandas as pd


INPUT = Path("data/interim/liar2_finance_candidates_high_precision.csv")
OUT_DIR = Path("data/processed")
REPORT_DIR = Path("outputs/tables")

SPLIT_MAP = {"train": "train", "valid": "validation", "test": "test"}


def infer_topic(row: pd.Series) -> str:
    text = " ".join(
        str(row.get(col, ""))
        for col in [
            "subject",
            "matched_financial_subjects",
            "matched_high_confidence_terms",
            "matched_supporting_terms",
            "statement",
        ]
    ).lower()

    rules = [
        ("taxation", ["tax", "capital gains", "property tax"]),
        ("public_finance", ["budget", "deficit", "debt", "spending", "bailout", "stimulus"]),
        ("labour_and_income", ["job", "employment", "unemployment", "wage", "salary", "income", "worker", "labor"]),
        ("macroeconomy", ["economy", "economic growth", "gdp", "inflation", "recession", "cost of living", "gas price"]),
        ("trade", ["trade", "tariff"]),
        ("business_and_markets", ["business", "corporation", "company", "stock", "market", "profit", "revenue", "dividend", "bankrupt"]),
        ("banking_and_credit", ["bank", "credit", "loan", "mortgage", "interest rate", "bond", "financial crisis"]),
        ("social_welfare", ["social security", "pension"]),
    ]

    for topic, keywords in rules:
        if any(keyword in text for keyword in keywords):
            return topic

    return "other_finance"


def choose_match(false_row: pd.Series, available_true: pd.DataFrame) -> int:
    candidates = available_true.copy()
    candidates["topic_penalty"] = (
        candidates["broad_topic"] != false_row["broad_topic"]
    ).astype(int) * 10
    candidates["length_difference"] = (
        candidates["word_count"] - false_row["word_count"]
    ).abs()
    candidates["score"] = (
        candidates["topic_penalty"] + candidates["length_difference"]
    )

    return int(
        candidates.sort_values(
            ["score", "length_difference", "id"]
        ).index[0]
    )


def match_split(split_df: pd.DataFrame, split_name: str):
    false_df = split_df[split_df["binary_label"] == 0].copy()
    true_df = split_df[split_df["binary_label"] == 1].copy()

    if len(true_df) < len(false_df):
        raise RuntimeError(
            f"{split_name}: insufficient genuine claims "
            f"({len(true_df)}) for misleading claims ({len(false_df)})."
        )

    true_topic_counts = true_df["broad_topic"].value_counts()
    false_df["topic_availability"] = (
        false_df["broad_topic"].map(true_topic_counts).fillna(0)
    )
    false_df = false_df.sort_values(
        ["topic_availability", "broad_topic", "word_count", "id"]
    )

    remaining_true = true_df.copy()
    selected = []
    diagnostics = []

    for pair_number, (_, false_row) in enumerate(false_df.iterrows(), start=1):
        true_index = choose_match(false_row, remaining_true)
        true_row = remaining_true.loc[true_index]

        pair_id = f"{split_name}_pair_{pair_number:04d}"

        false_copy = false_row.copy()
        true_copy = true_row.copy()
        false_copy["matched_pair_id"] = pair_id
        true_copy["matched_pair_id"] = pair_id

        selected.extend([false_copy, true_copy])

        diagnostics.append(
            {
                "split": split_name,
                "matched_pair_id": pair_id,
                "misleading_id": int(false_row["id"]),
                "genuine_id": int(true_row["id"]),
                "misleading_topic": false_row["broad_topic"],
                "genuine_topic": true_row["broad_topic"],
                "exact_topic_match": (
                    false_row["broad_topic"] == true_row["broad_topic"]
                ),
                "misleading_word_count": int(false_row["word_count"]),
                "genuine_word_count": int(true_row["word_count"]),
                "absolute_word_difference": int(
                    abs(false_row["word_count"] - true_row["word_count"])
                ),
            }
        )

        remaining_true = remaining_true.drop(index=true_index)

    return (
        pd.DataFrame(selected).reset_index(drop=True),
        pd.DataFrame(diagnostics),
    )


def validate(df: pd.DataFrame) -> None:
    if df["id"].duplicated().any():
        raise ValueError("Duplicate IDs found in final dataset.")

    if df["statement_normalised"].duplicated().any():
        raise ValueError("Duplicate normalised statements found.")

    for split_name, group in df.groupby("split"):
        counts = group["binary_label"].value_counts()
        if counts.get(0, 0) != counts.get(1, 0):
            raise ValueError(f"{split_name} is not class-balanced.")

    pair_check = df.groupby("matched_pair_id")["binary_label"].agg(
        ["count", "nunique", "sum"]
    )
    invalid = pair_check[
        (pair_check["count"] != 2)
        | (pair_check["nunique"] != 2)
        | (pair_check["sum"] != 1)
    ]
    if not invalid.empty:
        raise ValueError("Invalid matched pairs found.")


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT)
    df["broad_topic"] = df.apply(infer_topic, axis=1)

    matched_parts = []
    diagnostics_parts = []

    for original_split in ["train", "valid", "test"]:
        split_df = df[df["original_split"] == original_split].copy()
        matched, diagnostics = match_split(
            split_df,
            SPLIT_MAP[original_split],
        )
        matched_parts.append(matched)
        diagnostics_parts.append(diagnostics)

    final_df = pd.concat(matched_parts, ignore_index=True)
    diagnostics_df = pd.concat(diagnostics_parts, ignore_index=True)
    final_df["split"] = final_df["original_split"].map(SPLIT_MAP)

    final_df = (
        final_df.groupby("split", group_keys=False)
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )

    validate(final_df)

    audit_columns = [
        "id",
        "statement",
        "binary_label",
        "binary_label_name",
        "label",
        "label_name",
        "split",
        "original_split",
        "matched_pair_id",
        "broad_topic",
        "subject",
        "speaker",
        "date",
        "word_count",
        "statement_normalised",
        "matched_financial_subjects",
        "matched_high_confidence_terms",
        "matched_supporting_terms",
        "financial_match_source",
    ]

    model_columns = [
        "id",
        "statement",
        "binary_label",
        "binary_label_name",
        "split",
        "matched_pair_id",
        "broad_topic",
        "word_count",
    ]

    audit_df = final_df[audit_columns].copy()
    model_df = audit_df[model_columns].copy()

    audit_path = OUT_DIR / "liar2_finance_full_audit.csv"
    full_path = OUT_DIR / "liar2_finance_full.csv"
    diagnostics_path = REPORT_DIR / "liar2_finance_matching_diagnostics.csv"
    summary_path = REPORT_DIR / "liar2_finance_final_summary.csv"
    topic_path = REPORT_DIR / "liar2_finance_topic_distribution.csv"

    audit_df.to_csv(audit_path, index=False)
    model_df.to_csv(full_path, index=False)
    diagnostics_df.to_csv(diagnostics_path, index=False)

    for split_name in ["train", "validation", "test"]:
        model_df[model_df["split"] == split_name].to_csv(
            OUT_DIR / f"liar2_finance_{split_name}.csv",
            index=False,
        )

    summary_rows = []
    for split_name in ["train", "validation", "test", "all"]:
        subset = model_df if split_name == "all" else model_df[
            model_df["split"] == split_name
        ]
        diag = diagnostics_df if split_name == "all" else diagnostics_df[
            diagnostics_df["split"] == split_name
        ]

        summary_rows.append(
            {
                "split": split_name,
                "total_rows": len(subset),
                "genuine_rows": int((subset["binary_label"] == 1).sum()),
                "misleading_rows": int((subset["binary_label"] == 0).sum()),
                "exact_topic_match_rate": round(
                    float(diag["exact_topic_match"].mean()), 4
                ),
                "mean_word_difference": round(
                    float(diag["absolute_word_difference"].mean()), 3
                ),
                "median_word_difference": float(
                    diag["absolute_word_difference"].median()
                ),
                "maximum_word_difference": int(
                    diag["absolute_word_difference"].max()
                ),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_path, index=False)

    topic_df = (
        audit_df.groupby(
            ["split", "broad_topic", "binary_label_name"]
        )
        .size()
        .reset_index(name="count")
    )
    topic_df.to_csv(topic_path, index=False)

    print("\n" + "=" * 72)
    print("FINAL LIAR2-FINANCE DATASET CREATED")
    print("=" * 72)

    print("\nSummary:")
    print(summary_df.to_string(index=False))

    print("\nClass distribution:")
    print(
        pd.crosstab(
            model_df["split"],
            model_df["binary_label_name"],
        ).to_string()
    )

    print("\nSaved files:")
    for path in [
        audit_path,
        full_path,
        OUT_DIR / "liar2_finance_train.csv",
        OUT_DIR / "liar2_finance_validation.csv",
        OUT_DIR / "liar2_finance_test.csv",
        diagnostics_path,
        summary_path,
        topic_path,
    ]:
        print(path)


if __name__ == "__main__":
    main()
