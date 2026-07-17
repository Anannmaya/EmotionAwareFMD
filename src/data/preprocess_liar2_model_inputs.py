from __future__ import annotations

from pathlib import Path
import re

import pandas as pd


INPUT_DIR = Path("data/processed")
OUTPUT_DIR = INPUT_DIR / "model_ready"
TABLES_DIR = Path("outputs/tables")

FILES = {
    "train": INPUT_DIR / "liar2_finance_train.csv",
    "validation": INPUT_DIR / "liar2_finance_validation.csv",
    "test": INPUT_DIR / "liar2_finance_test.csv",
    "full": INPUT_DIR / "liar2_finance_full.csv",
}

# Remove only a leading fact-checking template such as:
#   Says taxes increased.
#   "Says that unemployment doubled."
# It does not remove "says" elsewhere in the claim.
LEADING_SAYS_PATTERN = re.compile(
    r"""^\s*
        ["'“”‘’]*
        \s*
        says
        \s*
        (?:
            that
            \s+
        )?
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)


def clean_statement(text: object) -> str:
    """Remove leading editorial framing and normalise whitespace."""

    cleaned = str(text)

    cleaned = LEADING_SAYS_PATTERN.sub(
        "",
        cleaned,
        count=1,
    )

    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned


def starts_with_says(text: object) -> bool:
    """Check whether a statement still begins with the template."""

    return bool(LEADING_SAYS_PATTERN.search(str(text)))


def preprocess_file(
    split_name: str,
    input_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Clean one LIAR2 file while preserving its rows and labels."""

    if not input_path.exists():
        raise FileNotFoundError(f"Missing input file: {input_path}")

    dataframe = pd.read_csv(input_path)

    required_columns = {
        "id",
        "statement",
        "binary_label",
        "binary_label_name",
    }

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"{input_path} is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    original_row_count = len(dataframe)
    original_ids = dataframe["id"].copy()
    original_labels = dataframe["binary_label"].copy()
    original_statements = dataframe["statement"].astype(str).copy()

    before_mask = original_statements.map(starts_with_says)

    dataframe["statement"] = original_statements.map(clean_statement)

    after_mask = dataframe["statement"].map(starts_with_says)

    # Integrity checks.
    if len(dataframe) != original_row_count:
        raise ValueError(
            f"{split_name}: row count changed during preprocessing."
        )

    if not dataframe["id"].equals(original_ids):
        raise ValueError(
            f"{split_name}: row order or IDs changed."
        )

    if not dataframe["binary_label"].equals(original_labels):
        raise ValueError(
            f"{split_name}: labels changed during preprocessing."
        )

    if dataframe["statement"].eq("").any():
        empty_ids = dataframe.loc[
            dataframe["statement"].eq(""),
            "id",
        ].tolist()

        raise ValueError(
            f"{split_name}: cleaning created empty statements for IDs "
            f"{empty_ids[:10]}"
        )

    if after_mask.any():
        remaining_ids = dataframe.loc[
            after_mask,
            "id",
        ].tolist()

        raise ValueError(
            f"{split_name}: some statements still begin with 'Says': "
            f"{remaining_ids[:10]}"
        )

    changes = pd.DataFrame(
        {
            "split": split_name,
            "id": dataframe["id"],
            "binary_label": dataframe["binary_label"],
            "binary_label_name": dataframe["binary_label_name"],
            "statement_original": original_statements,
            "statement_cleaned": dataframe["statement"],
            "leading_says_removed": before_mask,
        }
    )

    changes = changes[
        changes["leading_says_removed"]
    ].reset_index(drop=True)

    label_summary = (
        pd.DataFrame(
            {
                "binary_label_name": dataframe[
                    "binary_label_name"
                ],
                "leading_says_before": before_mask,
            }
        )
        .groupby("binary_label_name")["leading_says_before"]
        .agg(["sum", "count"])
        .reset_index()
    )

    label_summary["percentage_before"] = (
        100 * label_summary["sum"] / label_summary["count"]
    )

    summary = {
        "split": split_name,
        "rows": original_row_count,
        "statements_changed": int(before_mask.sum()),
        "remaining_leading_says": int(after_mask.sum()),
        "empty_statements": int(
            dataframe["statement"].eq("").sum()
        ),
        "duplicate_ids": int(dataframe["id"].duplicated().sum()),
        "duplicate_statements_after_cleaning": int(
            dataframe["statement"]
            .str.lower()
            .str.strip()
            .duplicated()
            .sum()
        ),
    }

    print("\n" + "=" * 70)
    print(split_name.upper())
    print("=" * 70)
    print(f"Rows: {original_row_count:,}")
    print(f"Leading 'Says' templates removed: {before_mask.sum():,}")
    print("\nBefore cleaning, by label:")
    print(
        label_summary[
            [
                "binary_label_name",
                "sum",
                "count",
                "percentage_before",
            ]
        ]
        .round(2)
        .to_string(index=False)
    )

    return dataframe, changes, summary


def verify_full_matches_splits(
    cleaned_files: dict[str, pd.DataFrame],
) -> None:
    """Check that full.csv contains exactly the same IDs as the three splits."""

    split_ids = set(
        pd.concat(
            [
                cleaned_files["train"][["id"]],
                cleaned_files["validation"][["id"]],
                cleaned_files["test"][["id"]],
            ],
            ignore_index=True,
        )["id"]
    )

    full_ids = set(cleaned_files["full"]["id"])

    if split_ids != full_ids:
        missing_from_full = sorted(split_ids - full_ids)
        extra_in_full = sorted(full_ids - split_ids)

        raise ValueError(
            "The cleaned full file does not match the three cleaned splits. "
            f"Missing from full: {missing_from_full[:10]}; "
            f"extra in full: {extra_in_full[:10]}"
        )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    cleaned_files = {}
    all_changes = []
    all_summaries = []

    for split_name, input_path in FILES.items():
        cleaned, changes, summary = preprocess_file(
            split_name=split_name,
            input_path=input_path,
        )

        cleaned_files[split_name] = cleaned
        all_changes.append(changes)
        all_summaries.append(summary)

    verify_full_matches_splits(cleaned_files)

    output_names = {
        "train": "liar2_finance_train.csv",
        "validation": "liar2_finance_validation.csv",
        "test": "liar2_finance_test.csv",
        "full": "liar2_finance_full.csv",
    }

    for split_name, dataframe in cleaned_files.items():
        output_path = OUTPUT_DIR / output_names[split_name]
        dataframe.to_csv(output_path, index=False)

    changes_df = pd.concat(
        all_changes,
        ignore_index=True,
    )

    changes_path = (
        TABLES_DIR / "liar2_leading_says_changes.csv"
    )
    changes_df.to_csv(changes_path, index=False)

    summary_df = pd.DataFrame(all_summaries)

    summary_path = (
        TABLES_DIR / "liar2_preprocessing_summary.csv"
    )
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)

    print(
        f"\nTotal leading templates removed across the three "
        f"model splits: "
        f"{sum(item['statements_changed'] for item in all_summaries[:3]):,}"
    )

    print("\nModel-ready files:")
    for filename in output_names.values():
        print(OUTPUT_DIR / filename)

    print("\nAudit files:")
    print(changes_path)
    print(summary_path)

    print(
        "\nOriginal files in data/processed were not modified."
    )


if __name__ == "__main__":
    main()
