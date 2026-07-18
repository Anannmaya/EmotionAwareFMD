"""Prepare the public RFC-Bench release for robustness evaluation.

The current public release contains manipulated financial articles only.
All prepared rows therefore receive target=1, meaning misleading.
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "rfc_bench"
OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rfc_bench_public_manipulated.csv"
)


FILES = [
    (
        RAW_DIR / "gold-label-set" / "causal_gold_label-1.csv",
        "main",
        "causal",
        "Perturbed",
    ),
    (
        RAW_DIR / "gold-label-set" / "flipping_gold_label-1.csv",
        "main",
        "flipping",
        "Perturbed",
    ),
    (
        RAW_DIR / "gold-label-set" / "numerical_gold_label-1.csv",
        "main",
        "numerical",
        "Perturbed",
    ),
    (
        RAW_DIR / "gold-label-set" / "sentiment_gold_label-1.csv",
        "main",
        "sentiment",
        "Perturbed",
    ),
    (
        RAW_DIR / "hard-case-set" / "causal_hard_case.csv",
        "hard_case",
        "causal",
        "perturbed",
    ),
    (
        RAW_DIR / "hard-case-set" / "flipping_hard_case.csv",
        "hard_case",
        "flipping",
        "Perturbed",
    ),
    (
        RAW_DIR / "hard-case-set" / "numerical_hard_case.csv",
        "hard_case",
        "numerical",
        "Perturbed",
    ),
    (
        RAW_DIR / "hard-case-set" / "sentiment_hard_case.csv",
        "hard_case",
        "sentiment",
        "Perturbed",
    ),
]


def main() -> None:
    rows = []

    for path, subset, category, text_column in FILES:
        if not path.exists():
            raise FileNotFoundError(f"Missing RFC file: {path}")

        df = pd.read_csv(path)

        required_columns = {"Ticker", "Link", text_column}
        missing = required_columns.difference(df.columns)

        if missing:
            raise ValueError(
                f"{path.name} is missing columns: {sorted(missing)}"
            )

        prepared = pd.DataFrame(
            {
                "ticker": df["Ticker"],
                "date": (
                    df["Date"]
                    if "Date" in df.columns
                    else pd.Series([pd.NA] * len(df))
                ),
                "title": (
                    df["Title"]
                    if "Title" in df.columns
                    else pd.Series([pd.NA] * len(df))
                ),
                "link": df["Link"],
                "text": df[text_column],
                "subset": subset,
                "manipulation_category": category,
                "target": 1,
                "label": "misleading",
            }
        )

        prepared["source_file"] = path.name
        rows.append(prepared)

    combined = pd.concat(rows, ignore_index=True)

    if combined["text"].isna().any():
        raise ValueError("RFC-Bench contains missing manipulated text.")

    combined["text"] = combined["text"].astype(str).str.strip()

    if (combined["text"] == "").any():
        raise ValueError("RFC-Bench contains empty manipulated text.")

    combined.insert(
        0,
        "rfc_id",
        [
            f"rfc_{index:04d}"
            for index in range(1, len(combined) + 1)
        ],
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_PATH, index=False)

    summary = (
        combined.groupby(
            ["subset", "manipulation_category"]
        )
        .size()
        .reset_index(name="rows")
    )

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Total rows: {len(combined):,}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()