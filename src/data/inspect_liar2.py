from pathlib import Path

import pandas as pd


DATA_DIR = Path("data/raw/liar2_finance")

FILES = {
    "train": DATA_DIR / "train.csv",
    "valid": DATA_DIR / "valid.csv",
    "test": DATA_DIR / "test.csv",
}


def load_splits() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Load LIAR2 splits and add their original split names."""

    datasets = {}

    for split_name, file_path in FILES.items():
        if not file_path.exists():
            raise FileNotFoundError(f"Missing file: {file_path}")

        dataframe = pd.read_csv(file_path)
        dataframe["original_split"] = split_name
        datasets[split_name] = dataframe

    combined = pd.concat(datasets.values(), ignore_index=True)

    return datasets, combined


def print_split_summary(datasets: dict[str, pd.DataFrame]) -> None:
    print("\n" + "=" * 70)
    print("SPLIT SIZES")
    print("=" * 70)

    for split_name, dataframe in datasets.items():
        print(
            f"{split_name:>5}: "
            f"{len(dataframe):,} rows × {len(dataframe.columns):,} columns"
        )


def check_column_consistency(
    datasets: dict[str, pd.DataFrame],
) -> None:
    print("\n" + "=" * 70)
    print("COLUMN CONSISTENCY")
    print("=" * 70)

    reference_split = next(iter(datasets))
    reference_columns = set(datasets[reference_split].columns)

    for split_name, dataframe in datasets.items():
        current_columns = set(dataframe.columns)

        missing = reference_columns - current_columns
        additional = current_columns - reference_columns

        print(f"\n{split_name}:")

        if not missing and not additional:
            print("Columns match.")
        else:
            print(f"Missing columns: {sorted(missing)}")
            print(f"Additional columns: {sorted(additional)}")


def print_dataset_summary(dataframe: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("COMBINED LIAR2 SUMMARY")
    print("=" * 70)

    print(f"\nTotal rows: {len(dataframe):,}")
    print(f"Total columns: {len(dataframe.columns):,}")

    print("\nColumns:")
    for index, column in enumerate(dataframe.columns, start=1):
        print(f"{index:>2}. {column}")

    print("\nData types:")
    print(dataframe.dtypes.to_string())

    missing_count = dataframe.isna().sum()
    missing_percentage = (
        missing_count / len(dataframe) * 100
    ).round(2)

    missing_summary = pd.DataFrame(
        {
            "missing_count": missing_count,
            "missing_percentage": missing_percentage,
        }
    ).sort_values("missing_count", ascending=False)

    print("\nMissing values:")
    print(missing_summary.to_string())

    print("\nUnique values:")
    for column in dataframe.columns:
        count = dataframe[column].nunique(dropna=False)
        print(f"{column}: {count:,}")

    print("\nExact duplicate rows:")
    print(dataframe.duplicated().sum())


def inspect_possible_label_columns(dataframe: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("POSSIBLE LABEL COLUMNS")
    print("=" * 70)

    keywords = [
        "label",
        "truth",
        "rating",
        "veracity",
        "class",
        "target",
    ]

    possible_columns = [
        column
        for column in dataframe.columns
        if any(keyword in column.lower() for keyword in keywords)
    ]

    if not possible_columns:
        print("No obvious label column found.")
        return

    for column in possible_columns:
        print(f"\nValue counts for '{column}':")
        print(dataframe[column].value_counts(dropna=False).to_string())

        print(f"\nValue counts for '{column}' by original split:")
        print(
            pd.crosstab(
                dataframe["original_split"],
                dataframe[column],
                dropna=False,
            ).to_string()
        )


def inspect_text_columns(dataframe: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("POSSIBLE TEXT COLUMNS")
    print("=" * 70)

    for column in dataframe.columns:
        if dataframe[column].dtype != "object":
            continue

        values = dataframe[column].dropna().astype(str)

        if values.empty:
            continue

        average_length = values.str.len().mean()
        median_length = values.str.len().median()
        maximum_length = values.str.len().max()

        print(
            f"{column}: "
            f"average={average_length:.1f}, "
            f"median={median_length:.1f}, "
            f"maximum={maximum_length}"
        )


def print_sample_rows(dataframe: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("SAMPLE ROWS")
    print("=" * 70)

    sample_size = min(5, len(dataframe))

    sample = dataframe.sample(
        n=sample_size,
        random_state=42,
    )

    print(sample.to_string(index=False))


def main() -> None:
    datasets, combined = load_splits()

    print_split_summary(datasets)
    check_column_consistency(datasets)
    print_dataset_summary(combined)
    inspect_possible_label_columns(combined)
    inspect_text_columns(combined)
    print_sample_rows(combined)

    print("\n" + "=" * 70)
    print("INSPECTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()