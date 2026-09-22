"""Create a reproducible stratified working sample from the full fraud dataset."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RAW_DATA_PATH = Path(
    "data/raw/bank_fraud.csv"
)

OUTPUT_DATA_PATH = Path(
    "data/processed/bank_fraud_poc_sample.csv"
)

TARGET_COLUMN = "is_fraud"

SAMPLE_SIZE = 100_000
RANDOM_STATE = 42


def load_target_column(
    path: Path,
) -> pd.Series:
    """Load only the fraud target column from the full dataset."""

    target_data = pd.read_csv(
        path,
        usecols=[TARGET_COLUMN],
    )

    return target_data[TARGET_COLUMN]


def select_stratified_indices(
    target: pd.Series,
    sample_size: int,
) -> list[int]:
    """Select reproducible row indices while preserving fraud distribution."""

    if sample_size >= len(target):
        return target.index.tolist()

    selected_indices, _ = train_test_split(
        target.index,
        train_size=sample_size,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    return sorted(selected_indices.tolist())


def load_selected_rows(
    path: Path,
    selected_indices: list[int],
) -> pd.DataFrame:
    """Load only selected rows from the full CSV."""

    selected_index_set = set(selected_indices)

    data = pd.read_csv(
        path,
        skiprows=lambda row_number: (
            row_number != 0
            and (row_number - 1)
            not in selected_index_set
        ),
    )

    return data


def save_sample(
    data: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save the sampled transaction dataset."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data.to_csv(
        output_path,
        index=False,
    )


def print_summary(
    data: pd.DataFrame,
) -> None:
    """Print sample shape and fraud distribution."""

    print("\nSample shape:")
    print(data.shape)

    print("\nFraud counts:")
    print(
        data[TARGET_COLUMN].value_counts()
    )

    print("\nFraud percentages:")
    print(
        data[TARGET_COLUMN]
        .value_counts(normalize=True)
        .mul(100)
    )


def main() -> None:
    """Create the stratified POC dataset sample."""

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_DATA_PATH}"
        )

    print("Reading target column...")

    target = load_target_column(
        RAW_DATA_PATH
    )

    print(
        f"Full dataset rows: {len(target):,}"
    )

    print(
        f"Selecting {SAMPLE_SIZE:,} "
        "stratified rows..."
    )

    selected_indices = (
        select_stratified_indices(
            target,
            SAMPLE_SIZE,
        )
    )

    print("Loading selected rows...")

    sample = load_selected_rows(
        RAW_DATA_PATH,
        selected_indices,
    )

    save_sample(
        sample,
        OUTPUT_DATA_PATH,
    )

    print_summary(sample)

    print(
        f"\nSaved sample to: "
        f"{OUTPUT_DATA_PATH}"
    )


if __name__ == "__main__":
    main()
