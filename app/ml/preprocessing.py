"""Reusable preprocessing for the tabular fraud dataset.

`Previous_Fraudulent_Transactions` is retained as a candidate historical feature.
Its definition and target relationship must still be reviewed for potential data
leakage before final model selection.
"""

from collections.abc import Sequence

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

TARGET_COLUMN = "Fraudulent"
IDENTIFIER_COLUMNS = ("Transaction_ID", "User_ID")


def remove_exact_duplicates(data: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the data with exact duplicate rows removed.

    Duplicate removal occurs before identifiers are excluded so that distinct
    transactions are not incorrectly treated as duplicates.
    """

    return data.drop_duplicates().reset_index(drop=True)


def split_features_and_target(
    data: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    identifier_columns: Sequence[str] = IDENTIFIER_COLUMNS,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate model features and target while excluding identifiers.

    Args:
        data: Raw or deduplicated transaction data.
        target_column: Name of the supervised target column.
        identifier_columns: Columns that identify records or users rather than
            provide predictive measurements.

    Raises:
        ValueError: If the target or any required identifier column is absent.
    """

    required_columns = {target_column, *identifier_columns}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        missing_names = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing_names}")

    excluded_columns = [target_column, *identifier_columns]
    features = data.drop(columns=excluded_columns).copy()
    target = data[target_column].copy()
    return features, target


def prepare_features_and_target(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Remove exact duplicates, then return identifier-free features and target."""

    deduplicated_data = remove_exact_duplicates(data)
    return split_features_and_target(deduplicated_data)


def identify_feature_columns(
    features: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """Return numeric and categorical feature names supported by the pipeline.

    Raises:
        ValueError: If a feature has an unsupported data type.
    """

    numeric_columns = features.select_dtypes(include="number").columns.tolist()
    categorical_columns = features.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    supported_columns = set(numeric_columns + categorical_columns)
    unsupported_columns = [
        column for column in features.columns if column not in supported_columns
    ]
    if unsupported_columns:
        unsupported_names = ", ".join(unsupported_columns)
        raise ValueError(f"Unsupported feature columns: {unsupported_names}")

    return numeric_columns, categorical_columns


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Build an unfitted preprocessor from the feature DataFrame schema.

    Numeric values are median-imputed without scaling. Categorical values are
    most-frequent-imputed and one-hot encoded. Unknown categories are ignored
    during transformation so inference data does not fail on new levels.
    """

    numeric_columns, categorical_columns = identify_feature_columns(features)

    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
