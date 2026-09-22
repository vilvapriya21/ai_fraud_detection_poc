"""Reusable preprocessing for the bank fraud transaction dataset."""

from collections.abc import Sequence

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "is_fraud"

IDENTIFIER_COLUMNS = (
    "transaction_id",
    "customer_id",
)

LEAKAGE_COLUMNS = (
    "fraud_type",
)

RAW_TIME_COLUMNS = (
    "transaction_date",
    "transaction_time",
)

EXCLUDED_FEATURE_COLUMNS = (
    *IDENTIFIER_COLUMNS,
    *LEAKAGE_COLUMNS,
    *RAW_TIME_COLUMNS,
)


def remove_exact_duplicates(data: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the dataset with exact duplicate rows removed."""

    return data.drop_duplicates().reset_index(drop=True)


def split_features_and_target(
    data: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    excluded_columns: Sequence[str] = EXCLUDED_FEATURE_COLUMNS,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate predictive features from the supervised fraud target.

    The function excludes record identifiers, known leakage fields, and raw
    date/time strings that already have derived numeric representations.

    Args:
        data: Raw or deduplicated transaction data.
        target_column: Binary fraud target column.
        excluded_columns: Columns that must not be used as model features.

    Returns:
        A tuple containing the feature DataFrame and target Series.

    Raises:
        ValueError: If the target or required excluded columns are missing.
    """

    required_columns = {target_column, *excluded_columns}
    missing_columns = required_columns.difference(data.columns)

    if missing_columns:
        missing_names = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing_names}")

    columns_to_drop = [target_column, *excluded_columns]

    features = data.drop(columns=columns_to_drop).copy()
    target = data[target_column].copy()

    return features, target


def prepare_features_and_target(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Remove exact duplicates and prepare model features and fraud target."""

    deduplicated_data = remove_exact_duplicates(data)
    return split_features_and_target(deduplicated_data)


def identify_feature_columns(
    features: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """Identify numeric and categorical columns supported by preprocessing.

    Args:
        features: Model feature DataFrame.

    Returns:
        Numeric column names and categorical column names.

    Raises:
        ValueError: If unsupported column types are present.
    """

    numeric_columns = features.select_dtypes(include="number").columns.tolist()

    categorical_columns = features.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    supported_columns = set(numeric_columns + categorical_columns)

    unsupported_columns = [
        column
        for column in features.columns
        if column not in supported_columns
    ]

    if unsupported_columns:
        unsupported_names = ", ".join(unsupported_columns)
        raise ValueError(
            f"Unsupported feature columns: {unsupported_names}"
        )

    return numeric_columns, categorical_columns


def build_preprocessor(
    features: pd.DataFrame,
    scale_numeric: bool = False,
) -> ColumnTransformer:
    """Build the preprocessing transformer for fraud model features.

    Numeric features use median imputation and can optionally be standardized.

    Categorical features use most-frequent imputation followed by one-hot
    encoding. Unknown categories are ignored during inference.

    Args:
        features: Model feature DataFrame.
        scale_numeric: Whether to standardize numeric features after imputation.

    Returns:
        Configured scikit-learn ColumnTransformer.
    """

    numeric_columns, categorical_columns = identify_feature_columns(features)

    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median")),
    ]

    if scale_numeric:
        numeric_steps.append(
            ("scaler", StandardScaler())
        )

    numeric_pipeline = Pipeline(
        steps=numeric_steps
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_columns,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
