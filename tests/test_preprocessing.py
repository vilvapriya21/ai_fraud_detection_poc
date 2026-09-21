"""Tests for tabular fraud preprocessing."""

import numpy as np
import pandas as pd

from app.ml.preprocessing import (
    build_preprocessor,
    identify_feature_columns,
    prepare_features_and_target,
    remove_exact_duplicates,
    split_features_and_target,
)


def make_sample_data() -> pd.DataFrame:
    """Create representative raw transactions, including one exact duplicate."""

    rows = [
        {
            "Transaction_ID": "T1",
            "User_ID": 101,
            "Transaction_Amount": 25.0,
            "Transaction_Type": "Purchase",
            "Time_of_Transaction": 10.0,
            "Device_Used": "Mobile",
            "Location": "Urban",
            "Previous_Fraudulent_Transactions": 0,
            "Account_Age": 24,
            "Number_of_Transactions_Last_24H": 2,
            "Payment_Method": "Card",
            "Fraudulent": 0,
        },
        {
            "Transaction_ID": "T2",
            "User_ID": 102,
            "Transaction_Amount": np.nan,
            "Transaction_Type": "Transfer",
            "Time_of_Transaction": np.nan,
            "Device_Used": np.nan,
            "Location": "Rural",
            "Previous_Fraudulent_Transactions": 1,
            "Account_Age": 6,
            "Number_of_Transactions_Last_24H": 8,
            "Payment_Method": np.nan,
            "Fraudulent": 1,
        },
        {
            "Transaction_ID": "T3",
            "User_ID": 103,
            "Transaction_Amount": 80.0,
            "Transaction_Type": "Withdrawal",
            "Time_of_Transaction": 18.0,
            "Device_Used": "Desktop",
            "Location": np.nan,
            "Previous_Fraudulent_Transactions": 2,
            "Account_Age": 48,
            "Number_of_Transactions_Last_24H": 1,
            "Payment_Method": "Bank Transfer",
            "Fraudulent": 0,
        },
    ]
    return pd.DataFrame([*rows, rows[0].copy()])


def dense_values(transformed: object) -> np.ndarray:
    """Convert dense or sparse transformer output to a NumPy array."""

    if hasattr(transformed, "toarray"):
        return transformed.toarray()
    return np.asarray(transformed)


def test_remove_exact_duplicates_preserves_input() -> None:
    data = make_sample_data()

    deduplicated = remove_exact_duplicates(data)

    assert len(data) == 4
    assert len(deduplicated) == 3
    assert deduplicated["Transaction_ID"].tolist() == ["T1", "T2", "T3"]


def test_identifiers_are_excluded_and_target_is_separated() -> None:
    features, target = prepare_features_and_target(make_sample_data())

    assert "Transaction_ID" not in features.columns
    assert "User_ID" not in features.columns
    assert "Fraudulent" not in features.columns
    assert "Previous_Fraudulent_Transactions" in features.columns
    assert target.name == "Fraudulent"
    assert target.tolist() == [0, 1, 0]


def test_split_rejects_missing_required_identifier() -> None:
    data = make_sample_data().drop(columns="User_ID")

    try:
        split_features_and_target(data)
    except ValueError as error:
        assert "User_ID" in str(error)
    else:
        raise AssertionError("Expected a missing-column ValueError")


def test_feature_types_match_observed_schema() -> None:
    features, _ = prepare_features_and_target(make_sample_data())

    numeric_columns, categorical_columns = identify_feature_columns(features)

    assert "Transaction_Amount" in numeric_columns
    assert "Previous_Fraudulent_Transactions" in numeric_columns
    assert "Transaction_Type" in categorical_columns
    assert "Payment_Method" in categorical_columns


def test_preprocessor_handles_numeric_and_categorical_missing_values() -> None:
    features, _ = prepare_features_and_target(make_sample_data())
    preprocessor = build_preprocessor(features)

    transformed = dense_values(preprocessor.fit_transform(features))

    assert transformed.shape[0] == len(features)
    assert not np.isnan(transformed).any()


def test_unseen_categories_do_not_break_transformation() -> None:
    features, _ = prepare_features_and_target(make_sample_data())
    preprocessor = build_preprocessor(features).fit(features)
    unseen = features.iloc[[0]].copy()
    unseen.loc[:, "Transaction_Type"] = "Crypto Purchase"
    unseen.loc[:, "Device_Used"] = "Smart Watch"

    transformed = preprocessor.transform(unseen)

    assert transformed.shape[0] == 1
    assert transformed.shape[1] == len(preprocessor.get_feature_names_out())
