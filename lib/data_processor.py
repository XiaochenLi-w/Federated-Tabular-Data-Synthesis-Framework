import numpy as np
import pandas as pd
import sklearn.preprocessing
from typing import List, Dict, Tuple


def cat_encode(X):
    """One-hot encode for categorical and ordinal features."""
    return sklearn.preprocessing.OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    ).fit(X)


def normalize(X, normalization="quantile"):
    """Normalize continuous features."""
    if normalization == "standard":
        scaler = sklearn.preprocessing.StandardScaler()
    elif normalization == "minmax":
        scaler = sklearn.preprocessing.MinMaxScaler()
    elif normalization == "quantile":
        scaler = sklearn.preprocessing.QuantileTransformer(
            output_distribution="normal",
            n_quantiles=max(min(X.shape[0] // 30, 1000), 10),
            subsample=int(1e8),
        )
    else:
        raise ValueError(f"Unknown normalization: {normalization}")

    return scaler.fit(X)


def preprocess(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    meta_data: Dict,
    discrete_cols: List[str],
    normalization: str = "quantile",
) -> Tuple:
    """
    Convert dataframe to numpy arrays with encoding and normalization.

    Args:
        train_data: Training data
        val_data: Validation data
        meta_data: Metadata containing task type and other info
        discrete_cols: List of discrete column names
        normalization: Normalization method

    Returns:
        tuple: ([train_x, train_y], [val_x, val_y], encodings)
    """

    def fit_encoder(col, data):
        """Fit appropriate encoder for a column"""
        if col == "label":
            if meta_data["task"] != "regression":
                return sklearn.preprocessing.LabelEncoder().fit(data.ravel())
            return normalize(data.reshape(-1, 1), normalization)
        return (cat_encode if col in discrete_cols else normalize)(
            data.reshape(-1, 1), normalization
        )

    def encode_features(df, encodings):
        """Transform all features in a dataframe"""
        features = [
            encodings[col].transform(df[col].values.reshape(-1, 1))
            for col in df.columns
            if col != "label"
        ]
        return np.concatenate(features, axis=1)

    # Fit encoders on combined data
    combined_data = pd.concat([train_data, val_data], ignore_index=True)
    encodings = {
        col: fit_encoder(col, combined_data[col].values)
        for col in combined_data.columns
    }

    # Transform data
    return (
        [
            encode_features(train_data, encodings),
            encodings["label"].transform(train_data["label"].values.ravel()),
        ],
        [
            encode_features(val_data, encodings),
            encodings["label"].transform(val_data["label"].values.ravel()),
        ],
        encodings,
    )


def transform_data(data: pd.DataFrame, encodings: Dict, meta_data: Dict) -> List:
    """
    Transform data using pre-fitted encodings.

    Args:
        data: Data to transform
        encodings: Pre-fitted encoders
        meta_data: Metadata containing task type and other info

    Returns:
        list: [features, labels]
    """
    features = [
        encodings[col].transform(data[col].values.reshape(-1, 1))
        for col in data.columns
        if col != "label"
    ]

    return [
        np.concatenate(features, axis=1),
        encodings["label"].transform(data["label"].values.ravel()),
    ]
