import tomli
from typing import Any, Union
from pathlib import Path
import json
import tomli_w
import numpy as np
import pandas as pd
import sklearn.preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, r2_score, roc_auc_score, root_mean_squared_error
import random
import os
import torch
from .data_processor import preprocess, transform_data

from sklearn.preprocessing import LabelEncoder


def read_csv(csv_filename: str, meta_filename: str = None) -> tuple:
    """
    Read a csv file and its metadata.

    Args:
        csv_filename: Path to CSV file
        meta_filename: Path to metadata JSON file

    Returns:
        tuple: (data, meta_data, discrete_cols)
    """
    with open(meta_filename) as meta_file:
        meta_data = json.load(meta_file)

    discrete_cols = [
        column["name"]
        for column in meta_data["columns"]
        if column["type"] != "continuous"
    ]

    data = pd.read_csv(csv_filename, header="infer")
    # Convert discrete columns to string
    for col in discrete_cols:
        data[col] = data[col].astype(str)

    return data, meta_data, discrete_cols


def get_n_class(meta_filename: str) -> int:
    """
    Get number of classes from metadata (-1 if regression).
    """
    meta_data = load_json(meta_filename)
    for column in meta_data["columns"]:
        if column["name"] == "label" and column["type"] != "continuous":
            return column["size"]
    return -1


def load_config(path: Union[Path, str]) -> Any:
    """Load TOML config file."""
    with open(path, "rb") as f:
        return tomli.load(f)


def dump_config(config: dict, path: Union[Path, str]) -> None:
    """Save config to TOML file."""
    with open(path, "wb") as f:
        tomli_w.dump(config, f)


def improve_reproducibility(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_json(path: Union[Path, str]) -> dict:
    """Load JSON file."""
    with open(path) as f:
        return json.load(f)


# --------------------------------------------------------------- #
# ------------------- tools for process data ------------------ #
# --------------------------------------------------------------- #
def cat_encode(X):
    """
    one-hot encode for categorical and ordinal features
    """
    oe = sklearn.preprocessing.OneHotEncoder(
        handle_unknown="ignore",  # type: ignore[code]
        sparse_output=False,  # type: ignore[code]
    ).fit(X)

    return oe


def normalize(X, normalization="quantile"):
    """
    normalize continuous features
    """
    if normalization == "standard":
        scaler = sklearn.preprocessing.StandardScaler()
    elif normalization == "minmax":
        scaler = sklearn.preprocessing.MinMaxScaler()
    elif normalization == "quantile":
        # adopt from Tab-DDPM
        scaler = sklearn.preprocessing.QuantileTransformer(
            output_distribution="normal",
            n_quantiles=max(min(X.shape[0] // 30, 1000), 10),
            subsample=int(1e8),
        )
    else:
        raise ValueError(
            "normalization must be standard, minmax, or quantile, but got "
            + normalization
        )

    scaler.fit(X)
    return scaler


def preprocess(
    train_data, val_data, meta_data, discrete_cols, normalization="quantile"
):
    """
    Convert dataframe to numpy arrays with encoding and normalization.

    Args:
        train_data (pd.DataFrame): Training data
        val_data (pd.DataFrame): Validation data
        meta_data (dict): Metadata containing task type and other info
        discrete_cols (list): List of discrete column names
        normalization (str): Normalization method

    Returns:
        tuple: ([train_x, train_y], [val_x, val_y], encodings)
    """
    encodings = {}

    def fit_encoder(col, data):
        """Fit appropriate encoder for a column"""
        if col == "label":
            if meta_data["task"] != "regression":
                return sklearn.preprocessing.LabelEncoder().fit(data.ravel())
            return normalize(data.reshape(-1, 1), normalization)
        if col in discrete_cols:
            return cat_encode(data.reshape(-1, 1))
        else:
            return normalize(data.reshape(-1, 1), normalization)
        # return (cat_encode if col in discrete_cols else normalize)(
        #     data.reshape(-1, 1), normalization
        # )

    def encode_features(df):
        """Transform all features in a dataframe"""
        features = []
        for col in df.columns:
            if col != "label":
                features.append(encodings[col].transform(df[col].values.reshape(-1, 1)))
        return np.concatenate(features, axis=1)

    # Fit encoders on combined data
    combined_data = pd.concat([train_data, val_data], ignore_index=True)
    encodings = {
        col: fit_encoder(col, combined_data[col].values)
        for col in combined_data.columns
    }

    # Transform data
    # X_label = train_data["label"].values.ravel().reshape(1, -1)
    # X_label_val = val_data["label"].values.ravel().reshape(1, -1)
    le = LabelEncoder()
    return (
        [
            encode_features(train_data),
            #encodings["label"].transform(train_data["label"].values.ravel()),
            le.fit_transform(train_data["label"].values.ravel()),
        ],
        [
            encode_features(val_data),
            #encodings["label"].transform(val_data["label"].values.ravel()),
            le.fit_transform(val_data["label"].values.ravel()),
        ],
        encodings,
    )


def transform_data(data, encodings, meta_data):
    """Transform data using pre-fitted encodings."""
    features = [
        encodings[col].transform(data[col].values.reshape(-1, 1))
        for col in data.columns
        if col != "label"
    ]
    
    le = LabelEncoder()
    return [
        np.concatenate(features, axis=1),
        #encodings["label"].transform(data["label"].values.ravel()),
        le.fit_transform(data["label"].values.ravel())
    ]


# --------------------------------------------------------------- #
# --------------------- tools for ML evaluator ------------------ #
# --------------------------------------------------------------- #
def f1(net, X, y):
    y_pred = net.predict(X)
    return f1_score(y, y_pred, average="weighted")


def r2(net, X, y):
    y_pred = net.predict(X)
    return r2_score(y, y_pred)


def rmse(net, X, y):
    y_pred = net.predict(X)
    return mean_squared_error(y, y_pred, squared=False)


def cal_metrics(
    y_true, y_pred, task_type, pred_prob=None, n_class=None, unique_labels=None
):
    """
    y_prob, n_class, unique_labels are only used in classification task
    n_class: number of classes in real dataset
    unique_labels: unique labels in training dataset (thus its the dimension of the output of the model)
    """
    if task_type == "regression":
        r2 = r2_score(y_true, y_pred)
        rmse = root_mean_squared_error(y_true, y_pred)
        return {
            "r2": r2,
            "rmse": rmse,
        }
    else:
        # f1 only need to evaluate on seen classes
        y_true = y_true.astype(int)
        f1 = f1_score(y_true, y_pred, average="weighted")
        if task_type == "binary_classification":
            roc_auc = roc_auc_score(y_true, pred_prob[:, 1])
        else:
            # roc need to evaluate on all classes
            # fill the probability of unseen class to 0
            rest_label = set(range(n_class)) - set(unique_labels)
            tmp = []
            j = 0
            for i in range(n_class):
                if i in rest_label:
                    # unseen class, we set the probability to 0
                    tmp.append(np.array([0] * y_true.shape[0])[:, np.newaxis])
                else:
                    try:
                        tmp.append(pred_prob[:, [j]])
                    except:
                        tmp.append(pred_prob[:, np.newaxis])
                    j += 1
            filled_pred_prob = np.hstack(tmp)
            filled_y_true = np.eye(n_class)[y_true]
            # if no data in one class of y_true, roc_auc_score will raise error
            # see detail: https://github.com/scikit-learn/scikit-learn/issues/24636
            # get rid of the class with no data from both y_true and filled_pred_prob
            # remove the dimension of all 0
            index = (np.sum(filled_y_true, axis=0) > 0) & (
                np.sum(filled_pred_prob, axis=0) > 0
            )
            # np.save("filled_y_true.npy", filled_y_true)
            # np.save("filled_pred_prob.npy", filled_pred_prob)
            # np.save("index.npy", index)
            filled_y_true = filled_y_true[:, index]
            filled_pred_prob = filled_pred_prob[:, index]
            roc_auc = roc_auc_score(filled_y_true, filled_pred_prob, multi_class="ovr")

        return {
            "roc_auc": roc_auc,
            "f1": f1,
        }
