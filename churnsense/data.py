"""Loading, cleaning and encoding the customer churn dataset.

The encoding is deliberately explicit (fixed column order, fixed category
levels) because the same encoding is re-implemented in the browser
(site/js/model.js). Any change here must be mirrored there; the parity test
in tests_js/ catches drift.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "data" / "customer_churn_dataset-training-master.csv"

TARGET = "Churn"
ID_COLUMN = "CustomerID"

NUMERIC = [
    "Age",
    "Tenure",
    "Usage Frequency",
    "Support Calls",
    "Payment Delay",
    "Total Spend",
    "Last Interaction",
]

# First level of each list is the reference level (dropped in one-hot).
CATEGORICAL = {
    "Gender": ["Female", "Male"],
    "Subscription Type": ["Basic", "Premium", "Standard"],
    "Contract Length": ["Annual", "Monthly", "Quarterly"],
}

RAW_COLUMNS = NUMERIC + list(CATEGORICAL)


def one_hot_names() -> list[str]:
    return [f"{col} = {level}" for col, levels in CATEGORICAL.items() for level in levels[1:]]


FEATURES = NUMERIC + one_hot_names()

# Behavioural features carried over from the original ChurnSense-Pro notebook.
ENGINEERED = ["Spend per Tenure", "Engagement Score", "High Support User", "Payment Risk"]


def source_feature(encoded_name: str) -> str:
    """Map an encoded column back to the raw column it came from."""
    return encoded_name.split(" = ")[0]


def load_raw(path: str | Path = DEFAULT_CSV) -> pd.DataFrame:
    """Read the CSV, drop the ID column, rows with missing values and duplicates."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df = df.drop(columns=[ID_COLUMN], errors="ignore")
    df = df.dropna().drop_duplicates().reset_index(drop=True)
    for col in NUMERIC:
        df[col] = df[col].astype(float)
    df[TARGET] = df[TARGET].astype(int)
    return df


def validate(df: pd.DataFrame) -> None:
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    for col, levels in CATEGORICAL.items():
        unknown = set(df[col].unique()) - set(levels)
        if unknown:
            raise ValueError(f"Unknown values in {col!r}: {sorted(unknown)}")


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """Raw columns -> model features (numeric as-is, categoricals one-hot)."""
    validate(df)
    out = df[NUMERIC].astype(float).copy()
    for col, levels in CATEGORICAL.items():
        for level in levels[1:]:
            out[f"{col} = {level}"] = (df[col] == level).astype(float)
    return out[FEATURES]


def add_engineered(X: pd.DataFrame) -> pd.DataFrame:
    """Append the engineered behavioural features to an encoded frame."""
    out = X.copy()
    out["Spend per Tenure"] = X["Total Spend"] / (X["Tenure"] + 1)
    out["Engagement Score"] = X["Usage Frequency"] / (X["Last Interaction"] + 1)
    out["High Support User"] = (X["Support Calls"] > 3).astype(float)
    out["Payment Risk"] = X["Payment Delay"] * X["Support Calls"]
    return out


def split(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    from sklearn.model_selection import train_test_split

    train, test = train_test_split(df, test_size=test_size, stratify=df[TARGET], random_state=seed)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def numeric_ranges(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    return {
        c: {"min": float(df[c].min()), "max": float(df[c].max()), "median": float(np.median(df[c]))}
        for c in NUMERIC
    }
