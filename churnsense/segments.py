"""Behavioural customer segments (KMeans) with auto-generated, readable names."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

SEGMENT_FEATURES = ["Tenure", "Usage Frequency", "Support Calls", "Payment Delay", "Total Spend", "Last Interaction"]

# How to phrase a feature that is unusually high / low in a segment.
PHRASES = {
    "Tenure": ("Long-standing", "New"),
    "Usage Frequency": ("Heavy users", "Light users"),
    "Support Calls": ("Frequent support callers", "Rarely call support"),
    "Payment Delay": ("Late payers", "Pay on time"),
    "Total Spend": ("High spenders", "Low spenders"),
    "Last Interaction": ("Gone quiet", "Recently active"),
}


def fit(df: pd.DataFrame, k: int = 4, sample: int = 60000, seed: int = 42) -> dict:
    X = df[SEGMENT_FEATURES].sample(min(sample, len(df)), random_state=seed)
    mean, std = X.mean(), X.std(ddof=0)
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit((X - mean) / std)
    return {
        "features": SEGMENT_FEATURES,
        "mean": mean.round(6).tolist(),
        "std": std.round(6).tolist(),
        "centroids": np.round(km.cluster_centers_, 6).tolist(),
    }


def assign(model: dict, df: pd.DataFrame) -> np.ndarray:
    Z = (df[model["features"]].to_numpy(float) - np.array(model["mean"])) / np.array(model["std"])
    C = np.array(model["centroids"])
    return ((Z[:, None, :] - C[None, :, :]) ** 2).sum(-1).argmin(1)


def name(centroid_z) -> str:
    order = np.argsort(-np.abs(centroid_z))[:2]
    parts = [PHRASES[SEGMENT_FEATURES[i]][0 if centroid_z[i] > 0 else 1] for i in order]
    return " · ".join(parts)


def profile(model: dict, df: pd.DataFrame, prob: np.ndarray, target: str = "Churn") -> list[dict]:
    seg = assign(model, df)
    rows = []
    for k, centroid in enumerate(model["centroids"]):
        m = seg == k
        rows.append(
            {
                "id": k,
                "name": name(np.array(centroid)),
                "share": float(m.mean()),
                "churn_rate": float(df.loc[m, target].mean()),
                "avg_risk": float(prob[m].mean()),
                "monthly_contract_share": float((df.loc[m, "Contract Length"] == "Monthly").mean()),
                "means": {f: float(df.loc[m, f].mean()) for f in SEGMENT_FEATURES},
                "z": {f: float(z) for f, z in zip(SEGMENT_FEATURES, centroid)},
            }
        )
    return rows
