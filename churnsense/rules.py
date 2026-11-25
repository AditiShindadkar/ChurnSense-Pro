"""A five-rule, no-ML baseline.

The dataset is synthetic: churn was generated from a handful of hard cut-offs.
These rules were found by looking at churn rate per feature bin (see the
Explore page). Every customer they flag churned. Any ML model has to be
judged against this baseline, not against a coin flip.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RULES = [
    ("Contract Length is Monthly", lambda d: d["Contract Length"] == "Monthly"),
    ("Support Calls > 5", lambda d: d["Support Calls"] > 5),
    ("Payment Delay > 20 days", lambda d: d["Payment Delay"] > 20),
    ("Total Spend < 500", lambda d: d["Total Spend"] < 500),
    ("Age > 50", lambda d: d["Age"] > 50),
]


def rule_flags(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({name: fn(df).to_numpy() for name, fn in RULES}, index=df.index)


def predict(df: pd.DataFrame) -> np.ndarray:
    """1 if any rule fires, else 0."""
    return rule_flags(df).any(axis=1).astype(int).to_numpy()


def rule_report(df: pd.DataFrame, target: str = "Churn") -> list[dict]:
    flags = rule_flags(df)
    rows = []
    for name in flags.columns:
        hit = flags[name]
        rows.append(
            {
                "rule": name,
                "share_of_customers": float(hit.mean()),
                "churn_rate_when_true": float(df.loc[hit, target].mean()),
                "churn_rate_when_false": float(df.loc[~hit, target].mean()),
            }
        )
    any_hit = flags.any(axis=1)
    rows.append(
        {
            "rule": "Any rule",
            "share_of_customers": float(any_hit.mean()),
            "churn_rate_when_true": float(df.loc[any_hit, target].mean()),
            "churn_rate_when_false": float(df.loc[~any_hit, target].mean()),
        }
    )
    return rows
