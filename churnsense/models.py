"""Model zoo, evaluation and the experiments shown on the Models page."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42

# The deployed model. Kept small so the exported JSON stays light for the browser.
LGBM_PARAMS = dict(n_estimators=150, learning_rate=0.1, num_leaves=31, min_child_samples=50, random_state=SEED, verbose=-1)


def lightgbm(**overrides):
    from lightgbm import LGBMClassifier

    return LGBMClassifier(**{**LGBM_PARAMS, **overrides})


def candidates() -> dict:
    from xgboost import XGBClassifier

    return {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "Random Forest": RandomForestClassifier(n_estimators=100, min_samples_leaf=5, n_jobs=-1, random_state=SEED),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, eval_metric="logloss", random_state=SEED, n_jobs=-1),
        "LightGBM": lightgbm(),
    }


def scores(y_true, prob, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true)
    prob = np.clip(np.asarray(prob, dtype=float), 1e-7, 1 - 1e-7)
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, prob),
        "pr_auc": average_precision_score(y_true, prob),
        "brier": brier_score_loss(y_true, prob),
        "log_loss": log_loss(y_true, prob),
        "errors": int(fp + fn),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def fit_and_score(name, model, X_train, y_train, X_test, y_test) -> tuple[dict, np.ndarray]:
    start = time.perf_counter()
    model.fit(X_train, y_train)
    seconds = time.perf_counter() - start
    prob = model.predict_proba(X_test)[:, 1]
    return {"model": name, "train_seconds": round(seconds, 2), **scores(y_test, prob)}, prob


def downsample_curve(x, y, n: int = 120) -> list[list[float]]:
    x, y = np.asarray(x), np.asarray(y)
    idx = np.unique(np.linspace(0, len(x) - 1, min(n, len(x))).astype(int))
    return [[round(float(a), 5), round(float(b), 5)] for a, b in zip(x[idx], y[idx])]


def curves(y_true, prob) -> dict:
    fpr, tpr, _ = roc_curve(y_true, prob)
    precision, recall, _ = precision_recall_curve(y_true, prob)
    return {"roc": downsample_curve(fpr, tpr), "pr": downsample_curve(recall[::-1], precision[::-1])}


def calibration(y_true, prob, bins: int = 10) -> list[dict]:
    y_true, prob = np.asarray(y_true), np.asarray(prob)
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(prob, edges[1:-1]), 0, bins - 1)
    out = []
    for b in range(bins):
        m = which == b
        if m.sum():
            out.append({"predicted": float(prob[m].mean()), "observed": float(y_true[m].mean()), "count": int(m.sum())})
    return out


def smote_experiment(X_train, y_train, X_test, y_test) -> dict:
    """Same LightGBM with and without SMOTE: ranking barely moves, calibration gets worse."""
    from imblearn.over_sampling import SMOTE

    X_res, y_res = SMOTE(random_state=SEED).fit_resample(X_train, y_train)
    plain = lightgbm().fit(X_train, y_train).predict_proba(X_test)[:, 1]
    smoted = lightgbm().fit(X_res, y_res).predict_proba(X_test)[:, 1]
    return {
        "train_churn_rate": float(np.mean(y_train)),
        "smote_churn_rate": float(np.mean(y_res)),
        "without_smote": {**pick(scores(y_test, plain)), "calibration": calibration(y_test, plain)},
        "with_smote": {**pick(scores(y_test, smoted)), "calibration": calibration(y_test, smoted)},
    }


def pick(s: dict) -> dict:
    return {k: s[k] for k in ("accuracy", "roc_auc", "brier", "log_loss", "errors")}


def feature_ablation(X_train, y_train, X_test, y_test, add_engineered) -> list[dict]:
    """Does the notebook's feature engineering help? Compared on two model families."""
    Xtr_e, Xte_e = add_engineered(X_train), add_engineered(X_test)
    rows = []
    for family, make in [
        ("Logistic Regression", lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))),
        ("LightGBM", lightgbm),
    ]:
        for label, tr, te in [("Raw features", X_train, X_test), ("Raw + engineered", Xtr_e, Xte_e)]:
            prob = make().fit(tr, y_train).predict_proba(te)[:, 1]
            rows.append({"model": family, "features": label, **pick(scores(y_test, prob))})
    return rows


def cross_validated_auc(model_factory, X, y, folds: int = 5) -> dict:
    from sklearn.model_selection import StratifiedKFold

    aucs = []
    for tr, va in StratifiedKFold(folds, shuffle=True, random_state=SEED).split(X, y):
        m = model_factory().fit(X.iloc[tr], y.iloc[tr])
        aucs.append(roc_auc_score(y.iloc[va], m.predict_proba(X.iloc[va])[:, 1]))
    return {"folds": folds, "mean": float(np.mean(aucs)), "std": float(np.std(aucs)), "values": [float(a) for a in aucs]}


def as_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{k: v for k, v in r.items() if k != "confusion"} for r in rows])
