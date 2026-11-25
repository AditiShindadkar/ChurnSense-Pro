"""Train, evaluate and export everything the website needs.

    python scripts/train.py            # full run (a few minutes)
    python scripts/train.py --fast     # skips Random Forest/XGBoost and CV, for quick iteration

Writes:
    site/data/model.json      LightGBM trees + input schema (browser inference)
    site/data/insights.json   metrics, curves, EDA, rules, segments, SHAP, experiments
    site/data/sample.json     held-out test customers with scores (retention planner)
    tests/fixtures/parity.json  Python predictions + SHAP values to check the JS port
    reports/metrics.md        model comparison table
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from churnsense import data, models, rules, segments  # noqa: E402
from churnsense.export import lightgbm_to_json  # noqa: E402

SITE_DATA = ROOT / "site" / "data"
FIXTURES = ROOT / "tests" / "fixtures"
REPORTS = ROOT / "reports"


def write_json(path: Path, obj, indent=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, indent=indent, separators=None if indent else (",", ":"), allow_nan=False)
    path.write_text(text, encoding="utf-8")
    print(f"  wrote {path.relative_to(ROOT)} ({len(text) / 1024:.0f} KB)")


def r(x, nd=5):
    return round(float(x), nd)


def eda(df: pd.DataFrame) -> dict:
    out = {}
    for col in data.NUMERIC:
        lo, hi = df[col].min(), df[col].max()
        edges = np.unique(np.round(np.linspace(lo, hi + 1e-9, 16)))
        which = np.clip(np.digitize(df[col], edges[1:-1]), 0, len(edges) - 2)
        rows = []
        for b in range(len(edges) - 1):
            m = which == b
            if m.sum():
                rows.append({"label": f"{edges[b]:g}–{edges[b + 1]:g}", "count": int(m.sum()), "churn_rate": r(df.loc[m, "Churn"].mean(), 4)})
        out[col] = rows
    for col, levels in data.CATEGORICAL.items():
        out[col] = [
            {"label": lv, "count": int((df[col] == lv).sum()), "churn_rate": r(df.loc[df[col] == lv, "Churn"].mean(), 4)}
            for lv in levels
        ]
    return out


def partial_dependence(model, df: pd.DataFrame, n_rows: int = 2000) -> dict:
    base = df.sample(n_rows, random_state=1)
    out = {}
    for col in data.NUMERIC:
        grid = np.unique(np.round(np.linspace(df[col].min(), df[col].max(), 25)))
        vals = []
        for g in grid:
            tmp = base.copy()
            tmp[col] = float(g)
            vals.append(r(model.predict_proba(data.encode(tmp))[:, 1].mean(), 4))
        out[col] = {"x": [float(g) for g in grid], "y": vals}
    for col, levels in data.CATEGORICAL.items():
        vals = []
        for lv in levels:
            tmp = base.copy()
            tmp[col] = lv
            vals.append(r(model.predict_proba(data.encode(tmp))[:, 1].mean(), 4))
        out[col] = {"x": levels, "y": vals}
    return out


def global_shap(model, X: pd.DataFrame) -> dict:
    contrib = model.predict(X, pred_contrib=True)[:, :-1]
    by_encoded = np.abs(contrib).mean(0)
    signed = pd.DataFrame(contrib, columns=X.columns)
    # One-hot columns of the same raw feature are summed before taking |.|
    grouped = signed.T.groupby(lambda c: data.source_feature(c)).sum().T
    ranked = sorted(((c, float(grouped[c].abs().mean())) for c in grouped.columns), key=lambda kv: -kv[1])
    return {
        "unit": "mean |SHAP| in log-odds",
        "by_feature": [{"feature": k, "value": r(v, 4)} for k, v in ranked],
        "by_encoded": [{"feature": c, "value": r(v, 4)} for c, v in zip(X.columns, by_encoded)],
    }


def main(fast: bool) -> None:
    print("Loading data...")
    df = data.load_raw()
    train, test = data.split(df)
    X_train, X_test = data.encode(train), data.encode(test)
    y_train, y_test = train["Churn"], test["Churn"]
    print(f"  {len(df):,} customers | train {len(train):,} | test {len(test):,} | churn rate {df['Churn'].mean():.1%}")

    print("Rule baseline...")
    rule_prob = rules.predict(test).astype(float)
    results = [{"model": "5-rule baseline (no ML)", "train_seconds": 0.0, **models.scores(y_test, rule_prob)}]
    curves = {"5-rule baseline (no ML)": models.curves(y_test, rule_prob)}

    print("Training models...")
    zoo = models.candidates()
    if fast:
        zoo = {k: v for k, v in zoo.items() if k in ("Logistic Regression", "LightGBM")}
    probs = {}
    for name, model in zoo.items():
        row, prob = models.fit_and_score(name, model, X_train, y_train, X_test, y_test)
        print(f"  {name:<20} acc={row['accuracy']:.5f} auc={row['roc_auc']:.6f} errors={row['errors']}")
        results.append(row)
        curves[name] = models.curves(y_test, prob)
        probs[name] = prob

    deployed = zoo["LightGBM"]
    prob = probs["LightGBM"]

    print("Experiments...")
    smote = models.smote_experiment(X_train, y_train, X_test, y_test)
    ablation = models.feature_ablation(X_train, y_train, X_test, y_test, data.add_engineered)
    cv = None if fast else models.cross_validated_auc(models.lightgbm, X_train, y_train)

    print("Segments, SHAP, EDA...")
    seg_model = segments.fit(train)
    seg_profiles = segments.profile(seg_model, test, prob)
    shap = global_shap(deployed, X_test.sample(5000, random_state=0))
    pdp = partial_dependence(deployed, test)

    print("Exporting...")
    model_json = lightgbm_to_json(deployed, data.FEATURES)
    model_json["schema"] = {
        "numeric": data.NUMERIC,
        "categorical": data.CATEGORICAL,
        "ranges": data.numeric_ranges(df),
    }
    model_json["base_rate"] = r(y_train.mean(), 4)
    write_json(SITE_DATA / "model.json", model_json)

    rng = np.random.default_rng(7)
    idx = np.sort(rng.choice(len(test), 5000, replace=False))
    seg_ids = segments.assign(seg_model, test)
    sample = {
        "columns": data.RAW_COLUMNS + ["Churn", "risk", "segment"],
        "rows": [
            [*(float(v) if c in data.NUMERIC else v for c, v in zip(data.RAW_COLUMNS, test.loc[i, data.RAW_COLUMNS])),
             int(y_test[i]), r(prob[i], 5), int(seg_ids[i])]
            for i in idx
        ],
        "of_test_rows": len(test),
    }
    write_json(SITE_DATA / "sample.json", sample)

    def clean(row):
        return {k: (r(v, 6) if isinstance(v, (float, np.floating)) else v) for k, v in row.items()}

    insights = {
        "dataset": {
            "rows": len(df),
            "train_rows": len(train),
            "test_rows": len(test),
            "churn_rate": r(df["Churn"].mean(), 4),
            "features": data.RAW_COLUMNS,
        },
        "models": [clean(x) for x in results],
        "deployed_model": "LightGBM",
        "curves": curves,
        "calibration": models.calibration(y_test, prob),
        "rules": [clean(x) for x in rules.rule_report(df)],
        "smote": smote,
        "ablation": [clean(x) for x in ablation],
        "cv": cv,
        "shap": shap,
        "pdp": pdp,
        "eda": eda(df),
        "segments": {"model": seg_model, "profiles": seg_profiles},
    }
    write_json(SITE_DATA / "insights.json", insights)

    parity_rows = test.sample(300, random_state=3)
    Xp = data.encode(parity_rows)
    contrib = deployed.predict(Xp, pred_contrib=True)
    write_json(
        FIXTURES / "parity.json",
        {
            "rows": parity_rows[data.RAW_COLUMNS].to_dict(orient="records"),
            "prob": deployed.predict_proba(Xp)[:, 1].tolist(),
            "contrib": contrib[:, :-1].tolist(),
            "expected_value": contrib[:, -1].tolist(),
        },
    )

    cols = ["model", "accuracy", "precision", "recall", "f1", "roc_auc", "brier", "errors", "train_seconds"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in results:
        cells = [f"{row[c]:.5f}" if isinstance(row[c], float) and c != "train_seconds" else str(row[c]) for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "metrics.md").write_text(
        f"# Model comparison\n\nHeld-out test set: {len(test):,} customers. Generated by `scripts/train.py`.\n\n" + "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true")
    main(parser.parse_args().fast)
