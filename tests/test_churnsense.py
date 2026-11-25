import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from churnsense import business, data, rules, segments
from churnsense.business import Offer

ROOT = Path(__file__).resolve().parents[1]


def customer(**overrides):
    row = {
        "Age": 30.0, "Gender": "Female", "Tenure": 20.0, "Usage Frequency": 15.0, "Support Calls": 1.0,
        "Payment Delay": 3.0, "Subscription Type": "Basic", "Contract Length": "Annual",
        "Total Spend": 800.0, "Last Interaction": 5.0, "Churn": 0,
    }
    row.update(overrides)
    return row


@pytest.fixture
def frame():
    return pd.DataFrame([
        customer(),
        customer(**{"Contract Length": "Monthly", "Churn": 1}),
        customer(**{"Support Calls": 9.0, "Gender": "Male", "Subscription Type": "Premium", "Churn": 1}),
        customer(**{"Contract Length": "Quarterly", "Subscription Type": "Standard"}),
    ])


def test_encode_has_fixed_columns_and_one_hot(frame):
    X = data.encode(frame)
    assert list(X.columns) == data.FEATURES
    assert X.loc[1, "Contract Length = Monthly"] == 1
    assert X.loc[0, ["Contract Length = Monthly", "Contract Length = Quarterly"]].sum() == 0
    assert X.loc[2, "Gender = Male"] == 1 and X.loc[2, "Subscription Type = Premium"] == 1


def test_encode_rejects_unknown_category(frame):
    frame.loc[0, "Contract Length"] = "Weekly"
    with pytest.raises(ValueError, match="Contract Length"):
        data.encode(frame)


def test_engineered_features(frame):
    X = data.add_engineered(data.encode(frame))
    assert X.loc[0, "Spend per Tenure"] == pytest.approx(800 / 21)
    assert X.loc[2, "High Support User"] == 1
    assert X.loc[2, "Payment Risk"] == pytest.approx(27)


def test_rules_flag_known_churners(frame):
    assert rules.predict(frame).tolist() == [0, 1, 1, 0]


def test_load_raw_drops_id_and_missing(tmp_path):
    path = tmp_path / "d.csv"
    rows = pd.DataFrame([customer(CustomerID=1), customer(CustomerID=2, Age=None)])
    rows.to_csv(path, index=False)
    df = data.load_raw(path)
    assert len(df) == 1 and "CustomerID" not in df.columns


def test_best_offer_worked_example():
    idx, ev = business.best_offers([0.8, 0.01], [1000, 100])
    assert idx.tolist() == [2, -1]
    assert ev[0] == pytest.approx(260) and ev[1] == 0


def test_budget_is_respected():
    chosen, spent = business.plan_with_budget([0.9, 0.9, 0.0], [1000, 1000, 1000], budget=60)
    assert chosen.tolist() == [2, -1, -1] and spent == 60


def test_realised_profit_counts_only_true_churners():
    offers = (Offer("x", 10, 0.5),)
    profit = business.realised_profit([1, 0], [0, 0], [100, 100], offers)
    assert profit == pytest.approx(0.5 * 100 - 20)


def test_lifetime_value_modes():
    assert business.lifetime_value([500], 0.4, 2).tolist() == [400]
    assert business.lifetime_value([500], 0.4, 2, flat_value=99).tolist() == [99]


def test_segment_assignment_is_nearest_centroid():
    df = pd.DataFrame({f: np.r_[np.zeros(50), np.ones(50) * 10] for f in segments.SEGMENT_FEATURES})
    model = segments.fit(df, k=2, sample=100)
    seg = segments.assign(model, df)
    assert len(set(seg[:50])) == 1 and len(set(seg[50:])) == 1 and seg[0] != seg[99]


def test_exported_model_schema_matches_python():
    model = json.loads((ROOT / "site" / "data" / "model.json").read_text(encoding="utf-8"))
    assert model["features"] == data.FEATURES
    assert model["schema"]["categorical"] == data.CATEGORICAL
