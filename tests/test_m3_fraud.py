import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from geopy.distance import great_circle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fraud import build_features, haversine_km  # noqa: E402


def test_haversine_matches_geopy():
    got = haversine_km(48.86, 2.35, 52.52, 13.40)
    want = great_circle((48.86, 2.35), (52.52, 13.40)).km
    assert abs(got - want) / want < 0.01


def test_velocity_and_gap_use_only_past():
    cust = pd.DataFrame({"customer_id": [1], "home_lat": [0.0], "home_lon": [0.0]})
    t = pd.DataFrame({"step": [1, 2, 3, 40], "customer_id": 1, "type": "PAYMENT", "amount": 10.0,
                      "oldbalanceOrg": 100.0, "newbalanceOrig": 90.0, "oldbalanceDest": 0.0,
                      "newbalanceDest": 0.0, "dest_id": "M1", "isFraud": 0,
                      "txn_lat": 0.0, "txn_lon": 0.0})
    f = build_features(t, cust).sort_values("step")
    assert f.velocity_24h.tolist() == [0, 1, 2, 0]
    assert f.time_gap.tolist() == [999, 1, 1, 37]


def test_fraud_outputs():
    O = ROOT / "outputs"
    m = json.loads((O / "fraud_metrics.json").read_text())
    for k in ("rules_only", "random_forest", "keras_mlp"):
        assert 0 <= m[k]["pr_auc"] <= 1
    # models must beat the no-skill baseline (= fraud rate) and the rules-only baseline
    assert m["keras_mlp"]["pr_auc"] > 5 * m["test_fraud_rate"]
    assert m["keras_mlp"]["pr_auc"] > m["rules_only"]["pr_auc"]
    f = pd.read_parquet(O / "fraud_customer_flags.parquet")
    assert len(f) == 10_000 and f.customer_id.is_unique and not f.isna().any().any()
