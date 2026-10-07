import json
from pathlib import Path

import pandas as pd

O = Path(__file__).resolve().parents[1] / "outputs"


def test_churn_outputs():
    s = pd.read_parquet(O / "churn_scores.parquet")
    assert len(s) == 10_000 and s.customer_id.is_unique
    assert s.churn_prob.between(0, 1).all()
    assert sorted(s.churn_decile.unique()) == list(range(1, 11))


def test_churn_metrics():
    m = json.loads((O / "churn_metrics.json").read_text())
    best = m[m["best_model"]]
    assert best["roc_auc_oof"] > 0.80
    assert best["top_decile_lift"] > 2.5
