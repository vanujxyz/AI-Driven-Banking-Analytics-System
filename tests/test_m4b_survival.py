import json
from pathlib import Path

import pandas as pd

O = Path(__file__).resolve().parents[1] / "outputs"


def test_survival_outputs():
    s = pd.read_parquet(O / "survival_scores.parquet")
    assert len(s) == 10_000 and s.customer_id.is_unique and not s.isna().any().any()
    assert s.surv_1y.between(0, 1).all()
    assert (s.surv_1y >= s.surv_3y).all() and (s.surv_3y >= s.surv_5y).all()  # survival is non-increasing
    assert s.expected_remaining_years_5y.between(0, 5).all()
    assert (s.sa_clv <= s.clv_pred + 1e-6).all()  # survival can only discount value


def test_survival_stats():
    m = json.loads((O / "survival_metrics.json").read_text())
    assert m["logrank_segment"]["p_value"] < 0.05
    assert m["cox"]["concordance_cv_mean"] > 0.6
    assert (O / "survival_km_curves.png").exists()
