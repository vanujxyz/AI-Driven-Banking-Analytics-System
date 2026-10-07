import json
from pathlib import Path

import pandas as pd

O = Path(__file__).resolve().parents[1] / "outputs"


def test_clv_outputs():
    s = pd.read_parquet(O / "clv_scores.parquet")
    assert len(s) == 10_000 and s.customer_id.is_unique
    assert (s.clv_proxy >= 0).all() and (s.clv_pred >= 0).all() and not s.isna().any().any()
    assert set(s.clv_split) == {"train", "test"}
    assert s.segment.nunique() == 6


def test_clv_beats_mean_baseline_on_holdout():
    m = json.loads((O / "clv_metrics.json").read_text())
    assert m["test_rmse"] < m["mean_baseline_rmse"]
    assert (O / "clv_leaderboard.csv").exists() and (O / "clv_segment_table.csv").exists()
