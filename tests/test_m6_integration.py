import json
from pathlib import Path

import pandas as pd

O = Path(__file__).resolve().parents[1] / "outputs"
ACTIONS = {"FRAUD_REVIEW", "RETAIN_HIGH_VALUE", "RETAIN_STANDARD", "CROSS_SELL"}


def test_action_list_complete_and_ranked():
    d = pd.read_parquet(O / "customer_action_list.parquet")
    assert len(d) == 10_000 and d.customer_id.is_unique and not d.isna().any().any()
    assert d.priority_rank.tolist() == list(range(1, 10_001))
    assert set(d.action) <= ACTIONS
    assert (d.action == "FRAUD_REVIEW").equals((d.fraud_flagged == 1))  # fraud rule has top precedence
    fr = (d.action == "FRAUD_REVIEW").sum()
    assert (d.action.iloc[:fr] == "FRAUD_REVIEW").all()  # fraud block ranked first
    assert d.loc[d.action == "CROSS_SELL", "action_detail"].str.startswith("Offer ").all()


def test_value_ranking_not_worse_than_random():
    k = json.loads((O / "integration_kpi.json").read_text())
    assert k["rank_by_value_at_risk"]["value_captured"] > 2 * k["random_contact"]["value_captured"]
    assert k["rank_by_churn_prob_only"]["precision"] > k["random_contact"]["precision"]
