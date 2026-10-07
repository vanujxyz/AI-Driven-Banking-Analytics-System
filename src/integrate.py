"""M6 - Cross-module synthesis (RG5): one customer-ID-keyed decision layer + ranked action list.

Action rules (first match wins):
  1. FRAUD_REVIEW        fraud module flagged the customer within the 1% alert budget
  2. RETAIN_HIGH_VALUE   churn decile <= 2 and survival-adjusted CLV >= median  (personal outreach)
  3. RETAIN_STANDARD     churn decile <= 2                                       (automated offer)
  4. CROSS_SELL          otherwise, recommend next-best product
Ranking inside the list: fraud first, then expected value at risk = churn_prob * sa_clv.
Business KPI: value of true churners captured in a fixed retention-contact budget, compared with
ranking by churn probability alone and with random contact.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC, OUT = ROOT / "data" / "processed", ROOT / "outputs"
CONTACT_BUDGET = 0.10  # share of customers the retention team can contact


def build():
    c = pd.read_parquet(PROC / "customers.parquet")[["customer_id", "Geography", "Age", "Tenure",
                                                      "NumOfProducts", "Balance", "Exited"]]
    parts = [pd.read_parquet(OUT / "churn_scores.parquet"),
             pd.read_parquet(OUT / "fraud_customer_flags.parquet")[["customer_id", "fraud_score_max", "fraud_alerts", "fraud_flagged"]],
             pd.read_parquet(OUT / "survival_scores.parquet")[["customer_id", "segment", "clv_pred", "sa_clv", "surv_5y",
                                                                "expected_remaining_years_5y"]],
             pd.read_parquet(OUT / "recommendations.parquet")]
    d = c
    for p in parts:
        n = len(d)
        d = d.merge(p, on="customer_id", how="left", validate="one_to_one")
        assert len(d) == n, "join changed row count"
    assert not d.drop(columns=[]).isna().any().any(), "missing values after join"
    d["value_at_risk"] = d.churn_prob * d.sa_clv
    med = d.sa_clv.median()
    d["action"] = np.select(
        [d.fraud_flagged == 1, (d.churn_decile <= 2) & (d.sa_clv >= med), d.churn_decile <= 2],
        ["FRAUD_REVIEW", "RETAIN_HIGH_VALUE", "RETAIN_STANDARD"], default="CROSS_SELL")
    d["action_detail"] = np.where(d.action == "CROSS_SELL", "Offer " + d.rec_product, "")
    d["sort_key"] = np.where(d.action == "FRAUD_REVIEW", 1e12 + d.fraud_score_max,
                             np.where(d.action.str.startswith("RETAIN"), d.value_at_risk, d.value_at_risk * 0.01))
    d = d.sort_values("sort_key", ascending=False).reset_index(drop=True)
    d.insert(0, "priority_rank", np.arange(1, len(d) + 1))
    return d.drop(columns="sort_key")


def retention_kpi(d):
    k = int(CONTACT_BUDGET * len(d))
    rng = np.random.default_rng(42)
    churn_value = d.Exited * d.sa_clv  # value actually lost to churners
    total = churn_value.sum()
    out = {"contact_budget": k, "total_value_lost_to_churn": float(total)}
    for name, order in (("rank_by_value_at_risk", d.value_at_risk.rank(ascending=False, method="first")),
                        ("rank_by_churn_prob_only", d.churn_prob.rank(ascending=False, method="first")),
                        ("random_contact", pd.Series(rng.permutation(len(d)) + 1, index=d.index))):
        sel = order <= k
        out[name] = {"churners_caught": int(d.Exited[sel].sum()), "precision": float(d.Exited[sel].mean()),
                     "value_captured": float(churn_value[sel].sum()), "share_of_value": float(churn_value[sel].sum() / total)}
    return out


def main():
    d = build()
    d.to_parquet(OUT / "customer_action_list.parquet", index=False)
    d.to_csv(OUT / "customer_action_list.csv", index=False)
    kpi = retention_kpi(d)
    kpi["action_counts"] = d.action.value_counts().to_dict()
    (OUT / "integration_kpi.json").write_text(json.dumps(kpi, indent=2))
    print(json.dumps(kpi, indent=2)); print(d.head(8).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
