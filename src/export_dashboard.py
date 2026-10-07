"""Export pipeline outputs to dashboard/public/data.json for the React dashboard."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

ROOT = Path(__file__).resolve().parents[1]
OUT, PROC = ROOT / "outputs", ROOT / "data" / "processed"
DEST = ROOT / "dashboard" / "public"


def J(name):
    return json.loads((OUT / name).read_text())


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    act = pd.read_parquet(OUT / "customer_action_list.parquet")
    cust = pd.read_parquet(PROC / "customers.parquet")
    seg = pd.read_parquet(OUT / "clv_scores.parquet")[["customer_id", "segment"]]
    km = {}
    d = cust.merge(seg, on="customer_id")
    for s, g in d.groupby("segment"):
        f = KaplanMeierFitter().fit(g.Tenure + 0.5, g.Exited)
        km[s] = [{"year": float(t), "retained": round(float(v), 4)}
                 for t, v in zip([0.0] + list(range(1, 11)), [1.0] + [float(f.predict(y + 0.5)) for y in range(1, 11)])]
    churn, fraud, clv = J("churn_metrics.json"), J("fraud_metrics.json"), J("clv_metrics.json")
    surv, rec, kpi = J("survival_metrics.json"), J("recommender_metrics.json"), J("integration_kpi.json")
    prov = json.loads((PROC / "provenance.json").read_text())

    cols = ["priority_rank", "customer_id", "Geography", "Age", "Tenure", "NumOfProducts", "Balance", "churn_prob",
            "churn_decile", "fraud_flagged", "clv_pred", "sa_clv", "surv_5y", "rec_product", "value_at_risk",
            "action", "Exited"]
    cl = act[cols].copy()
    for c in ("churn_prob", "surv_5y"):
        cl[c] = cl[c].round(3)
    for c in ("clv_pred", "sa_clv", "value_at_risk", "Balance"):
        cl[c] = cl[c].round(0).astype(int)

    data = {
        "churn": {"models": {k: churn[k] for k in ("logistic_regression", "xgboost")},
                  "deciles": pd.read_csv(OUT / "churn_decile_lift.csv").round(4).to_dict("records")},
        "fraud": {k: fraud[k] for k in ("rules_only", "random_forest", "keras_mlp")} | {
            "test_fraud_rate": fraud["test_fraud_rate"]},
        "clv": {"metrics": clv, "segments": pd.read_csv(OUT / "clv_segment_table.csv").round(0).to_dict("records")},
        "survival": {"metrics": surv, "km": km,
                     "segments": json.loads(pd.read_csv(OUT / "survival_segment_table.csv").round(4).to_json(orient="records")),
                     "clv_vs_survival": pd.read_csv(OUT / "clv_survival_segment_table.csv").round(3).to_dict("records"),
                     "hazard_ratios": pd.read_csv(OUT / "survival_cox_hr.csv").round(4).to_dict("records")},
        "recommender": rec,
        "integration": kpi,
        "provenance": prov,
        "action_list": json.loads(cl.to_json(orient="records")),
    }
    (DEST / "data.json").write_text(json.dumps(data, allow_nan=False))
    print("wrote", (DEST / "data.json").stat().st_size // 1024, "KB;", len(cl), "customers")


if __name__ == "__main__":
    main()
