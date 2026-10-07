"""M4b - Survival analysis (RG3, 'how soon'): Kaplan-Meier + log-rank, Cox PH, Weibull AFT.

Time-to-event = tenure (+0.5 year convention so tenure 0 is valid), event = Exited.
Outputs per-customer 5-year conditional survival, expected remaining tenure and a
survival-adjusted CLV that combines M4a ('how much') with this module ('how soon'):
  sa_clv = annual_revenue_pred * sum_{t=1..5} [S(k+t)/S(k)] / 1.08^t,  k = tenure + 0.5
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter, WeibullAFTFitter
from lifelines.statistics import multivariate_logrank_test
from lifelines.utils import k_fold_cross_validation

ROOT = Path(__file__).resolve().parents[1]
PROC, OUT = ROOT / "data" / "processed", ROOT / "outputs"
DISC, YEARS = 0.08, 5


def design(c):
    x = pd.DataFrame({
        "Age": (c.Age - 40) / 10, "CreditScore": (c.CreditScore - 650) / 100,
        "Germany": (c.Geography == "Germany").astype(int), "Spain": (c.Geography == "Spain").astype(int),
        "Male": (c.Gender == "Male").astype(int), "Active": c.IsActiveMember,
        "Prod2": (c.NumOfProducts == 2).astype(int), "Prod3plus": (c.NumOfProducts >= 3).astype(int),
        "HasBalance": (c.Balance > 0).astype(int), "Satisfaction": c["Satisfaction Score"]})
    x["duration"] = c.Tenure + 0.5
    x["event"] = c.Exited
    return x


def main():
    OUT.mkdir(exist_ok=True)
    c = pd.read_parquet(PROC / "customers.parquet").merge(
        pd.read_parquet(OUT / "clv_scores.parquet")[["customer_id", "segment", "annual_revenue", "clv_pred"]],
        on="customer_id")
    d = design(c)
    res = {}

    # Kaplan-Meier by segment + multivariate log-rank
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    rows = []
    for seg, g in c.groupby("segment"):
        km = KaplanMeierFitter().fit(d.duration[g.index], d.event[g.index], label=seg)
        km.plot_survival_function(ax=ax[0], ci_show=False)
        med = km.median_survival_time_
        rows.append({"segment": seg, "n": len(g), "churn_rate": g.Exited.mean(),
                     "median_survival_years": None if np.isinf(med) else float(med),
                     "S_at_5y": float(km.predict(5.5)), "S_at_10y": float(km.predict(10.5))})
    ax[0].set_title("Kaplan-Meier by segment"); ax[0].set_xlabel("Tenure (years)")
    pd.DataFrame(rows).to_csv(OUT / "survival_segment_table.csv", index=False)
    lr = multivariate_logrank_test(d.duration, c.segment, d.event)
    res["logrank_segment"] = {"chi2": float(lr.test_statistic), "df": int(lr.degrees_of_freedom),
                              "p_value": float(lr.p_value)}
    for k, g in c.groupby(np.minimum(c.NumOfProducts, 3)):
        KaplanMeierFitter().fit(d.duration[g.index], d.event[g.index],
                                label=f"{k}{'+' if k == 3 else ''} products").plot_survival_function(ax=ax[1], ci_show=False)
    ax[1].set_title("Kaplan-Meier by number of products"); ax[1].set_xlabel("Tenure (years)")
    lr2 = multivariate_logrank_test(d.duration, np.minimum(c.NumOfProducts, 3), d.event)
    res["logrank_products"] = {"chi2": float(lr2.test_statistic), "p_value": float(lr2.p_value)}
    plt.tight_layout(); plt.savefig(OUT / "survival_km_curves.png", dpi=130); plt.close()

    # Cox PH: hazard ratios + cross-validated concordance
    cph = CoxPHFitter(penalizer=0.01).fit(d, "duration", "event")
    cv_c = k_fold_cross_validation(CoxPHFitter(penalizer=0.01), d, "duration", "event", k=5,
                                   scoring_method="concordance_index", seed=42)
    res["cox"] = {"concordance_in_sample": float(cph.concordance_index_),
                  "concordance_cv_mean": float(np.mean(cv_c)), "concordance_cv_std": float(np.std(cv_c))}
    cph.summary[["exp(coef)", "p"]].rename(columns={"exp(coef)": "hazard_ratio"}).to_csv(OUT / "survival_cox_hr.csv")

    # Weibull AFT: extrapolates beyond observed tenure; used for per-customer forecasts
    aft = WeibullAFTFitter(penalizer=0.01).fit(d, "duration", "event")
    res["weibull_aft"] = {"concordance": float(aft.concordance_index_), "AIC": float(aft.AIC_)}
    grid = np.arange(0.5, 16.0, 1.0)  # 0.5 .. 15.5 covers k+t for k in [0.5,10.5], t<=5
    S = aft.predict_survival_function(d.drop(columns=["duration", "event"]), times=grid)  # (times x cust)
    S = S.values.T
    ki = np.rint(d.duration.values - 0.5).astype(int)  # index of k in grid
    disc = np.array([1 / (1 + DISC) ** t for t in range(1, YEARS + 1)])
    cond = np.stack([S[np.arange(len(c)), ki + t] / S[np.arange(len(c)), ki] for t in range(1, YEARS + 1)], 1)
    out = pd.DataFrame({"customer_id": c.customer_id.values, "segment": c.segment.values,
                        "surv_1y": cond[:, 0], "surv_3y": cond[:, 2], "surv_5y": cond[:, 4],
                        "expected_remaining_years_5y": cond.sum(1)})
    annual_pred = c.clv_pred.values / disc.sum()
    out["sa_clv"] = annual_pred * (cond * disc).sum(1)
    out["clv_pred"] = c.clv_pred.values
    out.to_parquet(OUT / "survival_scores.parquet", index=False)
    # sanity: calibration of Weibull vs KM at segment level (predicted vs observed 5y retention for new customers)
    res["mean_surv_5y_pred"] = float(out.surv_5y.mean())
    (OUT / "survival_metrics.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    seg = out.groupby("segment").agg(clv_pred=("clv_pred", "mean"), surv_5y=("surv_5y", "mean"),
                                     exp_years=("expected_remaining_years_5y", "mean"), sa_clv=("sa_clv", "mean"))
    seg.to_csv(OUT / "clv_survival_segment_table.csv")
    print(pd.DataFrame(rows).round(3).to_string(index=False)); print(seg.round(2).to_string())


if __name__ == "__main__":
    main()
