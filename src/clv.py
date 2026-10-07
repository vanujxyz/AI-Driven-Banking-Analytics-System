"""M4a - Customer lifetime value (RG3, 'how much'): engineered CLV proxy + H2O AutoML regression.

CLV proxy (assumption, documented): 5-year discounted revenue at 8%,
  annual_revenue = 0.02*Balance + 80*NumOfProducts + 40*HasCrCard + 0.002*EstimatedSalary*IsActiveMember
The AutoML model is NOT given the formula inputs (Balance, NumOfProducts, HasCrCard, EstimatedSalary,
Point Earned) or the label (Exited/Complain) -- otherwise it would just re-learn the formula.
"""
import json
import os
import sys
from pathlib import Path

# H2O needs a JDK; the conda env ships one under Library/bin (Windows) or bin (Unix)
for _d in (Path(sys.executable).parent / "Library" / "bin", Path(sys.executable).parent / "bin"):
    if _d.exists():
        os.environ["PATH"] = str(_d) + os.pathsep + os.environ["PATH"]

import h2o
import numpy as np
import pandas as pd
from h2o.automl import H2OAutoML

ROOT = Path(__file__).resolve().parents[1]
PROC, OUT = ROOT / "data" / "processed", ROOT / "outputs"
SEED, DISC, YEARS = 42, 0.08, 5
ANNUITY = sum(1 / (1 + DISC) ** t for t in range(1, YEARS + 1))  # ~3.993
PRED = ["CreditScore", "Geography", "Gender", "Age", "Tenure", "IsActiveMember",
        "Satisfaction Score", "Card Type"]


def add_clv_proxy(c):
    c = c.copy()
    c["annual_revenue"] = (0.02 * c.Balance + 80 * c.NumOfProducts + 40 * c.HasCrCard
                           + 0.002 * c.EstimatedSalary * c.IsActiveMember)
    c["clv_proxy"] = c.annual_revenue * ANNUITY
    c["segment"] = c.Geography + np.where(c.IsActiveMember == 1, "-active", "-inactive")
    return c


def main():
    OUT.mkdir(exist_ok=True)
    c = add_clv_proxy(pd.read_parquet(PROC / "customers.parquet"))
    rng = np.random.default_rng(SEED)
    c["clv_split"] = np.where(rng.random(len(c)) < 0.8, "train", "test")

    h2o.init(nthreads=4, max_mem_size="3G", verbose=False)
    h2o.no_progress()
    cols = PRED + ["clv_proxy"]
    tr = h2o.H2OFrame(c.loc[c.clv_split == "train", cols])
    te = h2o.H2OFrame(c.loc[c.clv_split == "test", cols])
    for f in (tr, te):
        for k in ("Geography", "Gender", "Card Type"):
            f[k] = f[k].asfactor()
    aml = H2OAutoML(max_models=12, max_runtime_secs=240, seed=SEED, sort_metric="RMSE",
                    exclude_algos=["DeepLearning"])
    aml.train(x=PRED, y="clv_proxy", training_frame=tr)
    lb = aml.leaderboard.as_data_frame()
    lb.to_csv(OUT / "clv_leaderboard.csv", index=False)
    print(lb.head(8).to_string(index=False))

    allf = h2o.H2OFrame(c[PRED])
    for k in ("Geography", "Gender", "Card Type"):
        allf[k] = allf[k].asfactor()
    c["clv_pred"] = aml.leader.predict(allf).as_data_frame().values.ravel().clip(min=0)

    t = c[c.clv_split == "test"]
    rmse = lambda d: float(np.sqrt(np.mean((d.clv_proxy - d.clv_pred) ** 2)))
    base = float(np.sqrt(np.mean((t.clv_proxy - c.loc[c.clv_split == "train", "clv_proxy"].mean()) ** 2)))
    r2 = float(1 - ((t.clv_proxy - t.clv_pred) ** 2).sum() / ((t.clv_proxy - t.clv_proxy.mean()) ** 2).sum())
    seg = t.groupby("segment").apply(lambda d: pd.Series({
        "n": len(d), "mean_clv_true": d.clv_proxy.mean(), "mean_clv_pred": d.clv_pred.mean(),
        "rmse": rmse(d)})).reset_index()
    seg.to_csv(OUT / "clv_segment_table.csv", index=False)
    res = {"leader": aml.leader.model_id, "test_rmse": rmse(t), "test_r2": r2,
           "mean_baseline_rmse": base, "test_n": len(t), "annuity_factor": ANNUITY}
    (OUT / "clv_metrics.json").write_text(json.dumps(res, indent=2))
    c[["customer_id", "segment", "annual_revenue", "clv_proxy", "clv_pred", "clv_split"]] \
        .to_parquet(OUT / "clv_scores.parquet", index=False)
    print(res)
    print(seg.round(1).to_string(index=False))
    h2o.cluster().shutdown(prompt=False)


if __name__ == "__main__":
    main()
