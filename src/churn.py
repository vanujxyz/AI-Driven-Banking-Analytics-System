"""M2 - Churn prediction (RG2): logistic regression vs gradient boosting.

Scores are 5-fold out-of-fold, so every customer gets an unbiased churn probability
that downstream modules can use. Reports ROC-AUC and top-decile lift.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
PROC, OUT = ROOT / "data" / "processed", ROOT / "outputs"
SEED = 42
CAT = ["Geography", "Gender", "Card Type"]
# Complain is excluded: corr(Complain, Exited)=0.996 -> target leakage (see docs/DATA_PROVENANCE.md)
NUM = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard",
       "IsActiveMember", "EstimatedSalary", "Satisfaction Score", "Point Earned"]


def decile_lift(y, p):
    df = pd.DataFrame({"y": y, "p": p}).sort_values("p", ascending=False).reset_index(drop=True)
    df["decile"] = (np.arange(len(df)) * 10 // len(df)) + 1
    t = df.groupby("decile").agg(customers=("y", "size"), churners=("y", "sum"), rate=("y", "mean"))
    t["lift"] = t.rate / df.y.mean()
    t["cum_capture"] = t.churners.cumsum() / df.y.sum()
    return t.reset_index()


def models():
    pre = lambda: ColumnTransformer([("c", OneHotEncoder(drop="first"), CAT),
                                     ("n", StandardScaler(), NUM)])
    return {
        "logistic_regression": make_pipeline(pre(), LogisticRegression(max_iter=1000)),
        "xgboost": make_pipeline(
            ColumnTransformer([("c", OneHotEncoder(drop="first"), CAT)], remainder="passthrough"),
            XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                          colsample_bytree=0.8, eval_metric="logloss", random_state=SEED, n_jobs=4)),
    }


def main():
    OUT.mkdir(exist_ok=True)
    c = pd.read_parquet(PROC / "customers.parquet")
    X, y = c[CAT + NUM], c.Exited.values
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    res, oof = {}, {}
    for name, m in models().items():
        p = cross_val_predict(m, X, y, cv=cv, method="predict_proba")[:, 1]
        fold_auc = [roc_auc_score(y[te], p[te]) for _, te in cv.split(X, y)]
        lift = decile_lift(y, p)
        res[name] = {"roc_auc_oof": float(roc_auc_score(y, p)),
                     "roc_auc_fold_mean": float(np.mean(fold_auc)),
                     "roc_auc_fold_std": float(np.std(fold_auc)),
                     "top_decile_lift": float(lift.lift.iloc[0]),
                     "top_decile_capture": float(lift.cum_capture.iloc[0])}
        oof[name] = (p, lift)
        print(name, res[name])
    best = max(res, key=lambda k: res[k]["roc_auc_oof"])
    res["best_model"] = best
    p, lift = oof[best]
    lift.to_csv(OUT / "churn_decile_lift.csv", index=False)
    score = pd.DataFrame({"customer_id": c.customer_id, "churn_prob": p})
    score["churn_decile"] = pd.qcut(score.churn_prob.rank(method="first", ascending=False), 10,
                                    labels=range(1, 11)).astype(int)
    score.to_parquet(OUT / "churn_scores.parquet", index=False)
    (OUT / "churn_metrics.json").write_text(json.dumps(res, indent=2))
    print(lift.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
