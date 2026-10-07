"""M3 - Fraud detection (RG1): rule features + Keras classifier, scored at a fixed alert budget.

Split is stratified random 70/10/20 (a time split was rejected: PaySim's legit volume is front-loaded,
which makes the late-step test window ~8% fraud vs ~1% overall). Velocity/gap features use past rows only.
Business metrics: PR-AUC, precision/recall and total cost at fixed alert budgets
(cost = missed-fraud amount + REVIEW_COST per alert reviewed).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PROC, OUT = ROOT / "data" / "processed", ROOT / "outputs"
SEED = 42
BUDGETS = [0.005, 0.01, 0.02]  # share of test transactions analysts can review
REVIEW_COST = 25.0  # cost of one analyst alert review (assumption, documented)
TYPES = ["CASH_OUT", "TRANSFER", "PAYMENT", "CASH_IN", "DEBIT"]


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def build_features(t, customers):
    t = t.merge(customers[["customer_id", "home_lat", "home_lon"]], on="customer_id", how="left")
    t = t.sort_values(["customer_id", "step"], kind="stable").reset_index(drop=True)
    cust = t.customer_id.astype("category").cat.codes.values.astype(np.int64)
    step = t.step.values.astype(np.int64)
    keys = cust * 10_000 + step
    pos = np.arange(len(t))
    # rule 1: velocity = number of the customer's earlier txns in the previous 24 steps
    left = np.searchsorted(keys, cust * 10_000 + np.maximum(step - 24, 0), side="left")
    t["velocity_24h"] = pos - left
    # rule 2: time gap since the customer's previous txn (999 = no history)
    prev = t.groupby("customer_id").step.shift(1)
    t["time_gap"] = (t.step - prev).fillna(999)
    # rule 3: distance from the customer's home city
    t["geo_km"] = haversine_km(t.txn_lat, t.txn_lon, t.home_lat, t.home_lon)
    # PaySim balance-consistency features
    t["err_orig"] = t.newbalanceOrig + t.amount - t.oldbalanceOrg
    t["err_dest"] = t.oldbalanceDest + t.amount - t.newbalanceDest
    t["drains_account"] = ((t.amount >= t.oldbalanceOrg) & (t.oldbalanceOrg > 0)).astype(int)
    t["dest_merchant"] = t.dest_id.str.startswith("M").astype(int)
    for ty in TYPES:
        t[f"type_{ty}"] = (t.type == ty).astype(int)
    return t.sort_values("step", kind="stable").reset_index(drop=True)


RULE_COLS = ["velocity_24h", "time_gap", "geo_km"]
BASE_COLS = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
             "err_orig", "err_dest", "drains_account", "dest_merchant"] + [f"type_{x}" for x in TYPES]
FEATS = BASE_COLS + RULE_COLS


def transform(df):
    x = df[FEATS].copy()
    for c in ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]:
        x[c] = np.log1p(x[c])
    for c in ["err_orig", "err_dest"]:
        x[c] = np.sign(x[c]) * np.log1p(np.abs(x[c]))
    x["geo_km"] = np.log1p(x.geo_km)
    x["time_gap"] = np.log1p(x.time_gap)
    return x.values.astype("float32")


def budget_metrics(y, amount, score, budget):
    k = max(1, int(round(budget * len(y))))
    idx = np.argsort(-score)[:k]
    caught = y[idx].sum()
    missed_amt = amount[(y == 1)].sum() - amount[idx][y[idx] == 1].sum()
    fp = k - caught
    return {"alerts": k, "precision": float(caught / k), "recall": float(caught / y.sum()),
            "false_positives": int(fp), "missed_fraud_amount": float(missed_amt),
            "total_cost": float(missed_amt + REVIEW_COST * k)}


def evaluate(y, amount, score):
    out = {"pr_auc": float(average_precision_score(y, score))}
    for b in BUDGETS:
        out[f"budget_{b}"] = budget_metrics(y, amount, score, b)
    return out


def keras_model(n_in):
    import tensorflow as tf
    tf.random.set_seed(SEED)
    m = tf.keras.Sequential([
        tf.keras.layers.Input((n_in,)),
        tf.keras.layers.Dense(64, activation="relu"), tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")])
    m.compile("adam", "binary_crossentropy", metrics=[tf.keras.metrics.AUC(curve="PR", name="pr_auc")])
    return m


def main():
    OUT.mkdir(exist_ok=True)
    np.random.seed(SEED)
    customers = pd.read_parquet(PROC / "customers.parquet")
    t = build_features(pd.read_parquet(PROC / "transactions.parquet"), customers)
    rest, te = train_test_split(t, test_size=0.2, stratify=t.isFraud, random_state=SEED)
    tr, va = train_test_split(rest, test_size=0.125, stratify=rest.isFraud, random_state=SEED)
    print("split sizes / fraud:", [(len(d), int(d.isFraud.sum())) for d in (tr, va, te)])
    Xtr, Xva, Xte = transform(tr), transform(va), transform(te)
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xva_s, Xte_s = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)
    ytr, yva, yte = tr.isFraud.values, va.isFraud.values, te.isFraud.values
    amt = te.amount.values
    results = {}

    # 1) rules-only baseline: z-scored rule features combined (higher = more suspicious)
    rule = (te.velocity_24h.values / 10 + (te.time_gap.values < 1) + (te.geo_km.values > 500))
    results["rules_only"] = evaluate(yte, amt, rule + np.random.default_rng(SEED).random(len(te)) * 1e-6)

    # 2) Random Forest baseline (literature reference, P02)
    rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=3, n_jobs=-1, random_state=SEED,
                                class_weight="balanced_subsample").fit(Xtr, ytr)
    rf_p = rf.predict_proba(Xte)[:, 1]
    results["random_forest"] = evaluate(yte, amt, rf_p)

    # 3) Keras classifier on balance + rule features
    import tensorflow as tf
    m = keras_model(Xtr_s.shape[1])
    cw = {0: 1.0, 1: float((ytr == 0).sum() / max(1, (ytr == 1).sum())) ** 0.5}
    m.fit(Xtr_s, ytr, validation_data=(Xva_s, yva), epochs=15, batch_size=2048, class_weight=cw, verbose=2,
          callbacks=[tf.keras.callbacks.EarlyStopping("val_pr_auc", mode="max", patience=3,
                                                      restore_best_weights=True)])
    nn_p = m.predict(Xte_s, batch_size=8192, verbose=0).ravel()
    results["keras_mlp"] = evaluate(yte, amt, nn_p)

    # ablation: Keras without rule features (shows what the rule features add)
    k = len(BASE_COLS)
    m2 = keras_model(k)
    m2.fit(Xtr_s[:, :k], ytr, validation_data=(Xva_s[:, :k], yva), epochs=15, batch_size=2048,
           class_weight=cw, verbose=0,
           callbacks=[tf.keras.callbacks.EarlyStopping("val_pr_auc", mode="max", patience=3,
                                                       restore_best_weights=True)])
    results["keras_mlp_no_rule_features"] = evaluate(yte, amt, m2.predict(Xte_s[:, :k], verbose=0).ravel())

    results["test_fraud_rate"] = float(yte.mean())
    results["review_cost_assumption"] = REVIEW_COST
    (OUT / "fraud_metrics.json").write_text(json.dumps(results, indent=2))

    # customer-level flags from the test window, using the Keras score and the 1% budget
    te = te.assign(fraud_score=nn_p)
    cut = np.sort(nn_p)[::-1][int(0.01 * len(te)) - 1]
    cs = te.groupby("customer_id").agg(fraud_score_max=("fraud_score", "max"),
                                       fraud_alerts=("fraud_score", lambda s: int((s >= cut).sum())),
                                       test_txns=("fraud_score", "size")).reset_index()
    cs["fraud_flagged"] = (cs.fraud_alerts > 0).astype(int)
    allc = customers[["customer_id"]].merge(cs, on="customer_id", how="left")
    allc[["fraud_score_max", "fraud_alerts", "test_txns", "fraud_flagged"]] = \
        allc[["fraud_score_max", "fraud_alerts", "test_txns", "fraud_flagged"]].fillna(0)
    allc.to_parquet(OUT / "fraud_customer_flags.parquet", index=False)
    for name, r in results.items():
        if isinstance(r, dict):
            print(name, "PR-AUC=%.4f" % r["pr_auc"], "P@1%%=%.3f R@1%%=%.3f" % (
                r["budget_0.01"]["precision"], r["budget_0.01"]["recall"]))


if __name__ == "__main__":
    main()
