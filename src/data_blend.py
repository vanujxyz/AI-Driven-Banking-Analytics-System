"""M1 - Blend PaySim transactions with the Bank Churn customer table (closes RG6).

Outputs (data/processed):
  customers.parquet     one row per customer (10,000), keyed by customer_id
  transactions.parquet  PaySim subset mapped onto those customers
  provenance.json       machine-readable record of every blended/synthetic field
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
SEED = 42
N_LEGIT = 800_000  # legit PaySim rows kept; ALL fraud rows are kept

CITIES = {  # country -> [(city, lat, lon)]
    "France": [("Paris", 48.86, 2.35), ("Lyon", 45.76, 4.84), ("Marseille", 43.30, 5.37)],
    "Spain": [("Madrid", 40.42, -3.70), ("Barcelona", 41.39, 2.17), ("Valencia", 39.47, -0.38)],
    "Germany": [("Berlin", 52.52, 13.40), ("Munich", 48.14, 11.58), ("Hamburg", 53.55, 9.99)],
}
ALL_CITIES = [c for v in CITIES.values() for c in v]


def build_customers(rng):
    c = pd.read_csv(RAW / "churn" / "Customer-Churn-Records.csv")
    c = c.rename(columns={"CustomerId": "customer_id"}).drop(columns=["RowNumber", "Surname"])
    assert c.customer_id.is_unique
    home = [CITIES[g][rng.integers(len(CITIES[g]))] for g in c.Geography]
    c["home_city"] = [h[0] for h in home]
    c["home_lat"] = [h[1] for h in home]
    c["home_lon"] = [h[2] for h in home]
    return c


def build_transactions(customers, rng):
    p = pd.read_csv(RAW / "paysim" / "PS_20174392719_1491204439457_log.csv")
    fraud = p[p.isFraud == 1]
    legit = p[p.isFraud == 0].sample(N_LEGIT, random_state=SEED)
    t = pd.concat([fraud, legit]).sort_values("step", kind="stable").reset_index(drop=True)
    t = t.drop(columns=["isFlaggedFraud"])
    n = len(t)
    # customer assignment is independent of the label (no manufactured signal)
    t["customer_id"] = customers.customer_id.values[rng.integers(len(customers), size=n)]
    # transaction location: 85% near home city, 15% random city; independent of label
    home = customers.set_index("customer_id")[["home_lat", "home_lon"]].loc[t.customer_id]
    away = rng.random(n) < 0.15
    pick = rng.integers(len(ALL_CITIES), size=n)
    lat = np.where(away, [ALL_CITIES[i][1] for i in pick], home.home_lat.values)
    lon = np.where(away, [ALL_CITIES[i][2] for i in pick], home.home_lon.values)
    t["txn_lat"] = lat + rng.normal(0, 0.05, n)
    t["txn_lon"] = lon + rng.normal(0, 0.05, n)
    return t.drop(columns=["nameOrig"]).rename(columns={"nameDest": "dest_id"})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    customers = build_customers(rng)
    txns = build_transactions(customers, rng)
    customers.to_parquet(OUT / "customers.parquet", index=False)
    txns.to_parquet(OUT / "transactions.parquet", index=False)
    prov = {
        "seed": SEED,
        "fictional_bank": "Synthetic European retail bank (France/Spain/Germany)",
        "customers": {
            "source": "Kaggle radheshyamkollipara/bank-customer-churn (Customer-Churn-Records.csv)",
            "rows": len(customers),
            "real_fields": "all original columns except RowNumber/Surname (dropped)",
            "synthetic_fields": ["home_city", "home_lat", "home_lon"],
        },
        "transactions": {
            "source": "Kaggle ealaxi/paysim1 (PaySim simulator output; itself synthetic)",
            "rows": len(txns),
            "fraud_rows": int(txns.isFraud.sum()),
            "sampling": f"all {int(txns.isFraud.sum())} fraud rows + {N_LEGIT} random legit rows",
            "synthetic_fields": ["customer_id (seeded random mapping, independent of label)",
                                 "txn_lat/txn_lon (85% home city, 15% random city, independent of label)"],
            "dropped": ["nameOrig", "isFlaggedFraud"],
        },
        "caveats": [
            "Fraud and churn come from unrelated sources; joining them by customer_id is a modelling "
            "convenience, not a real relationship. Cross-module correlations are NOT evidence of real behaviour.",
            "Geo-distance carries no fraud signal by construction because locations are label-independent.",
            "Fraud prevalence is inflated vs. PaySim (~1% vs 0.13%) by down-sampling legit rows.",
            "Complain column in churn data leaks the label and is excluded from the churn model.",
        ],
    }
    (OUT / "provenance.json").write_text(json.dumps(prov, indent=2))
    print(customers.shape, txns.shape, txns.isFraud.mean())


if __name__ == "__main__":
    main()
