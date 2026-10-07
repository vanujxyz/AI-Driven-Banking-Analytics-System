from pathlib import Path

import pandas as pd

P = Path(__file__).resolve().parents[1] / "data" / "processed"


def test_customers():
    c = pd.read_parquet(P / "customers.parquet")
    assert len(c) == 10_000 and c.customer_id.is_unique
    assert c.isna().sum().sum() == 0
    assert {"Exited", "Tenure", "NumOfProducts", "home_lat"} <= set(c.columns)


def test_transactions():
    c = pd.read_parquet(P / "customers.parquet")
    t = pd.read_parquet(P / "transactions.parquet")
    assert t.customer_id.isin(c.customer_id).all()
    assert t.isFraud.sum() == 8213
    assert t.step.is_monotonic_increasing
    assert 0.005 < t.isFraud.mean() < 0.02


def test_provenance():
    assert (P / "provenance.json").exists()
