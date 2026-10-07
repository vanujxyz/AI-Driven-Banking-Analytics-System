import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
O, P = ROOT / "outputs", ROOT / "data" / "processed"


def test_holdings_consistent_with_source():
    c = pd.read_parquet(P / "customers.parquet")
    h = pd.read_parquet(P / "product_holdings.parquet")
    assert (h.customer_id.values == c.customer_id.values).all()
    assert (h.drop(columns="customer_id").sum(1).values == c.NumOfProducts.values).all()


def test_recommendations_valid():
    r = pd.read_parquet(O / "recommendations.parquet")
    h = pd.read_parquet(P / "product_holdings.parquet").set_index("customer_id")
    assert len(r) == 10_000 and r.customer_id.is_unique
    held = [h.loc[cid, p] for cid, p in zip(r.customer_id, r.rec_product)]
    assert sum(held) == 0  # never recommend a product already held
    assert (r.rec_product != r.rec_product_2).all()


def test_recommender_beats_popularity():
    m = json.loads((O / "recommender_metrics.json").read_text())
    assert m["warm"]["cf"]["lift_vs_popularity@1"] > 1.0
    assert m["cold"]["hybrid"]["hit_rate@1"] >= m["cold"]["popularity"]["hit_rate@1"]
    assert "product_holdings" in json.loads((P / "provenance.json").read_text())
