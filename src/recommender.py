"""M5 - Lightweight next-best-product recommender (RG4): item-item CF + association rules,
with a demographic-segment fallback for cold-start customers.

DATA CAVEAT (also in provenance.json): the churn dataset has only NumOfProducts (a count), not which
products are held. We therefore GENERATE a 7-product holdings matrix consistent with each customer's real
NumOfProducts/age/balance/salary, with built-in co-purchase structure. Precision@k below measures how well
each method recovers that structure -- it is NOT evidence about real banking behaviour.

Evaluation: 80/20 customer split; for each test customer one held product is hidden and the model must
rank it among non-held products (hit-rate@k == precision@k with one hidden item, divided by k).
Compared against a global popularity baseline; cold-start (1-product customers -> 0 visible) reported separately.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

ROOT = Path(__file__).resolve().parents[1]
PROC, OUT = ROOT / "data" / "processed", ROOT / "outputs"
SEED = 42
PRODUCTS = ["savings", "fixed_deposit", "personal_loan", "home_loan", "insurance", "investment", "overdraft"]
P = {p: i for i, p in enumerate(PRODUCTS)}
COPURCHASE = {"home_loan": {"insurance": 2.5}, "investment": {"fixed_deposit": 1.6, "insurance": 1.4},
              "personal_loan": {"overdraft": 1.8}, "fixed_deposit": {"savings": 1.5}}


def segment_key(c):
    age = pd.cut(c.Age, [0, 35, 55, 200], labels=["young", "mid", "senior"]).astype(str)
    wealth = np.where((c.Balance > 100_000) | (c.EstimatedSalary > 150_000), "rich", "std")
    return (age + "|" + wealth).values


def generate_holdings(c, rng):
    mult = {"young": {"personal_loan": 2, "overdraft": 2.5, "savings": 1.5, "investment": .5, "fixed_deposit": .4},
            "mid": {"home_loan": 2.5, "insurance": 2, "personal_loan": 1.2},
            "senior": {"fixed_deposit": 3, "investment": 2, "insurance": 1.5, "overdraft": .3}}
    H = np.zeros((len(c), len(PRODUCTS)), dtype=int)
    ages = pd.cut(c.Age, [0, 35, 55, 200], labels=["young", "mid", "senior"]).astype(str).values
    rich = ((c.Balance > 100_000) | (c.EstimatedSalary > 150_000)).values
    for i in range(len(c)):
        w = np.ones(len(PRODUCTS))
        for p, m in mult[ages[i]].items():
            w[P[p]] *= m
        if rich[i]:
            w[P["investment"]] *= 2.5; w[P["fixed_deposit"]] *= 1.5
        if c.HasCrCard.iat[i]:
            w[P["overdraft"]] *= .5
        for _ in range(int(c.NumOfProducts.iat[i])):
            j = rng.choice(len(PRODUCTS), p=w / w.sum())
            H[i, j] = 1
            w[j] = 0
            for q, m in COPURCHASE.get(PRODUCTS[j], {}).items():
                w[P[q]] *= m
    return H


class Recommenders:
    def __init__(self, H, seg):
        self.pop = H.mean(0)
        n = np.linalg.norm(H, axis=0) + 1e-9
        self.sim = (H.T @ H) / np.outer(n, n)
        np.fill_diagonal(self.sim, 0)
        self.seg_pop = {s: H[seg == s].mean(0) for s in np.unique(seg)}
        df = apriori(pd.DataFrame(H.astype(bool), columns=PRODUCTS), min_support=0.01, use_colnames=True)
        r = association_rules(df, metric="confidence", min_threshold=0.05)
        self.rules = [(set(a), list(b)[0], conf) for a, b, conf in
                      zip(r.antecedents, r.consequents, r.confidence) if len(a) <= 2 and len(b) == 1]

    def popularity(self, x, s): return self.pop.copy()
    def segment_pop(self, x, s): return self.seg_pop.get(s, self.pop).copy()
    def cf(self, x, s): return (x @ self.sim) + 1e-6 * self.pop  # pop breaks ties / cold start

    def rules_score(self, x, s):
        held = {PRODUCTS[i] for i in np.where(x)[0]}
        sc = 1e-6 * self.pop.copy()
        for a, b, conf in self.rules:
            if a <= held:
                sc[P[b]] = max(sc[P[b]], conf)
        return sc

    def hybrid(self, x, s):
        """Fixed 50/50 blend of CF and segment popularity (weight not tuned on test data).
        With no history CF is uninformative, so it falls back to segment popularity alone (cold-start fix)."""
        sp = self.segment_pop(x, s)
        if x.sum() == 0:
            return sp
        cf = self.cf(x, s)
        return 0.5 * cf / cf.sum() + 0.5 * sp / sp.sum()


def evaluate(rec, H, seg, test_idx, rng, reps=5, ks=(1, 3)):
    methods = ["popularity", "segment_pop", "cf", "rules_score", "hybrid"]
    out = {}
    for group, cond in (("warm", lambda n: n >= 2), ("cold", lambda n: n == 1)):
        idx = [i for i in test_idx if cond(H[i].sum())]
        stats = {m: {k: [] for k in ks} for m in methods}
        for _ in range(reps):
            hide = {i: rng.choice(np.where(H[i])[0]) for i in idx}
            for m in methods:
                hit = {k: 0 for k in ks}
                for i in idx:
                    x = H[i].copy(); x[hide[i]] = 0
                    sc = getattr(rec, m)(x, seg[i]).astype(float)
                    sc[x == 1] = -np.inf
                    order = np.argsort(-sc)
                    for k in ks:
                        hit[k] += int(hide[i] in order[:k])
                for k in ks:
                    stats[m][k].append(hit[k] / len(idx))
        out[group] = {"n": len(idx), **{m: {f"precision@{k}": float(np.mean(stats[m][k]) / k)
                                            for k in ks} | {f"hit_rate@{k}": float(np.mean(stats[m][k])) for k in ks}
                                        for m in methods}}
        for m in methods:
            for k in ks:
                base = out[group]["popularity"][f"hit_rate@{k}"]
                out[group][m][f"lift_vs_popularity@{k}"] = float(np.mean(stats[m][k]) / base) if base else None
    return out


def main():
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    c = pd.read_parquet(PROC / "customers.parquet")
    H = generate_holdings(c, rng)
    hold = pd.DataFrame(H, columns=PRODUCTS); hold.insert(0, "customer_id", c.customer_id.values)
    hold.to_parquet(PROC / "product_holdings.parquet", index=False)
    assert (H.sum(1) == c.NumOfProducts.values).all()
    seg = segment_key(c)
    perm = rng.permutation(len(c)); cut = int(0.8 * len(c))
    tr, te = perm[:cut], perm[cut:]
    rec = Recommenders(H[tr], seg[tr])
    res = evaluate(rec, H, seg, te, np.random.default_rng(SEED))
    res["n_rules"] = len(rec.rules)
    res["holdings_prevalence"] = dict(zip(PRODUCTS, map(float, H.mean(0))))
    (OUT / "recommender_metrics.json").write_text(json.dumps(res, indent=2))

    # final next-best-product for every customer: hybrid fit on all customers, current holdings
    full = Recommenders(H, seg)
    rows = []
    for i in range(len(c)):
        sc = full.hybrid(H[i], seg[i]).astype(float); sc[H[i] == 1] = -np.inf
        o = np.argsort(-sc)
        rows.append((c.customer_id.iat[i], PRODUCTS[o[0]], float(sc[o[0]]), PRODUCTS[o[1]]))
    pd.DataFrame(rows, columns=["customer_id", "rec_product", "rec_score", "rec_product_2"]) \
        .to_parquet(OUT / "recommendations.parquet", index=False)

    prov = json.loads((PROC / "provenance.json").read_text())
    prov["product_holdings"] = {
        "status": "SYNTHETIC (generated by src/recommender.py, seed 42)",
        "reason": "source data has NumOfProducts count only, no product identities",
        "constraints": "row sum equals real NumOfProducts; propensities depend on age band, wealth, HasCrCard; "
                       "co-purchase boosts: " + json.dumps(COPURCHASE),
        "caveat": "recommender metrics measure recovery of this generated structure, not real behaviour"}
    (PROC / "provenance.json").write_text(json.dumps(prov, indent=2))
    for g in ("warm", "cold"):
        print(g, res[g]["n"], {m: {k: round(v, 3) for k, v in res[g][m].items() if "@" in k and "lift" not in k
                                   and "hit" not in k} for m in ("popularity", "segment_pop", "cf", "rules_score", "hybrid")})
        print("  lift@1", {m: round(res[g][m]["lift_vs_popularity@1"], 2) for m in ("segment_pop", "cf", "rules_score", "hybrid")})


if __name__ == "__main__":
    main()
