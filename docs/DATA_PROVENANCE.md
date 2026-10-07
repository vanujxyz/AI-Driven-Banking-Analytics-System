# Data Provenance (closes RG6)

Machine-readable version: `data/processed/provenance.json`. Seed 42 everywhere.

## Sources

| Dataset | Source | Used for | Nature |
|---|---|---|---|
| Bank customer churn (`Customer-Churn-Records.csv`, 10,000 rows) | Kaggle `radheshyamkollipara/bank-customer-churn` | Customers: churn, CLV, survival, recommendations | Public, widely used, itself believed to be synthetic/blended |
| PaySim (`PS_..._log.csv`, 6.36M rows) | Kaggle `ealaxi/paysim1` | Transactions: fraud | Simulator output derived from a real mobile-money log |

## What is real, blended or synthetic

| Field(s) | Status | How |
|---|---|---|
| All original churn columns (except `RowNumber`, `Surname`, dropped) | As published | - |
| `home_city`, `home_lat`, `home_lon` | **Synthetic** | Random city within the customer's `Geography` |
| Transactions sample | **Down-sampled** | All 8,213 fraud rows + 800,000 random legit rows (fraud rate 1.02% vs 0.13% original) |
| `customer_id` on transactions | **Synthetic** | Seeded random assignment to the 10,000 customers, independent of the fraud label |
| `txn_lat`, `txn_lon` | **Synthetic** | 85% near home city, 15% random city; independent of fraud label |
| `nameOrig`, `isFlaggedFraud` | Dropped | - |
| Product holdings (7 products) | **Synthetic** | Generated in `src/recommender.py`; row sums equal real `NumOfProducts`; propensities depend on age band, wealth, `HasCrCard`; co-purchase boosts (home loan -> insurance etc.) |
| CLV proxy | **Assumption** | 5-yr, 8% discounted `0.02*Balance + 80*Products + 40*HasCrCard + 0.002*Salary*Active` |
| Fraud review cost | **Assumption** | 25 per alert reviewed |

## Consequences you must state when citing results

1. **Cross-module links are not real.** Fraud and churn come from unrelated populations; a customer's fraud flag
   says nothing real about their churn. The integration demonstrates the *architecture*, not a discovered relationship.
2. **Geo-distance has no fraud signal by construction.** Rule features (velocity, time gap, distance) are label-independent,
   so the rules-only baseline scores at/below chance (PR-AUC 0.006) and the no-rule ablation matches the full model.
3. **Recommender precision@k measures recovery of the generated co-purchase structure**, not real behaviour.
4. **`Complain` leaks the churn label** (corr(Complain, Exited) = 0.996) and is excluded from the churn model.
   Papers that report AUC > 0.95 on this dataset likely include it.
5. **PaySim fraud is close to deterministic** given the balance-error features, which is why PR-AUC is ~1.
   High scores here do not imply real-world performance.
6. `clv_pred` for the 80% training customers is in-sample; the reported CLV metrics use only the 20% holdout.
