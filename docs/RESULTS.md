# Results, Module Memos and Limitations

All numbers come from `outputs/*.json|csv` produced by `python run_pipeline.py`.
Objective O1: unify fraud, churn, CLV, survival and recommendations on one customer-ID-keyed layer, scored by business KPIs.

## Success metrics vs. design document

| Metric in design doc | Target | Achieved | Met? |
|---|---|---|---|
| Fraud PR-AUC at fixed alert budget | report | 0.997 (Keras), 0.998 (RF); 1% budget = 1,616 alerts: precision 0.998, recall 0.982 | Yes (see caveat 5) |
| Churn ROC-AUC | >= ~0.86 | 0.8635 (XGBoost OOF, fold std 0.008) | Yes, narrowly |
| Churn top-decile lift | report | 4.04x, captures 40% of churners in top 10% | Yes |
| CLV RMSE by segment | report | `outputs/clv_segment_table.csv` | Yes |
| Survival median + log-rank | report | log-rank p = 1.6e-87 (6 segments) | Yes |
| Recommender precision@k, lift vs popularity | > baseline | warm 1.30x, cold 1.51x at k=1 | Yes (synthetic data) |
| Baseline: RF ~95.8% fraud acc (P02); churn AUC 0.86-0.87 (P16) | match/exceed | RF PR-AUC 0.998; AUC 0.864 | Matched |

## M1 Data (W1.2)
10,000 customers, 808,213 transactions (1.02% fraud), provenance in `docs/DATA_PROVENANCE.md`.

## M2 Churn (W2.2, RG2)
| Model | ROC-AUC (5-fold OOF) | Top-decile lift |
|---|---|---|
| Logistic regression | 0.764 | 2.72 |
| XGBoost | **0.864** | **4.04** |

Decile table: `outputs/churn_decile_lift.csv` (decile 1 churn rate 82% vs 20% base). Out-of-fold scores feed the integration layer, so no customer is scored by a model that saw their label.

## M3 Fraud (W2.1, RG1)
Stratified 70/10/20 split (a time split was rejected: PaySim legit volume is front-loaded, giving a ~8% fraud test window).
Cost = missed-fraud amount + 25 per alert.

| Model | PR-AUC | P / R @ 0.5% budget | P / R @ 1% | P / R @ 2% |
|---|---|---|---|---|
| Rules only (velocity, time gap, geo) | 0.006 | 0 / 0 | 0 / 0 | 0.001 / 0.001 |
| Random forest | 0.998 | 1.00 / 0.49 | 1.00 / 0.98 | 0.51 / 1.00 |
| Keras MLP (final) | 0.997 | 1.00 / 0.49 | 1.00 / 0.98 | 0.51 / 1.00 |
| Keras, no rule features | 0.998 | 1.00 / 0.49 | 1.00 / 0.98 | 0.51 / 1.00 |

Reading it as an operations team would: with a budget below the fraud count (0.5%) recall is capped at 49% regardless of model;
at 1% (about the fraud count) nearly all fraud is caught with almost no false positives; at 2% half the alerts are wasted.
That is the RG1 point: the *budget*, not the accuracy, determines cost. Total cost at 1% is ~26M (RF) / ~59M (Keras) vs ~1,010M-1,115M at 0.5%.

## M4a CLV (W3.1)
H2O AutoML (GBM leader, 12 models, 4 min cap), predictors exclude the CLV formula inputs.
Holdout RMSE 4,549 vs 4,940 mean-baseline, R2 = 0.15 -> **individual CLV is only weakly predictable from demographics; segment-level CLV is reliable.**

| Segment | true mean CLV | predicted |
|---|---|---|
| Germany-active | 10,924 | 10,546 |
| Germany-inactive | 10,096 | 9,933 |
| France-active | 6,481 | 6,253 |
| Spain-active | 6,315 | 6,455 |
| France-inactive | 5,869 | 5,797 |
| Spain-inactive | 5,197 | 5,759 |

## M4b Survival (W3.2)
Kaplan-Meier by segment, log-rank chi2 = 415 (df 5), p = 1.6e-87. Cox concordance 0.770 (5-fold CV, std 0.011); Weibull AFT concordance 0.770.
Hazard ratios (`outputs/survival_cox_hr.csv`): age +10y 1.48, Germany 1.75, active member 0.54, 2 products 0.32, 3+ products 2.55.
"Median survival 10.5" in the segment table means the KM curve is still above 50% at the last observed tenure (10 yrs): not reached.

**CLV + survival together (closes RG3).** `sa_clv` = CLV discounted by each customer's conditional 5-year survival (Weibull):

| Segment | CLV | 5-yr survival | Expected years (of 5) | Survival-adjusted CLV |
|---|---|---|---|---|
| Germany-active | 10,583 | 0.72 | 4.15 | 8,864 |
| Germany-inactive | 9,949 | 0.54 | 3.53 | 7,148 |
| France-active | 6,240 | 0.82 | 4.47 | 5,619 |
| Spain-active | 6,456 | 0.81 | 4.43 | 5,766 |
| France-inactive | 5,790 | 0.72 | 4.16 | 4,863 |
| Spain-inactive | 5,698 | 0.71 | 4.12 | 4,742 |

Germany-inactive loses 28% of its CLV to expected churn; France-active only 10%. This is the "how much and how soon" read-out the literature (P08-P10) lacked on banking data.

## M5 Recommender (W4.1, RG4)
7 products, synthetic holdings (see provenance). 80/20 customer split, one held product hidden per test customer, 5 repetitions.

| Method | Warm (>=2 products, n=975) p@1 / lift | Cold-start (1 product -> 0 visible, n=1025) p@1 / lift |
|---|---|---|
| Popularity baseline | 0.247 / 1.00 | 0.163 / 1.00 |
| Item-item CF | 0.271 / 1.10 | 0.163 / 1.00 (no history) |
| Association rules (42 rules) | 0.270 / 1.09 | 0.163 / 1.00 |
| Segment popularity | 0.310 / 1.26 | 0.246 / 1.51 |
| **Hybrid (50/50 CF + segment, fallback to segment)** | **0.322 / 1.30** | **0.246 / 1.51** |

The simple cold-start mitigation (demographic-segment popularity) lifts precision 51% over popularity where plain CF is blind - the RG4 gap.
The 50/50 blend weight was fixed a priori, not tuned on test data. CF alone adds little because the generated holdings are mostly driven by demographics.

## M6 Integration (W4.2, RG5)
`outputs/customer_action_list.csv`: one row per customer with fraud flag, churn probability/decile, CLV, survival-adjusted CLV,
5-yr survival, next-best product, value at risk (= churn prob x sa_clv), action and priority rank.
Actions: 1,500 FRAUD_REVIEW, 625 RETAIN_HIGH_VALUE, 1,097 RETAIN_STANDARD, 6,778 CROSS_SELL.

Retention KPI at a 10% contact budget (1,000 customers), `outputs/integration_kpi.json`:

| Ranking | Churners caught | Share of churn value (sa_clv) captured |
|---|---|---|
| Value at risk (integrated) | 727 | **38.1%** |
| Churn probability only | 823 | 34.4% |
| Random | 197 | 9.5% |

The integrated ranking catches fewer churners but more value (+11% relative) - the benefit of combining modules, supporting the working hypothesis.

## Limitations (for the write-up)
- Offline pipeline on public/synthetic data only; out of scope per design: real-time deployment, compliance, generalisation beyond the blended dataset.
- Cross-module relationships are artefacts of joining unrelated sources (provenance caveat 1); the 15% of customers flagged for fraud review is a by-product of random assignment.
- Fraud scores are near-perfect because PaySim is easy; geo/velocity rules contribute nothing by construction.
- Recommender evaluated on generated holdings; results show method behaviour, not market performance.
- CLV is a proxy under stated assumptions; individual-level R2 is low.
- Churn uses fixed XGBoost hyper-parameters (no tuning), single 5-fold CV.
- The `value_at_risk` retention KPI uses the same `sa_clv` for ranking and for scoring value; it measures prioritisation quality, not realised financial benefit.
- H2O AutoML runtime cap (240s) makes the leaderboard slightly non-deterministic across machines.
- No Streamlit demo; a React dashboard (IBM Carbon Design System) is provided in `dashboard/` instead.
- Re-running the pipeline can shift a few borderline counts by 1 (e.g. 624 vs 625 high-value retention calls) because H2O AutoML is not bit-for-bit deterministic.
