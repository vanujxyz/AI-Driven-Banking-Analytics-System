# AI-Driven Banking Analytics System

A bank usually runs fraud, churn, customer-value and product-recommendation analytics as separate projects, each with its own accuracy number and none of them talking to the others. This project builds all five on one shared customer table and ends with a ranked list of who the bank should deal with first.

It is an undergraduate research project (VIT Vellore) built on public data for a fictional bank. It is an offline pipeline, not a production system.

**Authors:** Vanuj Gangrade, Aryan Bhandari  |  **Guide:** Dr. Gopinath M

## What it does

| Module | Question | Method |
|---|---|---|
| Churn | Who is likely to leave? | XGBoost, compared with logistic regression |
| Fraud | Which payments look fraudulent? | Rule features + Keras network, random forest as a baseline |
| Customer value | What is a customer worth over 5 years? | Engineered CLV proxy, H2O AutoML |
| Survival | How soon will they leave? | Kaplan-Meier, log-rank test, Cox and Weibull models |
| Recommendations | What should we offer next? | Item-based collaborative filtering, association rules, segment fallback |

`src/integrate.py` joins the outputs on `customer_id`, multiplies each customer's chance of leaving by their survival-adjusted value to get a "value at risk", and assigns one of four actions: fraud review, personal retention call, retention offer, or cross-sell. Everyone is ranked.

Each module is scored on a measure an operations team would care about rather than raw accuracy: PR-AUC and cost at a fixed alert budget for fraud, top-decile lift for churn, and value captured at a fixed retention-contact budget for the combined list.

## Results

| Module | Result |
|---|---|
| Churn | ROC-AUC 0.864 (5-fold out-of-fold), top-decile lift 4.0x. The top 10% of scores holds 40% of churners. |
| Fraud | PR-AUC 0.997. At a 1% alert budget: precision 1.00, recall 0.98. |
| Customer value | Holdout RMSE 4,549 vs 4,940 for a mean baseline. R² is only 0.15 for individuals, but group averages are close. |
| Survival | Groups differ strongly (log-rank p < 1e-80). Cox concordance 0.77 under 5-fold CV. |
| Recommendations | Precision@1 is 1.30x a popularity baseline for existing customers, 1.51x for new customers with no history. |
| Combined list | With a 10% contact budget, ranking by value at risk reaches 38% of the value lost to churn. Ranking by churn probability alone reaches 34%, random contact about 10%. |

Full tables, per-module notes and limitations are in [`docs/RESULTS.md`](docs/RESULTS.md).

## Read this before trusting the numbers

The data is public and partly synthetic, and several parts of the pipeline are generated. [`docs/DATA_PROVENANCE.md`](docs/DATA_PROVENANCE.md) lists every field. The points that matter most:

- **Fraud and churn data come from different, unrelated sources.** They are joined by a seeded random assignment. A customer's fraud flag has no real relationship to their churn risk. The combined list demonstrates the architecture; it does not discover anything about customers.
- **PaySim fraud is easy.** The near-perfect fraud scores come from the simulator's balance-error pattern and say little about real-world performance. The velocity, time-gap and distance rules score at chance because locations and times were generated independently of the label.
- **Product holdings are generated.** The churn dataset only records how many products a customer has. The recommender is evaluated on holdings produced by this project, so its scores show how the method behaves, not how well it would sell.
- **The `Complain` column in the churn dataset leaks the label** (correlation 0.996 with `Exited`). It is excluded. Results on that dataset that look much better than 0.86 AUC probably include it.
- **Customer value is a proxy** built from a formula chosen in this project, not accounting data.

## Quick start

Requires Anaconda (or any Python 3.10 environment) and a JDK for H2O. Node.js is only needed for the dashboard.

```bash
conda create -n bankml python=3.10 openjdk=17 -c conda-forge -y
conda activate bankml
pip install -r requirements.txt "numpy<1.24"

# datasets are downloaded from Kaggle; create a token in your Kaggle account settings
export KAGGLE_API_TOKEN=your_token        # PowerShell: $env:KAGGLE_API_TOKEN="your_token"

python run_pipeline.py                    # about 3 minutes, runs all steps, then the tests
```

The pipeline downloads PaySim (about 180 MB compressed) and the bank churn dataset, builds everything under `data/processed/` and `outputs/`, and finishes by running the 17 tests. The main output is `outputs/customer_action_list.csv`.

Individual steps can be run on their own, for example `python src/churn.py`. The raw and processed data are not committed, so run the pipeline once before running the tests.

The Kaggle download uses the REST API directly, so the `kaggle` CLI package is not needed. The token is read from the environment and never written to disk.

## Dashboard

A React dashboard presents the results in plain language for non-technical readers: what each chart means, an interactive retention-budget slider, and a searchable customer list that explains each customer in a sentence. It uses the [IBM Carbon Design System](https://carbondesignsystem.com).

```bash
cd dashboard
npm install
npm run dev          # http://localhost:5173
npm run build        # static site in dashboard/dist
```

`dashboard/public/data.json` is committed so the dashboard runs without the pipeline. To refresh it after re-running the pipeline: `python src/export_dashboard.py`.

## Repository layout

```
run_pipeline.py          runs every step, then the tests
src/
  download_data.py       Kaggle download
  data_blend.py          blend PaySim and churn data, write provenance
  churn.py  fraud.py  clv.py  survival.py  recommender.py
  integrate.py           shared customer view, action rules, retention KPI
  export_dashboard.py    results to JSON for the dashboard
tests/                   one test file per stage (17 tests)
outputs/                 metrics (JSON), tables (CSV), survival curves (PNG), final action list
dashboard/               React + IBM Carbon app
docs/
  RESULTS.md             results, per-module notes, limitations
  DATA_PROVENANCE.md     what is real, blended or generated
```

## Reproducibility notes

All random seeds are fixed (42). Re-running can still shift a count by one or two (for example 624 vs 625 high-value retention calls), because H2O AutoML runs under a time limit and is not bit-for-bit deterministic. The metrics in `outputs/` are from one full run.

## Data sources

- [Bank Customer Churn](https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn), Kaggle
- [PaySim synthetic financial transactions](https://www.kaggle.com/datasets/ealaxi/paysim1), Kaggle

Check each dataset's licence on Kaggle before reusing the data.
