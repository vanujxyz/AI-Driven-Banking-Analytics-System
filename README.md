# Artificial Intelligence for Banking

A machine learning project focused on high-value banking analytics use cases such as fraud detection, customer lifetime value prediction, churn analysis, survival analysis, and recommendation systems.

This repository brings together Python workflows, Jupyter notebooks, trained model artifacts, sample datasets, and supporting references to demonstrate how AI can be applied across the customer lifecycle in banking and credit card businesses.

## Overview

The project is organized around practical banking analytics problems:

- Fraud detection for credit card transactions
- Customer lifetime value (CLV) estimation
- Next transaction amount and timing prediction
- Churn prediction and retention analysis
- Product and merchant recommendation generation

It combines exploratory analysis with model scoring workflows, making it useful both as a learning resource and as a project portfolio piece.

## Core Modules

### Fraud Detection

The fraud workflow combines rule-based checks with a deep learning scoring model to flag potentially suspicious transactions.

Key components:

- [fraud_detection_rules_deeplearning.py](/c:/Users/vgang/Desktop/ai%20banking%20analysis/01_code/01_01_fraud_detection/fraud_detection_rules_deeplearning.py)
- [fraud_detection.ipynb](/c:/Users/vgang/Desktop/ai%20banking%20analysis/03_ipy_notebooks/fraud_detection.ipynb)
- [fraud_dl_model.json](/c:/Users/vgang/Desktop/ai%20banking%20analysis/02_models/fraud_dl_model.json)
- [fraud_dl_model.h5](/c:/Users/vgang/Desktop/ai%20banking%20analysis/02_models/fraud_dl_model.h5)

The implementation evaluates transaction behavior using conditions such as transaction geography, abnormal movement, time gaps, and unusual spending patterns, then combines those signals with a trained neural network score.

### Customer Lifetime Value (CLV)

The CLV workflow addresses both new and existing customers:

- New customer CLV estimation through clustering
- Existing customer CLV support through next transaction prediction

Key components:

- [CustomerLifetimeValue_Prediction_NewCustomer.py](/c:/Users/vgang/Desktop/ai%20banking%20analysis/01_code/01_02_clv_survival/CustomerLifetimeValue_Prediction_NewCustomer.py)
- [Customer_NextTransaction_Prediction.py](/c:/Users/vgang/Desktop/ai%20banking%20analysis/01_code/01_02_clv_survival/Customer_NextTransaction_Prediction.py)
- [clv_prediction.ipynb](/c:/Users/vgang/Desktop/ai%20banking%20analysis/03_ipy_notebooks/clv_prediction.ipynb)

Saved models:

- [KMeans_model_python_1537328280878_1](/c:/Users/vgang/Desktop/ai%20banking%20analysis/02_models/KMeans_model_python_1537328280878_1)
- [clv_amt_dl_model.json](/c:/Users/vgang/Desktop/ai%20banking%20analysis/02_models/clv_amt_dl_model.json)
- [clv_amt_dl_model.h5](/c:/Users/vgang/Desktop/ai%20banking%20analysis/02_models/clv_amt_dl_model.h5)
- [clv_days_dl_model.json](/c:/Users/vgang/Desktop/ai%20banking%20analysis/02_models/clv_days_dl_model.json)
- [clv_days_dl_model.h5](/c:/Users/vgang/Desktop/ai%20banking%20analysis/02_models/clv_days_dl_model.h5)

### Churn Prediction and Survival Analysis

This part of the repository focuses on customer retention by estimating churn likelihood and analyzing survival patterns across customer groups.

Key components:

- [Customer_Churn_Prediction.py](/c:/Users/vgang/Desktop/ai%20banking%20analysis/01_code/01_02_clv_survival/Customer_Churn_Prediction.py)
- [Survival_Analysis.py](/c:/Users/vgang/Desktop/ai%20banking%20analysis/01_code/01_02_clv_survival/Survival_Analysis.py)
- [churn_pred.ipynb](/c:/Users/vgang/Desktop/ai%20banking%20analysis/03_ipy_notebooks/churn_pred.ipynb)

The project uses deep learning for churn scoring and Kaplan-Meier based survival analysis for retention-focused exploration.

### Recommendation Engine

The recommendation module demonstrates how transaction context and customer behavior can be used to generate targeted offers and recommendations.

Key components:

- [recommend_app.py](/c:/Users/vgang/Desktop/ai%20banking%20analysis/01_code/01_03_recommendation/recommend_app.py)
- [recommendation_ml.ipynb](/c:/Users/vgang/Desktop/ai%20banking%20analysis/03_ipy_notebooks/recommendation_ml.ipynb)
- [default.cfg](/c:/Users/vgang/Desktop/ai%20banking%20analysis/01_code/01_03_recommendation/default.cfg)

This workflow is designed around database-backed feature retrieval and external model scoring, followed by ranked recommendations and offer messaging.

## Repository Structure

```text
.
|-- 01_code/
|   |-- 01_01_fraud_detection/
|   |-- 01_02_clv_survival/
|   `-- 01_03_recommendation/
|-- 02_models/
|-- 03_ipy_notebooks/
|-- 04_documents/
|-- 05_visualizations/
|-- 98_references/
`-- 99_sample_data/
```

## Datasets and Assets

The repository includes sample data and model artifacts to support experimentation and demonstration.

Sample datasets:

- [Sample_CustTransactions_1354564.csv](/c:/Users/vgang/Desktop/ai%20banking%20analysis/99_sample_data/Sample_CustTransactions_1354564.csv)
- [Sample_CustChurnData.csv](/c:/Users/vgang/Desktop/ai%20banking%20analysis/99_sample_data/Sample_CustChurnData.csv)
- [customerpreddata.csv](/c:/Users/vgang/Desktop/ai%20banking%20analysis/99_sample_data/customerpreddata.csv)
- [custclv.csv](/c:/Users/vgang/Desktop/ai%20banking%20analysis/99_sample_data/custclv.csv)
- [creditcarddata.csv](/c:/Users/vgang/Desktop/ai%20banking%20analysis/99_sample_data/creditcarddata.csv)
- [reco_data.csv](/c:/Users/vgang/Desktop/ai%20banking%20analysis/99_sample_data/reco_data.csv)

Additional assets:

- Jupyter notebook output images in `03_ipy_notebooks`
- Reference material in [98_references](/c:/Users/vgang/Desktop/ai%20banking%20analysis/98_references)

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Keras
- H2O
- Lifelines
- Matplotlib
- Seaborn
- PyODBC
- Geopy
- Jupyter Notebook

## How to Run

There is no single application entry point. The project is best explored by running individual notebooks or scripts.

Example commands:

```bash
python 01_code/01_01_fraud_detection/fraud_detection_rules_deeplearning.py
python 01_code/01_02_clv_survival/Customer_NextTransaction_Prediction.py
python 01_code/01_02_clv_survival/CustomerLifetimeValue_Prediction_NewCustomer.py
python 01_code/01_02_clv_survival/Customer_Churn_Prediction.py
python 01_code/01_02_clv_survival/Survival_Analysis.py
```

For the best experience:

1. Explore the notebooks in `03_ipy_notebooks` first.
2. Keep the current folder structure unchanged because scripts rely on relative paths.
3. Install the required Python packages before running the workflows.

## Notes

- Some modules rely on older Python and library conventions.
- The recommendation workflow depends on external infrastructure such as SQL Server and a hosted scoring endpoint.
- The repository includes sample and illustrative code intended for learning, experimentation, and project demonstration.

## Future Improvements

- Add a `requirements.txt` or environment file
- Modernize the codebase for current Python and Keras versions
- Refactor scripts into reusable modules
- Add reproducible training pipelines
- Add tests and better configuration management

## License

This repository does not currently include a license file. If you plan to publish a fork or derivative version publicly, add the appropriate license before distribution.
