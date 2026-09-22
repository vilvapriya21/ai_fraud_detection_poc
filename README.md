# AI-Powered Fraud Detection & Investigation Platform

A proof-of-concept AI/ML application for detecting potentially fraudulent
financial transactions and assisting analysts with fraud investigations.

## Project Status

Currently under development.

## Core Assessment Areas

- Transaction fraud-risk prediction
- Hyperparameter tuning
- Deep-learning model comparison
- Similar fraud case retrieval using embeddings
- RAG-based investigation assistant
- Multi-agent investigation using LangGraph
- Explainable AI and fairness analysis
- AI security controls
- MLflow experiment tracking
- Airflow workflow orchestration
- Docker and CI/CD

## Dataset

The project uses a public bank fraud transaction dataset containing approximately
1,000,000 transaction records and 26 columns.

For local POC development, a reproducible stratified sample of 100,000
transactions is created to reduce training and hyperparameter-tuning cost while
preserving the original fraud-class distribution.

Raw dataset:

`data/raw/bank_fraud.csv`

POC working sample:

`data/processed/bank_fraud_poc_sample.csv`

The supervised target is:

`is_fraud`

- `0` — Legitimate transaction
- `1` — Fraudulent transaction

The following columns are excluded from predictive features:

- `transaction_id` — transaction identifier
- `customer_id` — customer identifier
- `fraud_type` — post-outcome information and target leakage
- `transaction_date` — raw date representation
- `transaction_time` — raw high-cardinality time representation

Derived time features such as `hour_of_day`, `is_weekend`, and
`is_night_transaction` remain available to the model.

## Environment

Python 3.11

## Setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run the API

```powershell
uvicorn app.main:app --reload
```

The prediction endpoint is available at `POST /predict`; health is available at
`GET /health`.
