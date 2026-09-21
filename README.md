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

The project uses a public fraud transaction dataset containing 51,000
transaction records.

The target column is:

`Fraudulent`

- `0` — Legitimate transaction
- `1` — Fraudulent transaction

## Environment

Python 3.11

## Setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt