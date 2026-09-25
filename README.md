# AI-Powered Fraud Detection & Investigation Platform

A Python 3.11 proof of concept for fraud-risk prediction and analyst-facing investigation support. The FastAPI application combines a saved Random Forest pipeline, SHAP evidence, basic fairness summaries, semantic case retrieval, grounded RAG, and a LangGraph workflow.

## Architecture

FastAPI routes call reusable services rather than training models at request time. The prediction service loads the saved scikit-learn preprocessing-plus-Random-Forest pipeline. MiniLM embeddings query persisted FAISS indexes for similar cases and investigation documents. LangChain RAG uses a switchable OpenAI, Groq, or Ollama provider, with a deterministic fallback when no LLM is available. Security checks protect investigation and agent flows.

See [architecture.md](docs/architecture.md) for the Mermaid diagram and component relationships.

## Feature Status

| Area | Status |
| --- | --- |
| Tabular data understanding, preprocessing, and baseline/tuning notebooks | Implemented |
| Saved Random Forest prediction pipeline | Implemented |
| CNN, RNN, LSTM, attention-style, and pretrained text-model comparison evidence | Implemented as experimentation/results artifact |
| Similar-case search with MiniLM and FAISS | Implemented |
| LangChain RAG with visible sources and fallback | Implemented |
| OpenAI, Groq, and Ollama provider factory | Implemented |
| Security controls for investigation inputs and retrieved content | Implemented |
| LangGraph multi-agent investigation workflow | Implemented |
| SHAP explanations and descriptive fairness summaries | Implemented |
| In-memory prediction monitoring | Implemented |
| MLflow logging, Airflow validation DAG, Docker, and GitHub Actions CI | Implemented |

## Dataset Pivot

The early Phase 1 notebooks use `data/raw/fraud_detection.csv` with target `Fraudulent`. Signal analysis and model experiments on that dataset produced near-chance ROC-AUC results, so it was not used for the application prediction pipeline.

The application uses the public/synthetic bank-fraud dataset at `data/raw/bank_fraud.csv`, with target `is_fraud`. A deterministic 100,000-row stratified working sample is written to `data/processed/bank_fraud_poc_sample.csv`. The raw datasets are never overwritten.

## Setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

## LLM Configuration

Set one provider in `.env`; never commit this file.

```text
LLM_PROVIDER=openai
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4.1-mini
```

```text
LLM_PROVIDER=groq
GROQ_API_KEY=
GROQ_MODEL=
```

```text
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=
```

If the selected provider is unavailable or unconfigured, the investigation service returns its safe deterministic fallback.

## Build Runtime Artifacts

Place the required public/synthetic raw bank-fraud CSV at `data/raw/bank_fraud.csv`, then run:

```powershell
python scripts/build_runtime_artifacts.py
```

The script creates only missing artifacts in this order:

1. Reproducible processed sample.
2. Final saved model, including local MLflow logging.
3. Historical similar-case records and FAISS index.
4. Investigation knowledge-base documents and FAISS index.

It fails clearly if the processed sample is missing and `data/raw/bank_fraud.csv` is unavailable.

## Run the API

```powershell
uvicorn app.main:app --reload
```

Interactive API documentation is available at `http://127.0.0.1:8000/docs`.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Saved-model availability |
| `GET` | `/metrics` | In-memory `/predict` count, error count, and latency |
| `POST` | `/predict` | Random Forest fraud-risk prediction |
| `POST` | `/explain` | Prediction with top SHAP feature contributions |
| `GET` | `/fairness` | Cached descriptive group-rate fairness summary |
| `GET` | `/model-comparison` | Saved text-model comparison results |
| `POST` | `/similar-cases` | MiniLM/FAISS historical-case retrieval |
| `POST` | `/investigate` | Security-checked, source-grounded RAG response |
| `POST` | `/agent-investigate` | Security-checked LangGraph investigation workflow |

## Tests

```powershell
pytest -v
```

## MLflow

`scripts/train_final_model.py` logs final-model parameters, Precision, Recall, F1, ROC-AUC, and model artifacts to the local `mlflow.db` store.

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Docker

The image includes the FastAPI application, saved Random Forest pipeline, MiniLM cache, FAISS indexes, and required application data. It does not retrain at startup.

```powershell
docker build -t ai-fraud-detection-poc .
docker run --rm --env-file .env --add-host host.docker.internal:host-gateway -p 8000:8000 ai-fraud-detection-poc
```

Or use Compose:

```powershell
docker compose up --build
```

For Ollama from a Docker container, configure `OLLAMA_BASE_URL=http://host.docker.internal:11434` in `.env`.

## Airflow

`dags/fraud_mlops_dag.py` validates the processed sample, final model, FAISS/knowledge-base assets, and runs a lightweight deterministic evaluation without retraining.

```powershell
$env:AIRFLOW__CORE__DAGS_FOLDER = (Resolve-Path .\dags)
airflow dags test fraud_mlops_dag 2026-09-25
```

## CI/CD

`.github/workflows/ci.yml` runs on pushes and pull requests. It installs dependencies, runs `pytest -v`, and builds the Docker image. It does not deploy the application.

## Limitations

- This is a POC built from public/synthetic data; its fraud metrics, fairness summaries, and historical cases are not production evidence.
- The fairness endpoint is descriptive, uses operational groups and age bands, and does not establish real-world fairness or policy compliance.
- Similarity retrieval is contextual evidence, not proof of a matching fraud outcome.
- RAG output is grounded in the local knowledge base but requires human review; its deterministic fallback is deliberately conservative.
- Monitoring is process-local and resets on restart; it is not a persistent observability system.
- Docker and CI builds require the saved model and runtime artifacts to be available in the build context.

## Final Demo Flow

1. Build missing runtime artifacts with `python scripts/build_runtime_artifacts.py`.
2. Start the API and confirm `GET /health`.
3. Submit a transaction to `POST /predict`, then show `POST /explain` and `GET /metrics`.
4. Show `GET /fairness` and `GET /model-comparison`.
5. Retrieve related cases with `POST /similar-cases`.
6. Run `POST /investigate` and `POST /agent-investigate` to show grounded sources, security handling, and the conditional agent route.
