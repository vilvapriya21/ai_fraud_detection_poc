# AI Fraud Detection POC - Agent Instructions

## Project Purpose

This repository implements an AI-Powered Fraud Detection & Investigation Platform
as a single-person AI/ML proof of concept.

The implementation must stay aligned with the assessment requirements.

Main required capabilities:

- Transaction fraud-risk prediction
- Grid Search, Random Search, and Bayesian Optimization
- CNN, RNN, and LSTM comparison
- Attention / Transformer demonstration
- Pre-trained model usage
- Semantic embeddings
- FAISS or Chroma vector search
- LangChain RAG
- LangGraph or CrewAI multi-agent workflow
- AI security and prompt-injection protection
- Explainability using SHAP, LIME, or equivalent
- Basic fairness analysis
- MLflow experiment tracking
- Airflow workflow
- FastAPI backend
- Docker
- Docker Compose
- CI/CD
- Monitoring and health endpoint
- Documentation and reproducibility

## Development Approach

Build the project incrementally, module by module.

Do not implement unrelated future features unless explicitly requested.

Do not generate the entire application in one step.

Prefer the simplest correct implementation suitable for a working POC.

Avoid unnecessary enterprise abstractions, excessive wrappers, unnecessary design
patterns, or deeply nested folder structures.

Preserve existing working functionality when modifying code.

## Backend First

Focus on backend implementation first.

Do not create frontend code unless explicitly requested.

FastAPI will be used for the application/API layer.

## Python Version

Use Python 3.11.

Virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## Install Dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run Tests

```powershell
pytest
```

When tests are added, prefer:

```powershell
pytest -v
```

## Run FastAPI

Once FastAPI is implemented:

```powershell
uvicorn app.main:app --reload
```

## Code Quality Rules

Use:

* Python type hints
* Meaningful docstrings for public classes and functions
* Clear variable and function names
* Small focused functions
* PEP 8 style
* Explicit error handling where appropriate
* Reusable modules for production logic

Avoid:

* Decorative comments
* Excessive comments that merely repeat the code
* Empty abstractions
* Overengineering
* Huge functions
* Hard-coded secrets
* Unnecessary dependencies

## Notebook Rules

Use notebooks for:

* Data understanding
* Exploratory data analysis
* ML experimentation
* Model comparison
* Hyperparameter experiments
* Explainability analysis
* Fairness analysis

Use numbered notebook names, for example:

```text
01_data_understanding.ipynb
02_preprocessing_experiments.ipynb
03_baseline_model.ipynb
04_hyperparameter_tuning.ipynb
05_deep_learning_comparison.ipynb
06_explainability_fairness.ipynb
```

Do not put reusable production logic only inside notebooks.

Once logic is finalized, move reusable functionality into Python modules.

## Data Rules

Raw source data must remain unchanged.

Current source dataset location:

```text
data/raw/fraud_detection.csv
```

Never overwrite the raw dataset.

Processed data should go under:

```text
data/processed/
```

Generated historical cases should go under:

```text
data/cases/
```

RAG knowledge-base documents should go under:

```text
data/knowledge_base/
```

Do not use real customer financial information.

Only use public, anonymized, or synthetic data.

## Dataset Target

The supervised fraud target is:

```text
Fraudulent
```

Values:

```text
0 = legitimate
1 = fraudulent
```

Do not use identifiers such as `Transaction_ID` as predictive ML features.

Treat `User_ID` as an identifier unless an experiment explicitly justifies its use.

Always check for data leakage before training.

## ML Evaluation Rules

The dataset is imbalanced.

Do not evaluate the fraud model using accuracy alone.

Use appropriate metrics such as:

* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion matrix

Model selection must be documented and based on appropriate fraud-detection metrics.

## Dependency Rules

Only add dependencies when required by the currently implemented module.

Whenever adding a dependency:

1. Add it to `requirements.txt`.
2. Explain why it is required.
3. Avoid duplicate libraries that solve the same problem unless necessary.

Do not install packages globally.

## Secrets and Environment Variables

Never hard-code:

* API keys
* Tokens
* Passwords
* Database credentials
* LLM credentials

Use environment variables.

Real secrets belong in:

```text
.env
```

Example keys belong in:

```text
.env.example
```

Never commit `.env`.

## Testing Rules

Add tests for reusable production functionality.

Tests belong under:

```text
tests/
```

Do not use notebooks as a substitute for automated tests.

Security controls must eventually include at least:

* Prompt-injection test
* Jailbreak-style test
* Sensitive-data request test
* Malicious retrieved-content test

## Verification Loop

Before reporting a task as complete:

1. Run `pytest -v` and confirm all tests pass.
2. If the module includes a FastAPI route, start the app and confirm the relevant
   endpoint responds correctly.
3. Report actual command output, not an assumption that it worked.

## Architecture Rules

Keep architecture production-style but appropriate for a single-person POC.

Preferred responsibility boundaries:

```text
app/api/        FastAPI routes
app/core/       Configuration and shared infrastructure
app/schemas/    Pydantic request/response models
app/services/   Application-level services
app/ml/         ML inference and preprocessing
app/rag/        Embeddings, retrieval, and RAG
app/agents/     LangGraph/CrewAI workflows
app/security/   AI safety and validation
```

Do not create folders or abstractions before they are actually needed.

## Docker Rules

Docker will be implemented after core modules are stable.

The final repository must include:

```text
Dockerfile
docker-compose.yml
```

Docker and Docker Compose must actually run the application rather than existing
only for documentation.

## Documentation Rules

Update `README.md` as features are implemented.

README should eventually contain:

* Project purpose
* Architecture
* Setup
* Environment variables
* Dataset information
* Training instructions
* API execution
* Docker execution
* MLflow instructions
* Airflow instructions
* Testing
* Assessment feature mapping
* Limitations

## Git Rules

Do not commit:

* `.venv`
* `.env`
* Python caches
* notebook checkpoints
* runtime logs
* temporary MLflow data
* Airflow runtime data
* generated model binaries unless explicitly required
* generated vector indexes unless explicitly required

Make focused changes.

Do not rewrite unrelated files during a feature implementation.

## Working Style

Before implementing a requested module:

1. Inspect the existing repository.
2. Reuse existing working code where appropriate.
3. State which files will be changed.
4. Implement only the requested scope.
5. Run relevant tests or validation.
6. Report what changed.
7. Report commands needed to verify the result.

When assumptions are necessary, make them explicit.

Do not silently change business or assessment requirements.