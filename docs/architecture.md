# Architecture

```mermaid
flowchart TB
    Client[API client] --> API[FastAPI app]
    API --> Routes[Prediction API routes]
    API --> Monitor[In-memory monitoring]
    Monitor --> Metrics[GET /metrics<br/>prediction count, errors, latency]

    Routes --> Predict[Prediction service]
    Predict --> Pipeline[Saved scikit-learn pipeline<br/>preprocessing + Random Forest]
    Pipeline --> ModelArtifact[models/fraud_detection_pipeline.joblib]

    Routes --> Explain[Explainability service]
    Explain --> Pipeline
    Explain --> SHAP[Tree SHAP contributions]
    Explain --> Fairness[Held-out group-rate fairness summary]
    Fairness --> ProcessedData[Processed POC sample]

    Routes --> CaseSearch[Similar-case service]
    CaseSearch --> MiniLM[MiniLM embeddings]
    MiniLM --> CaseFAISS[Historical-case FAISS index]
    CaseSearch --> CaseData[Historical case records]

    Routes --> Security[Security service]
    Security --> RAG[LangChain investigation service]
    Security --> Agents[LangGraph investigation workflow]
    RAG --> MiniLM
    RAG --> KnowledgeFAISS[Knowledge-base FAISS index]
    RAG --> KnowledgeData[Policies, guidelines,<br/>historical case summaries]
    RAG --> ProviderFactory[LLM provider factory]
    ProviderFactory --> OpenAI[OpenAI]
    ProviderFactory --> Groq[Groq]
    ProviderFactory --> Ollama[Ollama]
    Agents --> Predict
    Agents --> CaseSearch
    Agents --> RAG

    RawData[Public raw bank fraud CSV] --> Build[Runtime artifact build script]
    Build --> ProcessedData
    Build --> Pipeline
    Build --> CaseData
    Build --> CaseFAISS
    Build --> KnowledgeData
    Build --> KnowledgeFAISS
    Pipeline --> MLflow[Local MLflow tracking]

    Airflow[Airflow DAG] --> ProcessedData
    Airflow --> ModelArtifact
    Airflow --> CaseFAISS
    Airflow --> KnowledgeFAISS
    GitHubActions[GitHub Actions CI] --> Tests[pytest]
    GitHubActions --> Docker[Docker image build]
    Docker --> API
```

## Components

- **FastAPI:** exposes prediction, explainability, fairness, retrieval, investigation, agent, health, comparison, and metrics endpoints. Runtime monitoring records only `/predict` request count, error count, and latency in process memory.
- **Prediction and explainability:** the prediction service loads the saved preprocessing-plus-Random-Forest pipeline. The explainability service uses Tree SHAP and calculates descriptive fairness group rates from the processed POC sample.
- **Retrieval and investigation:** MiniLM creates embeddings for persisted FAISS indexes. Similar-case search retrieves historical cases; LangChain RAG retrieves policy, guideline, and case context before using a configured LLM or its deterministic fallback.
- **LLM and security:** the provider factory selects OpenAI, Groq, or Ollama from environment variables. The security service validates investigation inputs, filters retrieved content as untrusted, and checks generated output.
- **Agents:** the LangGraph workflow routes simple cases to direct assessment and complex cases through Fraud Analysis, Similar Case/Evidence, and Investigation Summary agents, using the existing services as tools.
- **MLOps:** final-model training logs parameters, metrics, and artifacts to local MLflow. The Airflow DAG validates saved runtime assets and performs a lightweight deterministic model evaluation without retraining.
- **Delivery:** Docker packages the FastAPI app, saved model, FAISS indexes, and knowledge-base artifacts. GitHub Actions installs dependencies, runs pytest, and builds the Docker image; it does not deploy.
