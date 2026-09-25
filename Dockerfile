FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/huggingface \
    HF_HUB_DISABLE_XET=1 \
    HF_HUB_DOWNLOAD_TIMEOUT=300

WORKDIR /app

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

# Download MiniLM into the Docker image.
# The application loads it with local_files_only=True at runtime.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')"

COPY app ./app
COPY models/fraud_detection_pipeline.joblib ./models/fraud_detection_pipeline.joblib
COPY data/cases ./data/cases
COPY data/knowledge_base ./data/knowledge_base
COPY data/processed/bank_fraud_poc_sample.csv ./data/processed/bank_fraud_poc_sample.csv
COPY data/processed/text_model_comparison_results.json ./data/processed/text_model_comparison_results.json

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]