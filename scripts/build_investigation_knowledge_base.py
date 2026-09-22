"""Build a small local fraud-investigation knowledge base and FAISS index."""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = PROJECT_ROOT / "data" / "cases" / "historical_fraud_cases.csv"
DOCUMENTS_PATH = PROJECT_ROOT / "data" / "knowledge_base" / "fraud_investigation_documents.json"
INDEX_PATH = PROJECT_ROOT / "data" / "knowledge_base" / "fraud_investigation.faiss"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CASES_PER_FRAUD_TYPE = 2


POLICY_DOCUMENTS = [
    {
        "document_id": "POLICY-TRANSACTION-REVIEW",
        "title": "Transaction Review Policy",
        "document_type": "policy",
        "content": (
            "Policy: Review reported transaction details against authoritative records before "
            "making a fraud decision. International status, transaction timing, payment method, "
            "or merchant category alone do not prove fraud. Preserve the review rationale."
        ),
    },
    {
        "document_id": "POLICY-CUSTOMER-VERIFICATION",
        "title": "Customer Verification Policy",
        "document_type": "policy",
        "content": (
            "Policy: When account compromise is a concern, use approved customer verification "
            "steps and document the verified information. Do not request unnecessary sensitive "
            "information, and do not infer an outcome from a single transaction attribute."
        ),
    },
    {
        "document_id": "GUIDELINE-ACCOUNT-CONTROLS",
        "title": "Account Control Investigation Guideline",
        "document_type": "guideline",
        "content": (
            "Guideline: Review recent authentication events, failed attempts, PIN-change history, "
            "and transaction history when investigating a possible account-control issue. Compare "
            "the evidence with approved control procedures before taking action."
        ),
    },
    {
        "document_id": "GUIDELINE-EVIDENCE-HANDLING",
        "title": "Investigation Evidence Guideline",
        "document_type": "guideline",
        "content": (
            "Guideline: Record the transaction facts supplied by source systems, the policy context "
            "reviewed, and the next verification step. Historical cases are context only and are not "
            "proof that a new transaction has the same outcome."
        ),
    },
]


def historical_case_documents(cases: pd.DataFrame) -> list[dict[str, str]]:
    """Create concise case-summary documents from existing synthetic case records."""

    selected_cases = (
        cases.groupby("fraud_type", group_keys=False)
        .head(CASES_PER_FRAUD_TYPE)
        .sort_values("case_id")
    )
    return [
        {
            "document_id": f"HISTORY-{record.case_id}",
            "title": f"Historical case {record.case_id}",
            "document_type": "historical_case",
            "content": (
                f"Historical synthetic case summary. {record.description} "
                f"Recorded observations: {record.key_observations}. "
                f"Recorded outcome: {record.outcome}"
            ),
        }
        for record in selected_cases.itertuples(index=False)
    ]


def build_documents() -> list[dict[str, str]]:
    """Combine local policy, guideline, and historical-case documents."""

    cases = pd.read_csv(CASES_PATH)
    return POLICY_DOCUMENTS + historical_case_documents(cases)


def persist_documents_and_index(documents: list[dict[str, str]]) -> None:
    """Persist documents and a normalized MiniLM FAISS inner-product index."""

    DOCUMENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOCUMENTS_PATH.write_text(json.dumps(documents, indent=2), encoding="utf-8")
    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL_NAME,
        device="cpu",
        local_files_only=True,
    )
    embeddings = embedding_model.encode(
        [document["content"] for document in documents],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(INDEX_PATH))


def main() -> None:
    """Create and persist the local knowledge-base assets."""

    documents = build_documents()
    persist_documents_and_index(documents)
    print(f"Saved {len(documents)} knowledge-base documents to: {DOCUMENTS_PATH}")
    print(f"Saved FAISS index to: {INDEX_PATH}")


if __name__ == "__main__":
    main()
