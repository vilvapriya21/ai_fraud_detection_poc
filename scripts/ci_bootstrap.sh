#!/usr/bin/env bash

# Generate every gitignored runtime artifact required by tests and Docker COPY.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

python scripts/generate_synthetic_bank_fraud.py
python scripts/create_poc_sample.py
python scripts/train_final_model.py
python scripts/build_similar_case_index.py
python scripts/build_investigation_knowledge_base.py
