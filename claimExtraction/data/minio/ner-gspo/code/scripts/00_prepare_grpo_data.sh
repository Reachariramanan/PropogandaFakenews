#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /home/hariramanan/NTRO/NER_Training/.venv/bin/activate
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://<INTERNAL_IP>:5000}"
cd "${ROOT}"
python grpo/prepare_parquet.py \
  --source "${ROOT}/../NER_TOPIC_Modelling/data/ner_topic_llm_dataset.jsonl" \
  --out_dir "${ROOT}/data/grpo" \
  --experiment gspo-ie-qwen3-0.6b \
  "$@"
