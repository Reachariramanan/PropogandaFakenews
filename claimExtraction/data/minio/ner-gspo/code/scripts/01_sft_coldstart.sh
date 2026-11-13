#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /home/hariramanan/NTRO/NER_Training/.venv/bin/activate
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://<INTERNAL_IP>:5000}"
cd "${ROOT}"
python grpo/run_sft_coldstart.py \
  --train_file "${ROOT}/data/grpo/train.parquet" \
  --output_dir "${ROOT}/outputs/sft_qwen3_0.6b_ie" \
  --experiment gspo-ie-qwen3-0.6b \
  "$@"
