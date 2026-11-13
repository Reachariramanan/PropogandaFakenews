#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /home/hariramanan/NTRO/NER_Training/.venv/bin/activate
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://<INTERNAL_IP>:5000}"
export MLFLOW_EXPERIMENT_NAME="${MLFLOW_EXPERIMENT_NAME:-gspo-ie-qwen3-0.6b}"
# Prefer SFT cold-start if present
if [[ -d "${ROOT}/outputs/sft_qwen3_0.6b_ie/final" ]]; then
  export MODEL_PATH="${MODEL_PATH:-${ROOT}/outputs/sft_qwen3_0.6b_ie/final}"
fi
bash "${ROOT}/grpo/run_gspo_qwen3_0.6b.sh" "$@"
