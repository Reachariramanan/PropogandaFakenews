#!/usr/bin/env bash
# GSPO (+ optional LoRA) training for Qwen3.5-0.8B text-only.
# Thin wrapper around grpo/run_gspo_qwen3_5_0.8b.sh — pass flags/env straight through, e.g.
#   LORA_RANK=16 MAX_STEPS=3 bash scripts/04_gspo_train_qwen3_5.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /home/hariramanan/NTRO/NER_Training/.venv/bin/activate
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://<INTERNAL_IP>:5000}"
export MLFLOW_EXPERIMENT_NAME="${MLFLOW_EXPERIMENT_NAME:-gspo-ie-qwen3-0.6b}"

TEXT_ONLY="${ROOT}/outputs/qwen3_5_0.8b_text_only"
if [[ ! -f "${TEXT_ONLY}/config.json" ]]; then
  echo "Text-only checkpoint missing at ${TEXT_ONLY}"
  echo "Build it first:  python grpo/build_qwen3_5_text_only.py"
  exit 1
fi

# Prefer the minimal-schema split when present: it keeps generations inside the
# response budget so they stop truncating (and losing all reward). See
# grpo/prepare_parquet.py --schema and grpo/reschema_parquet.py.
if [[ -z "${TRAIN_FILE:-}" && -f "${ROOT}/data/grpo/train.minimal.parquet" ]]; then
  export TRAIN_FILE="${ROOT}/data/grpo/train.minimal.parquet"
  export VAL_FILE="${VAL_FILE:-${ROOT}/data/grpo/val.minimal.parquet}"
  echo "Using minimal-schema data: $(basename "${TRAIN_FILE}")"
fi

bash "${ROOT}/grpo/run_gspo_qwen3_5_0.8b.sh" "$@"
