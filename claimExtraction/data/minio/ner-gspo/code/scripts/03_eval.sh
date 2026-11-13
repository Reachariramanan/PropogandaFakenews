#!/usr/bin/env bash
# Offline eval + MLflow. Auto-selects the best available checkpoint unless
# MODEL_PATH is set explicitly.
#
#   bash scripts/03_eval.sh --max_samples 30              # auto-pick
#   MODEL_PATH=outputs/... bash scripts/03_eval.sh        # explicit
#   DATA=data/grpo/test.minimal.parquet bash scripts/03_eval.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /home/hariramanan/NTRO/NER_Training/.venv/bin/activate
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://<INTERNAL_IP>:5000}"
cd "${ROOT}"

# Preference order: explicit > merged GSPO LoRA (Qwen3.5) > Qwen3.5 base
# text-only > Qwen3-0.6B SFT > hub base. LoRA-only VERL checkpoints are NOT
# loadable here — merge them first:
#   python -m verl.model_merger merge --backend fsdp \
#     --local_dir outputs/gspo_qwen3_5_0.8b_ie/global_step_N/actor \
#     --target_dir outputs/gspo_qwen3_5_0.8b_ie/global_step_N_merged
if [[ -z "${MODEL_PATH:-}" ]]; then
  # newest merged GSPO checkpoint, if any
  MERGED="$(ls -1dt "${ROOT}"/outputs/gspo_qwen3_5_0.8b_ie/*_merged 2>/dev/null | head -1 || true)"
  if [[ -n "${MERGED}" && -f "${MERGED}/config.json" ]]; then
    MODEL_PATH="${MERGED}"
  elif [[ -f "${ROOT}/outputs/qwen3_5_0.8b_text_only/config.json" ]]; then
    MODEL_PATH="${ROOT}/outputs/qwen3_5_0.8b_text_only"
  elif [[ -f "${ROOT}/outputs/sft_qwen3_0.6b_ie/final/config.json" ]]; then
    MODEL_PATH="${ROOT}/outputs/sft_qwen3_0.6b_ie/final"
  else
    MODEL_PATH="Qwen/Qwen3-0.6B"
  fi
fi

# Match the data schema to the model where possible: a model trained on the
# minimal schema must be scored against minimal-schema prompts.
if [[ -z "${DATA:-}" ]]; then
  if [[ -f "${ROOT}/data/grpo/val.minimal.parquet" ]]; then
    DATA="${ROOT}/data/grpo/val.minimal.parquet"
  else
    DATA="${ROOT}/data/grpo/val.parquet"
  fi
fi

echo "model: ${MODEL_PATH}"
echo "data : ${DATA}"

python grpo/eval_rollouts.py \
  --model "${MODEL_PATH}" \
  --data "${DATA}" \
  --experiment "${MLFLOW_EXPERIMENT_NAME:-gspo-ie-qwen3-0.6b}" \
  "$@"
