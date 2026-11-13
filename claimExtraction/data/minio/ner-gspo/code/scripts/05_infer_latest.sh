#!/usr/bin/env bash
# Evaluate the most recent GSPO checkpoint, merging it first if needed.
#
# VERL writes LoRA-only checkpoints in its own FSDP .pt layout, which
# eval_rollouts.py (plain AutoModelForCausalLM.from_pretrained) cannot load. This
# merges the newest global_step_N into a standalone HF model, then evaluates it.
# Safe to run while training continues — merging only reads the checkpoint.
#
#   bash scripts/05_infer_latest.sh                          # newest checkpoint, val split
#   bash scripts/05_infer_latest.sh --max_samples 100
#   STEP=50 bash scripts/05_infer_latest.sh                  # a specific step
#   DATA=data/grpo_full/test.parquet bash scripts/05_infer_latest.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /home/hariramanan/NTRO/NER_Training/.venv/bin/activate
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://<INTERNAL_IP>:5000}"
export MLFLOW_EXPERIMENT_NAME="${MLFLOW_EXPERIMENT_NAME:-gspo-ie-qwen3-0.6b}"
export PYTHONPATH="${ROOT}/verl:${ROOT}:${PYTHONPATH:-}"
cd "${ROOT}"

CKPT_DIR="${CHECKPOINT_DIR:-${ROOT}/outputs/gspo_qwen3_5_0.8b_full}"

# Pick the checkpoint: explicit STEP, else the highest-numbered global_step_N.
if [[ -n "${STEP:-}" ]]; then
  SRC="${CKPT_DIR}/global_step_${STEP}"
else
  # `|| true` matters: under `set -o pipefail` a no-match `ls` would abort the
  # script before the friendly error below could run.
  SRC="$(ls -1d "${CKPT_DIR}"/global_step_* 2>/dev/null | sed 's/.*global_step_//' | sort -n | tail -1 | sed "s|^|${CKPT_DIR}/global_step_|" || true)"
  # Exclude the *_merged dirs from "newest step" detection.
  SRC="${SRC%_merged}"
fi

if [[ -z "${SRC:-}" || ! -d "${SRC}" ]]; then
  echo "No checkpoint found under ${CKPT_DIR}"
  echo "Train one first, or request a save mid-run:  bash scripts/train_ctl.sh save"
  exit 1
fi

MERGED="${SRC}_merged"
if [[ -f "${MERGED}/config.json" ]]; then
  echo "Using already-merged checkpoint: ${MERGED}"
else
  echo "Merging ${SRC} -> ${MERGED}"
  python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "${SRC}/actor" \
    --target_dir "${MERGED}"
fi

# Default to the full-data val split when present, else the older smoke split.
if [[ -z "${DATA:-}" ]]; then
  if [[ -f "${ROOT}/data/grpo_full/val.parquet" ]]; then
    DATA="${ROOT}/data/grpo_full/val.parquet"
  elif [[ -f "${ROOT}/data/grpo/val.minimal.parquet" ]]; then
    DATA="${ROOT}/data/grpo/val.minimal.parquet"
  else
    DATA="${ROOT}/data/grpo/val.parquet"
  fi
fi

STEP_NUM="$(basename "${SRC}" | sed 's/global_step_//')"
echo "model: ${MERGED}"
echo "data : ${DATA}"

python grpo/eval_rollouts.py \
  --model "${MERGED}" \
  --data "${DATA}" \
  --experiment "${MLFLOW_EXPERIMENT_NAME}" \
  --run_name "eval_gspo_step${STEP_NUM}" \
  "$@"
