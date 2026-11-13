#!/usr/bin/env bash
# GSPO + VIGOR training on the **v2** data + reward stack (Qwen3.5-0.8B text-only).
#
# v2 is a data/reward change, NOT a training-loop change: this wrapper reuses
# grpo/run_gspo_vigor.sh verbatim (same GSPO sequence-level loss, GRPO
# group-relative advantages, VIGOR variance-guided rollout allocation, LoRA and
# the Qwen3.5 correctness constraints) and only redirects three of its env-var
# override points:
#
#   TRAIN_FILE / VAL_FILE -> data/grpo_v2/{train,val}.parquet   (was data/grpo/*.minimal.parquet)
#   REWARD_PATH           -> grpo/reward_ie_v2.py               (was grpo/reward_ie.py)
#
# Everything else is inherited, so every knob run_gspo_vigor.sh documents still
# works here (MODEL_PATH, LORA_RANK, TRAIN_BATCH_SIZE, PPO_MINI_BATCH_SIZE,
# MAX_PROMPT_LENGTH, MAX_RESPONSE_LENGTH, VIGOR_M0/T/GAMMA/ALPHA, MAX_STEPS, ...).
#
# Usage:
#   LORA_RANK=16 \
#   MODEL_PATH=outputs/gspo_vigor_qwen3_5_0.8b_run1/global_step_1010_merged \
#   bash scripts/08_gspo_train_v2.sh
#
#   MAX_STEPS=3 TRAIN_BATCH_SIZE=4 PPO_MINI_BATCH_SIZE=4 bash scripts/08_gspo_train_v2.sh  # smoke
#
# MODEL_PATH is deliberately left to run_gspo_vigor.sh's default
# (outputs/qwen3_5_0.8b_text_only) so you can start v2 from the base text-only
# backbone or from any merged checkpoint by exporting MODEL_PATH yourself.
#
# PPO_MINI_BATCH_SIZE must be lowered alongside TRAIN_BATCH_SIZE: it defaults to
# 16 (grpo/run_gspo_vigor.sh), and ActorConfig.validate rejects
# train_batch_size < ppo_mini_batch_size before Ray even starts.
#
# CHECKPOINT_DIR defaults to a NEW directory, distinct from the v1 VIGOR run's
# outputs/gspo_vigor_qwen3_5_0.8b_run1. This matters: trainer.resume_mode=auto
# reads latest_checkpointed_iteration.txt from the checkpoint dir, so pointing
# v2 at the v1 dir would silently resume (and then overwrite) the v1 run.
#
# Compare with scripts/07_gspo_vigor_train.sh — same loop on the v1 data/reward.
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

# ---- v2 data (override TRAIN_FILE/VAL_FILE to point elsewhere) ----
export TRAIN_FILE="${TRAIN_FILE:-${ROOT}/data/grpo_v2/train.parquet}"
export VAL_FILE="${VAL_FILE:-${ROOT}/data/grpo_v2/val.parquet}"
for f in "${TRAIN_FILE}" "${VAL_FILE}"; do
  if [[ ! -f "${f}" ]]; then
    echo "Missing v2 data file: ${f}"
    echo "Build it first:  python grpo/prepare_parquet_v2.py --source data/ner_topic_llm_dataset.jsonl --out_dir data/grpo_v2"
    exit 1
  fi
done

# ---- v2 reward (same reward.custom_reward_function.path hook as v1) ----
export REWARD_PATH="${REWARD_PATH:-${ROOT}/grpo/reward_ie_v2.py}"
if [[ ! -f "${REWARD_PATH}" ]]; then
  echo "Missing v2 reward function: ${REWARD_PATH}"
  exit 1
fi

# ---- fresh checkpoint dir, never the live v1 run's ----
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-${ROOT}/outputs/gspo_vigor_v2_qwen3_5_0.8b_ie}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-gspo_vigor_v2_qwen3_5_0.8b_$(date +%Y%m%d_%H%M%S)}"

# prepare_parquet_v2.py already fits every prompt inside a 1536-token budget
# (stats.json: prompt_max_tokens_seen=1536, prompt_unfittable=0), which matches
# run_gspo_vigor.sh's MAX_PROMPT_LENGTH default — so data.filter_overlong_prompts
# should drop nothing. Raise both together if you rebuild with a larger budget.
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1536}"

echo "v2 train : ${TRAIN_FILE}"
echo "v2 val   : ${VAL_FILE}"
echo "v2 reward: ${REWARD_PATH}"
echo "ckpt dir : ${CHECKPOINT_DIR}"
echo "model    : ${MODEL_PATH:-${ROOT}/outputs/qwen3_5_0.8b_text_only (default)}"

bash "${ROOT}/grpo/run_gspo_vigor.sh" "$@"
