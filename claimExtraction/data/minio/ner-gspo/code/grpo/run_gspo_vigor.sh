#!/usr/bin/env bash
# GSPO + VIGOR (variance-guided progressive rollout allocation) on Qwen3.5-0.8B
# (text-only backbone) for IE JSON generation (entities + events).
# MLflow: http://<INTERNAL_IP>:5000  experiment gspo-ie-qwen3-0.6b
#
# This is grpo/run_gspo_qwen3_5_0.8b.sh with one substitution: instead of
# `python -m verl.trainer.main_ppo` (which draws a fixed rollout.n completions
# per prompt), it launches `grpo/run_gspo_vigor.py`, which keeps every piece of
# VERL's machinery but replaces the single-generate-per-step schedule with
# VIGOR's multi-round variance-guided allocation. Everything else — the GSPO
# sequence-level loss, GRPO group-relative advantages, LoRA, the Qwen3.5
# correctness constraints below — is identical.
#
# VIGOR in one paragraph: each training step runs up to T rounds of generation
# against a FROZEN policy snapshot. Round 0 gives every prompt m0 rollouts;
# each later round keeps the ceil(alpha * |active|) prompts whose CUMULATIVE
# reward variance is highest and gives each survivor m0*gamma^round new
# rollouts. Rollouts are merged, never discarded, so groups end up with
# variable sizes — which VERL's GRPO advantage estimator already handles,
# because it groups by the `uid` column rather than assuming a uniform count.
# With the defaults alpha=0.5 / gamma=2 (alpha*gamma == 1), the total rollout
# budget per step is exactly train_batch_size * m0 * T — the same spend as
# baseline GSPO with rollout.n = m0*T — just concentrated on the prompts whose
# reward signal is actually informative.
#
# NOTE: ROLLOUT_N is deliberately NOT a VIGOR knob. VIGOR replaces the fixed
# rollout_n concept entirely with its m0/T/gamma/alpha schedule, and the custom
# loop never calls gen_batch.repeat(rollout.n). rollout.n survives only as the
# multiplier VERL uses to turn ppo_mini_batch_size (prompts) into a row count
# (verl/trainer/ppo/ray_trainer.py:1328) and to size real_train_batch_size in
# validate_config. It therefore defaults to m0*T below — the equivalent uniform
# rollouts-per-prompt that VIGOR spends — so the actor mini-batch works out to
# the same number of rows a fixed-rollout_n=m0*T run would use.
#
# --- Qwen3.5-0.8B specifics (identical to run_gspo_qwen3_5_0.8b.sh) ---
#
# Qwen3.5-0.8B is a multimodal (vision+text) hybrid: 18 of 24 decoder layers are
# Gated DeltaNet (linear attention), 6 are standard full attention (layers
# 3,7,11,15,19,23, per full_attention_interval=4). MODEL_PATH must point at a
# self-contained local checkpoint with architectures=["Qwen3_5ForCausalLM"] baked
# into config.json — build it once with:
#
#   python grpo/build_qwen3_5_text_only.py
#
# use_remove_padding is forced False: VERL's Qwen3.5 GatedDeltaNet monkey-patch
# (verl/verl/models/transformers/qwen3_5.py) threads cu_seqlens into the
# torch-fallback chunk_gated_delta_rule (used when flash-linear-attention isn't
# installed), but that fallback silently ignores cu_seqlens instead of resetting
# recurrent state at sequence boundaries — with use_remove_padding=True this
# would silently leak state across packed samples (wrong training signal, no
# crash). Only re-enable after installing flash-linear-attention +
# causal-conv1d and verifying packed vs. unpacked GDN outputs match per-sample.
#
# Usage (from NER_Training, with Entity_extraction venv):
#   source /home/hariramanan/NTRO/NER_Training/.venv/bin/activate
#   bash grpo/run_gspo_vigor.sh
#
# Env overrides: everything run_gspo_qwen3_5_0.8b.sh accepts —
#                MODEL_PATH, TRAIN_FILE, VAL_FILE, ROLLOUT_N, CLIP_RATIO_LOW,
#                CLIP_RATIO_HIGH, TRAIN_BATCH_SIZE, PPO_MINI_BATCH_SIZE,
#                MAX_PROMPT_LENGTH, MAX_RESPONSE_LENGTH, ACTOR_LR, LORA_RANK,
#                LORA_ALPHA, LORA_TARGET_MODULES, USE_REMOVE_PADDING,
#                ATTN_IMPLEMENTATION, INFER_BACKEND, TOTAL_EPOCHS, MAX_STEPS,
#                SAVE_FREQ, TEST_FREQ, CHECKPOINT_DIR, MLFLOW_TRACKING_URI —
#                PLUS the VIGOR schedule:
#                VIGOR_M0 (default 2), VIGOR_T (4), VIGOR_GAMMA (2.0),
#                VIGOR_ALPHA (0.5), VIGOR_SCORE_SOURCE (auto|direct|rm).

set -xeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

# ---- MLflow (VERL reads MLFLOW_TRACKING_URI) ----
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://<INTERNAL_IP>:5000}"
export MLFLOW_EXPERIMENT_NAME="${MLFLOW_EXPERIMENT_NAME:-gspo-ie-qwen3-0.6b}"

# Prefer project venv if active; else Entity_extraction
if [[ -z "${VIRTUAL_ENV:-}" && -f /home/hariramanan/NTRO/NER_Training/.venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source /home/hariramanan/NTRO/NER_Training/.venv/bin/activate
fi

export PYTHONPATH="${ROOT}/verl:${ROOT}:${PYTHONPATH:-}"

########################### user-adjustable ###########################
MODEL_PATH="${MODEL_PATH:-${ROOT}/outputs/qwen3_5_0.8b_text_only}"
TRAIN_FILE="${TRAIN_FILE:-${ROOT}/data/grpo/train.parquet}"
VAL_FILE="${VAL_FILE:-${ROOT}/data/grpo/val.parquet}"

NNODES="${NNODES:-1}"
NGPUS_PER_NODE="${NGPUS_PER_NODE:-1}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1536}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}"

ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"

# Correctness-critical: see header note. Do not flip to True without installing
# flash-linear-attention + causal-conv1d and verifying packed-vs-unpadded parity.
USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-False}"

PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-2}"
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-2}"

# LoRA (optional): set LORA_RANK>0 to train adapters only. Default scope is the
# final decoder block (layer 23 of 24 — a standard full-attention layer).
LORA_RANK="${LORA_RANK:-0}"
LORA_ALPHA="${LORA_ALPHA:-$((LORA_RANK * 2))}"
if [[ -z "${LORA_TARGET_MODULES:-}" ]]; then
  LORA_TARGET_MODULES='.*\.layers\.23\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$'
fi

ACTOR_LR="${ACTOR_LR:-1e-6}"
# Paper GSPO clips (NOT GRPO 0.2)
CLIP_RATIO_LOW="${CLIP_RATIO_LOW:-3e-4}"
CLIP_RATIO_HIGH="${CLIP_RATIO_HIGH:-4e-4}"

ROLLOUT_TP="${ROLLOUT_TP:-1}"
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.45}"

# ---------------------------- VIGOR schedule ----------------------------
VIGOR_M0="${VIGOR_M0:-2}"          # rollouts per prompt in round 0
VIGOR_T="${VIGOR_T:-4}"            # number of rounds per training step
VIGOR_GAMMA="${VIGOR_GAMMA:-2.0}"  # rollout growth factor for survivors
VIGOR_ALPHA="${VIGOR_ALPHA:-0.5}"  # fraction of active prompts kept per round

# See header: not a generation knob under VIGOR. Defaults to the equivalent
# uniform rollouts-per-prompt (m0*T) so ppo_mini_batch_size maps to the same
# row count a fixed-rollout_n run would use.
ROLLOUT_N="${ROLLOUT_N:-$((VIGOR_M0 * VIGOR_T))}"
# auto  = reuse the rm_scores the agent loop already computed with
#         grpo/reward_ie.py (guarantees selection reward == training reward)
# direct= always decode + call grpo/reward_ie.py::compute_score on the driver
# rm    = require rm_scores, error if absent
VIGOR_SCORE_SOURCE="${VIGOR_SCORE_SOURCE:-auto}"
# ------------------------------------------------------------------------

# Auto-pick rollout backend
if python -c "import vllm" 2>/dev/null; then
  INFER_BACKEND="${INFER_BACKEND:-vllm}"
else
  echo "WARN: vllm not installed; using hf rollout (slower)"
  INFER_BACKEND="${INFER_BACKEND:-hf}"
fi

TOTAL_EPOCHS="${TOTAL_EPOCHS:-2}"
SAVE_FREQ="${SAVE_FREQ:-20}"
TEST_FREQ="${TEST_FREQ:-10}"
# Optional smoke: MAX_STEPS=20
MAX_STEPS="${MAX_STEPS:-}"

PROJECT_NAME="${PROJECT_NAME:-${MLFLOW_EXPERIMENT_NAME}}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-gspo_vigor_qwen3_5_0.8b_$(date +%Y%m%d_%H%M%S)}"
# NOTE: resume_mode=auto reads latest_checkpointed_iteration.txt from this dir.
# Point it somewhere new when you intend a fresh run — otherwise VERL silently
# resumes old weights.
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${ROOT}/outputs/gspo_vigor_qwen3_5_0.8b_ie}"

REWARD_PATH="${REWARD_PATH:-${ROOT}/grpo/reward_ie.py}"
########################### end user-adjustable ###########################

if [[ ! -f "${TRAIN_FILE}" ]]; then
  echo "Missing ${TRAIN_FILE}. Run: python grpo/prepare_parquet.py"
  exit 1
fi

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "Missing ${MODEL_PATH}/config.json. Build the text-only checkpoint first: python grpo/build_qwen3_5_text_only.py"
  exit 1
fi

mkdir -p "${CHECKPOINT_DIR}"

# Export for the pre-log block
export ROOT MODEL_PATH ROLLOUT_N CLIP_RATIO_LOW CLIP_RATIO_HIGH
export TRAIN_BATCH_SIZE INFER_BACKEND EXPERIMENT_NAME MLFLOW_EXPERIMENT_NAME MLFLOW_TRACKING_URI
export VIGOR_M0 VIGOR_T VIGOR_GAMMA VIGOR_ALPHA VIGOR_SCORE_SOURCE

# Log launch metadata to MLflow before VERL (survives if VERL logger misconfigured)
python - <<'PY'
import os, sys
sys.path.insert(0, os.environ["ROOT"])
os.environ.setdefault("MLFLOW_TRACKING_URI", "http://<INTERNAL_IP>:5000")
try:
    from grpo.mlflow_utils import mlflow_run, log_metrics, log_dict_as_json
    m0 = int(os.environ.get("VIGOR_M0", 2))
    T = int(os.environ.get("VIGOR_T", 4))
    B = int(os.environ.get("TRAIN_BATCH_SIZE", 32))
    params = {
        "model_path": os.environ.get("MODEL_PATH"),
        "clip_ratio_low": os.environ.get("CLIP_RATIO_LOW"),
        "clip_ratio_high": os.environ.get("CLIP_RATIO_HIGH"),
        "train_batch_size": B,
        "infer_backend": os.environ.get("INFER_BACKEND"),
        "algo": "gspo",
        "adv_estimator": "grpo",
        "allocator": "vigor",
        "vigor_m0": m0,
        "vigor_T": T,
        "vigor_gamma": os.environ.get("VIGOR_GAMMA"),
        "vigor_alpha": os.environ.get("VIGOR_ALPHA"),
        "vigor_score_source": os.environ.get("VIGOR_SCORE_SOURCE"),
        "vigor_budget_rows_per_step": B * m0 * T,
        "base_model": "Qwen3.5-0.8B-text-only",
    }
    with mlflow_run(
        run_name=f"launch_{os.environ.get('EXPERIMENT_NAME','gspo_vigor')}",
        tags={"stage": "gspo_vigor_launch", "algo": "gspo", "allocator": "vigor"},
        params=params,
        experiment_name=os.environ.get("MLFLOW_EXPERIMENT_NAME"),
    ):
        log_metrics({"launch_ok": 1.0})
        log_dict_as_json(params, "gspo_vigor_launch_params.json")
    print("MLflow launch metadata logged")
except Exception as e:
    print(f"WARN: MLflow pre-log failed: {e}")
PY

EXTRA=()
if [[ -n "${MAX_STEPS}" ]]; then
  EXTRA+=(trainer.total_training_steps="${MAX_STEPS}")
fi
if [[ "${LORA_RANK}" -gt 0 ]]; then
  EXTRA+=(actor_rollout_ref.model.lora_rank="${LORA_RANK}")
  EXTRA+=(actor_rollout_ref.model.lora_alpha="${LORA_ALPHA}")
  EXTRA+=("actor_rollout_ref.model.target_modules='${LORA_TARGET_MODULES}'")
  EXTRA+=(actor_rollout_ref.rollout.load_format=safetensors)
  # layered_summon is a multi-GPU memory optimization; on a single GPU FSDP uses
  # NO_SHARD and its fallback (summon_full_params(offload_to_cpu=True)) is
  # unsupported for NO_SHARD. Keep this False here.
  EXTRA+=(actor_rollout_ref.rollout.layered_summon=False)
fi

python "${ROOT}/grpo/run_gspo_vigor.py" \
  vigor.m0="${VIGOR_M0}" \
  vigor.T="${VIGOR_T}" \
  vigor.gamma="${VIGOR_GAMMA}" \
  vigor.alpha="${VIGOR_ALPHA}" \
  vigor.score_source="${VIGOR_SCORE_SOURCE}" \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.use_remove_padding="${USE_REMOVE_PADDING}" \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  +actor_rollout_ref.model.override_config.attn_implementation="${ATTN_IMPLEMENTATION}" \
  actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
  actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
  actor_rollout_ref.actor.clip_ratio_low="${CLIP_RATIO_LOW}" \
  actor_rollout_ref.actor.clip_ratio_high="${CLIP_RATIO_HIGH}" \
  actor_rollout_ref.actor.clip_ratio_c=10.0 \
  actor_rollout_ref.actor.optim.lr="${ACTOR_LR}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.use_dynamic_bsz=False \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name="${INFER_BACKEND}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM_UTIL}" \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
  reward.custom_reward_function.path="${REWARD_PATH}" \
  reward.custom_reward_function.name=compute_score \
  trainer.critic_warmup=0 \
  trainer.logger='["console","mlflow"]' \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
  trainer.nnodes="${NNODES}" \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  trainer.default_local_dir="${CHECKPOINT_DIR}" \
  trainer.val_before_train=True \
  trainer.use_v1=False \
  "${EXTRA[@]}" \
  "$@"

echo "Checkpoints: ${CHECKPOINT_DIR}"
echo "MLflow UI: ${MLFLOW_TRACKING_URI}  experiment=${PROJECT_NAME}  run=${EXPERIMENT_NAME}"
