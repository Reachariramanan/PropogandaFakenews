#!/usr/bin/env bash
# GSPO on Qwen3.5-0.8B (text-only backbone) for IE JSON generation (entities + events).
# MLflow: http://<INTERNAL_IP>:5000  experiment gspo-ie-qwen3-0.6b
#
# Qwen3.5-0.8B is a multimodal (vision+text) hybrid architecture: 18 of 24 decoder
# layers are Gated DeltaNet (linear attention), 6 are standard full attention
# (layers 3,7,11,15,19,23 0-indexed, per full_attention_interval=4). We only need
# the text backbone. MODEL_PATH below must point at a self-contained local
# checkpoint with architectures=["Qwen3_5ForCausalLM"] baked into config.json —
# both the FSDP actor (AutoModelForCausalLM) and vLLM's `serve <path>` (which reads
# config.json straight off disk, with no config-override hook of its own) need to
# see the same on-disk config to both resolve to the text-only class. Build it once:
#
#   python3 -c "
#   import torch
#   from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
#   cfg = AutoConfig.from_pretrained('Qwen/Qwen3.5-0.8B').get_text_config()
#   cfg.architectures = ['Qwen3_5ForCausalLM']
#   m = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3.5-0.8B', config=cfg, torch_dtype=torch.bfloat16)
#   m.save_pretrained('outputs/qwen3_5_0.8b_text_only')
#   AutoTokenizer.from_pretrained('Qwen/Qwen3.5-0.8B').save_pretrained('outputs/qwen3_5_0.8b_text_only')
#   "
#
# use_remove_padding is forced False: VERL's Qwen3.5 GatedDeltaNet monkey-patch
# (verl/verl/models/transformers/qwen3_5.py) threads cu_seqlens into the
# torch-fallback chunk_gated_delta_rule (used when flash-linear-attention isn't
# installed), but that fallback silently ignores cu_seqlens instead of resetting
# recurrent state at sequence boundaries — with use_remove_padding=True this would
# silently leak state across packed samples (wrong training signal, no crash).
# Only re-enable it after installing flash-linear-attention + causal-conv1d and
# verifying packed vs. unpacked GDN outputs match per-sample.
#
# Usage (from NER_Training, with Entity_extraction venv):
#   source /home/hariramanan/NTRO/NER_Training/.venv/bin/activate
#   bash grpo/run_gspo_qwen3_5_0.8b.sh
#
# Env overrides: MODEL_PATH, ROLLOUT_N, CLIP_RATIO_LOW, CLIP_RATIO_HIGH,
#                TRAIN_BATCH_SIZE, MLFLOW_TRACKING_URI, TOTAL_EPOCHS, MAX_STEPS,
#                ATTN_IMPLEMENTATION (default sdpa), LORA_RANK (0=disabled;
#                >0 enables adapter-only training scoped to the final decoder
#                block, layer 23 of 24 — a standard full-attention layer),
#                LORA_ALPHA, LORA_TARGET_MODULES, USE_REMOVE_PADDING

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

# Since use_remove_padding=False, dynamic token-budget batching isn't meaningful
# (no packing to fill a token budget without padding waste) — use fixed
# micro-batch sizes instead. Kept small given padded batches at
# MAX_RESPONSE_LENGTH cost more memory per sample than packed sequences would.
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-2}"
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-2}"

# LoRA (optional): set LORA_RANK>0 to train adapters only, instead of full
# fine-tuning. Default scope is the final decoder block only (layer 23 of 24 —
# confirmed via config.json layer_types to be a standard full-attention layer,
# not Gated DeltaNet, so standard q/k/v/o_proj + gate/up/down_proj names apply).
LORA_RANK="${LORA_RANK:-0}"
LORA_ALPHA="${LORA_ALPHA:-$((LORA_RANK * 2))}"
if [[ -z "${LORA_TARGET_MODULES:-}" ]]; then
  LORA_TARGET_MODULES='.*\.layers\.23\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$'
fi

ACTOR_LR="${ACTOR_LR:-1e-6}"
# Paper GSPO clips (NOT GRPO 0.2)
CLIP_RATIO_LOW="${CLIP_RATIO_LOW:-3e-4}"
CLIP_RATIO_HIGH="${CLIP_RATIO_HIGH:-4e-4}"

ROLLOUT_N="${ROLLOUT_N:-4}"
ROLLOUT_TP="${ROLLOUT_TP:-1}"
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.45}"

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
EXPERIMENT_NAME="${EXPERIMENT_NAME:-gspo_qwen3_5_0.8b_$(date +%Y%m%d_%H%M%S)}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${ROOT}/outputs/gspo_qwen3_5_0.8b_ie}"

REWARD_PATH="${REWARD_PATH:-${ROOT}/grpo/reward_ie.py}"
########################### end user-adjustable ###########################

if [[ ! -f "${TRAIN_FILE}" ]]; then
  echo "Missing ${TRAIN_FILE}. Run: python grpo/prepare_parquet.py"
  exit 1
fi

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "Missing ${MODEL_PATH}/config.json. Build the text-only checkpoint first (see header comment)."
  exit 1
fi

mkdir -p "${CHECKPOINT_DIR}"

# Export for the pre-log block
export ROOT MODEL_PATH ROLLOUT_N CLIP_RATIO_LOW CLIP_RATIO_HIGH
export TRAIN_BATCH_SIZE INFER_BACKEND EXPERIMENT_NAME MLFLOW_EXPERIMENT_NAME MLFLOW_TRACKING_URI

# Log launch metadata to MLflow before VERL (survives if VERL logger misconfigured)
python - <<'PY'
import os, sys
sys.path.insert(0, os.environ["ROOT"])
os.environ.setdefault("MLFLOW_TRACKING_URI", "http://<INTERNAL_IP>:5000")
try:
    from grpo.mlflow_utils import mlflow_run, log_metrics, log_dict_as_json
    params = {
        "model_path": os.environ.get("MODEL_PATH"),
        "rollout_n": os.environ.get("ROLLOUT_N"),
        "clip_ratio_low": os.environ.get("CLIP_RATIO_LOW"),
        "clip_ratio_high": os.environ.get("CLIP_RATIO_HIGH"),
        "train_batch_size": os.environ.get("TRAIN_BATCH_SIZE"),
        "infer_backend": os.environ.get("INFER_BACKEND"),
        "algo": "gspo",
        "adv_estimator": "grpo",
        "base_model": "Qwen3.5-0.8B-text-only",
    }
    with mlflow_run(
        run_name=f"launch_{os.environ.get('EXPERIMENT_NAME','gspo')}",
        tags={"stage": "gspo_launch", "algo": "gspo"},
        params=params,
        experiment_name=os.environ.get("MLFLOW_EXPERIMENT_NAME"),
    ):
        log_metrics({"launch_ok": 1.0})
        log_dict_as_json(params, "gspo_launch_params.json")
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
  # layered_summon is a multi-GPU memory optimization (gathers sharded FSDP
  # params one unit at a time); on a single GPU FSDP uses NO_SHARD (nothing to
  # shard), and its fallback path uses summon_full_params(offload_to_cpu=True),
  # which PyTorch's FSDP doesn't support for NO_SHARD. Keep this False here.
  EXTRA+=(actor_rollout_ref.rollout.layered_summon=False)
fi

python -m verl.trainer.main_ppo \
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
