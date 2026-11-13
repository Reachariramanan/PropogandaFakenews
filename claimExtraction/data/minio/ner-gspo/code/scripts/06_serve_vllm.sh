#!/usr/bin/env bash
# Host a merged GSPO/SFT checkpoint behind an OpenAI-compatible vLLM server.
#
#   bash scripts/06_serve_vllm.sh outputs/gspo_qwen3_5_0.8b_sft_l19l23/global_step_100_merged
#   PORT=8200 GPU_MEM_UTIL=0.3 bash scripts/06_serve_vllm.sh <model_dir>
#
# Two environment fixes are baked in here (do not remove without re-verifying):
#
# 1. VLLM_USE_FLASHINFER_SAMPLER=0 — the installed FlashInfer build
#    (0.6.16.post3) bundles its own CCCL/CUB headers pinned to a specific CUDA
#    range. Neither nvcc available on this machine satisfies it: the system
#    nvcc (12.0) is too old (missing cub::BlockAdjacentDifference::FlagHeads),
#    and the venv-bundled nvcc (13.3, under .venv/lib/*/site-packages/nvidia/cu13)
#    is too new by FlashInfer's own compatibility check
#    (cuda/std/__cccl/cuda_toolkit.h errors "CUDA compiler and CUDA toolkit
#    headers are incompatible"). This is a genuine FlashInfer/toolkit version
#    gap, not a PATH issue — pointing at a different nvcc does not fix it.
#    Disabling the FlashInfer top-p/top-k sampler is vLLM's own documented
#    escape hatch for exactly this: it falls back to the native PyTorch
#    sampler with no output-quality difference, only marginally slower
#    sampling. Only remove this once FlashInfer or the CUDA toolkit on this
#    machine is upgraded to compatible versions and verified with a fresh
#    `vllm serve` with this flag unset.
#
# 2. --max-num-seqs: vLLM's default (256) reserves one Mamba cache block per
#    concurrent sequence for this Gated-DeltaNet hybrid model. At low
#    --gpu-memory-utilization (needed when sharing the GPU with other
#    tenants), the available block count can fall below 256, and CUDA graph
#    capture fails with "max_num_seqs exceeds available Mamba cache blocks".
#    Default here is conservative (64) to fit comfortably at low memory
#    budgets; raise it (and/or GPU_MEM_UTIL) if you have the whole GPU.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /home/hariramanan/NTRO/NER_Training/.venv/bin/activate

MODEL_DIR="${1:?usage: bash scripts/06_serve_vllm.sh <merged_model_dir>}"
if [[ ! -f "${MODEL_DIR}/config.json" ]]; then
  echo "No config.json under ${MODEL_DIR} — pass a MERGED (standalone HF) checkpoint dir, not a raw FSDP actor checkpoint."
  echo "Merge one first: python -m verl.model_merger merge --backend fsdp --local_dir <ckpt>/actor --target_dir <ckpt>_merged"
  exit 1
fi

SERVED_NAME="${SERVED_NAME:-$(basename "$(dirname "${MODEL_DIR}")")-$(basename "${MODEL_DIR}")}"
PORT="${PORT:-8101}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.09}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"

echo "Serving ${MODEL_DIR} as '${SERVED_NAME}' on port ${PORT} (gpu_mem_util=${GPU_MEM_UTIL}, max_num_seqs=${MAX_NUM_SEQS})"

VLLM_USE_FLASHINFER_SAMPLER=0 exec vllm serve "${MODEL_DIR}" \
  --served-model-name "${SERVED_NAME}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --port "${PORT}" \
  --trust-remote-code
