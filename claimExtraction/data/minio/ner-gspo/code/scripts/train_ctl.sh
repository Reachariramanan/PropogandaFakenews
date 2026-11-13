#!/usr/bin/env bash
# Control a running GSPO job: checkpoint on demand, or stop it cleanly so it can
# be resumed later from exactly where it left off.
#
#   bash scripts/train_ctl.sh status        # is it running, what step, what checkpoints exist
#   bash scripts/train_ctl.sh save          # checkpoint after the current step, keep training
#   bash scripts/train_ctl.sh stop          # checkpoint after the current step, then exit cleanly
#   bash scripts/train_ctl.sh kill          # hard kill (only after `stop` finished, or to abandon)
#
# `save`/`stop` are honored at the END of the step in flight, so allow one step
# (~2 min at batch 16) before the checkpoint appears. They work by dropping a
# sentinel file in the checkpoint dir, which the training loop polls each step —
# this is deliberately not a signal to the launcher, because the loop runs inside
# a Ray actor in a different process.
#
# Override the target run with CHECKPOINT_DIR=... if not the default.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_DIR="${CHECKPOINT_DIR:-${ROOT}/outputs/gspo_qwen3_5_0.8b_full}"
CMD="${1:-status}"

latest_step() {
  if [[ -f "${CKPT_DIR}/latest_checkpointed_iteration.txt" ]]; then
    cat "${CKPT_DIR}/latest_checkpointed_iteration.txt"
  else
    echo "none"
  fi
}

case "${CMD}" in
  save|stop)
    if [[ ! -d "${CKPT_DIR}" ]]; then
      echo "Checkpoint dir does not exist: ${CKPT_DIR}"
      echo "Set CHECKPOINT_DIR to the running job's trainer.default_local_dir."
      exit 1
    fi
    if ! pgrep -f "verl.trainer.main_ppo" > /dev/null 2>&1; then
      echo "WARNING: no verl.trainer.main_ppo process found — is training actually running?"
      echo "Writing the sentinel anyway; it will be consumed by the next run that starts."
    fi
    if [[ "${CMD}" == "save" ]]; then
      touch "${CKPT_DIR}/SAVE_NOW"
      echo "Requested checkpoint (SAVE_NOW). Training continues afterwards."
    else
      touch "${CKPT_DIR}/STOP_NOW"
      echo "Requested checkpoint + clean stop (STOP_NOW)."
    fi
    echo "Watch for it with:  bash scripts/train_ctl.sh status"
    echo "Current saved step: $(latest_step)"
    ;;

  kill)
    echo "Hard-killing trainer, Ray, and vLLM. Anything since the last checkpoint is lost."
    # Match on the module path, and exclude this script's own command line.
    for pid in $(pgrep -f "verl\.trainer\.main_ppo" 2>/dev/null || true); do
      [[ "${pid}" == "$$" ]] && continue
      kill -KILL "${pid}" 2>/dev/null && echo "  killed trainer ${pid}"
    done
    ray stop --force > /dev/null 2>&1 || true
    sleep 1
    echo "GPU now:"
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
    ;;

  status)
    echo "checkpoint dir : ${CKPT_DIR}"
    echo "latest saved   : $(latest_step)"
    if pgrep -f "verl\.trainer\.main_ppo" > /dev/null 2>&1; then
      echo "trainer        : RUNNING"
    else
      echo "trainer        : not running"
    fi
    for f in SAVE_NOW STOP_NOW; do
      [[ -f "${CKPT_DIR}/${f}" ]] && echo "pending req    : ${f} (not yet consumed)"
    done
    echo "checkpoints    :"
    ls -1d "${CKPT_DIR}"/global_step_* 2>/dev/null | sed 's/^/  /' || echo "  none yet"
    echo "gpu            : $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader)"
    ;;

  *)
    echo "usage: bash scripts/train_ctl.sh {status|save|stop|kill}"
    exit 1
    ;;
esac
