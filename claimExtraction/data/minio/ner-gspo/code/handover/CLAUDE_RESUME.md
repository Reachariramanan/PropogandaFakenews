# Paste into Claude Code

Continue work on **NER_Training GSPO IE**.

## Read first

1. `/home/hariramanan/NTRO/NER_Training/handover/HANDOVER.md`
2. `/home/hariramanan/NTRO/NER_Training/handover/STATUS.md`
3. `/home/hariramanan/NTRO/NER_Training/handover/NEXT_STEPS.md`
4. `/home/hariramanan/NTRO/NER_Training/grpo/README.md`

## Immediate task

Unblock and **smoke GSPO** (3 steps) with:

- Venv: `source /home/hariramanan/NTRO/Entity_extraction/.venv/bin/activate`
- Model init: `/home/hariramanan/NTRO/NER_Training/outputs/sft_qwen3_0.6b_ie/final`
- Script: `grpo/run_gspo_qwen3_0.6b.sh` (`loss_mode=gspo`, group adv, MLflow)
- Tracking: `MLFLOW_TRACKING_URI=http://<INTERNAL_IP>:5000`, experiment `gspo-ie-qwen3-0.6b`

**Last known error:** FlashAttention2 toggled but not installed. Fix with `attn_implementation=sdpa` (or install flash-attn). `trainer.use_v1=False` already set (no `transfer_queue`).

After smoke passes, prepare full dataset and document MLflow run links.

## Constraints

- Causal free JSON generation (entities+events), not BIO-only.
- GSPO sequence-level clipping (paper ε ~ 3e-4/4e-4), not GRPO 0.2 token clips.
- Do not commit secrets.
