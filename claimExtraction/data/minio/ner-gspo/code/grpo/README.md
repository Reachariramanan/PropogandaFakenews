# GSPO IE — Qwen3-0.6B + VERL + MLflow

Causal free-form JSON extraction (`entities` + `events`) with **Group Sequence Policy Optimization** ([arXiv:2507.18071](https://arxiv.org/abs/2507.18071)).

## MLflow

| Setting | Value |
|---------|--------|
| UI | http://<INTERNAL_IP>:5000 |
| Experiment | `gspo-ie-qwen3-0.6b` |
| Env | `MLFLOW_TRACKING_URI=http://<INTERNAL_IP>:5000` |

**What gets logged**

| Stage | Run name pattern | Metrics / artifacts |
|-------|------------------|---------------------|
| Data prep | `prepare_parquet` | row counts, split sizes, example JSON |
| SFT cold-start | `sft_coldstart_*` | `sft/loss`, checkpoint meta |
| GSPO launch | `launch_*` | hyperparams snapshot |
| **GSPO train (VERL)** | `gspo_qwen3_0.6b_*` | trainer metrics via `trainer.logger=["console","mlflow"]` |
| Eval | `eval_*` | entity/topic F1, format, predictions.jsonl |

VERL uses `MLFLOW_TRACKING_URI` + `trainer.project_name` as the MLflow experiment name.

## Why GSPO (not token-level GRPO)

For one document, sample **G** JSON completions. Group-relative advantages only push **better-than-group-mean** outputs up and **worse** ones down. Sequence-level ratio + one clip decision avoids token-level clipping chaos on long JSON.

## Setup

```bash
source /home/hariramanan/NTRO/NER_Training/.venv/bin/activate
export MLFLOW_TRACKING_URI=http://<INTERNAL_IP>:5000
cd /home/hariramanan/NTRO/NER_Training
pip install -e ./verl   # once
pip install mlflow pyarrow pandas datasets tqdm
```

## Pipeline

```bash
# 0) GSPO math self-check (optional)
PYTHONPATH=verl python grpo/verify_gspo_math.py
python grpo/reward_ie.py

# 1) Data → parquet + MLflow stats
bash scripts/00_prepare_grpo_data.sh
# smoke: bash scripts/00_prepare_grpo_data.sh --max_docs 500

# 2) SFT cold-start (recommended)
bash scripts/01_sft_coldstart.sh
# smoke: bash scripts/01_sft_coldstart.sh --max_samples 200 --max_steps 20

# 3) GSPO RL
bash scripts/02_gspo_train.sh
# smoke: MAX_STEPS=20 ROLLOUT_N=2 TRAIN_BATCH_SIZE=4 bash scripts/02_gspo_train.sh

# 4) Eval + MLflow
bash scripts/03_eval.sh --max_samples 30
```

## Key files

| File | Role |
|------|------|
| `grpo/reward_ie.py` | VERL `compute_score` (format + entity/topic/action F1) |
| `grpo/prepare_parquet.py` | JSONL → parquet + MLflow |
| `grpo/run_gspo_qwen3_0.6b.sh` | GSPO launch, `loss_mode=gspo`, MLflow logger |
| `grpo/mlflow_utils.py` | tracking URI helpers |
| `verl/` | upstream VERL (GSPO already registered) |

## GSPO knobs

```bash
CLIP_RATIO_LOW=3e-4 CLIP_RATIO_HIGH=4e-4   # paper-scale sequence clip (not 0.2)
ROLLOUT_N=4                                 # group size
MODEL_PATH=outputs/sft_qwen3_0.6b_ie/final
```

## Checkpoints

- SFT: `outputs/sft_qwen3_0.6b_ie/final`
- GSPO: `outputs/gspo_qwen3_0.6b_ie/`
