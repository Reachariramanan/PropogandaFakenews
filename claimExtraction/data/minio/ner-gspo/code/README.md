# NER_Training

Primary path: **causal GSPO** IE with **Qwen3-0.6B** + **VERL** + **MLflow**.

Generates free-form JSON (`entities` + `events`) from `../NER_TOPIC_Modelling/data/ner_topic_llm_dataset.jsonl`.

## MLflow

| | |
|--|--|
| UI | **http://<INTERNAL_IP>:5000** |
| Experiment | `gspo-ie-qwen3-0.6b` |
| Env | `export MLFLOW_TRACKING_URI=http://<INTERNAL_IP>:5000` |

Logged stages: data prep, SFT, GSPO launch + VERL training metrics, eval.

## Quick start

```bash
source /home/hariramanan/NTRO/Entity_extraction/.venv/bin/activate
export MLFLOW_TRACKING_URI=http://<INTERNAL_IP>:5000
cd /home/hariramanan/NTRO/NER_Training

pip install -r requirements.txt
pip install -e ./verl   # once (needs ray, etc.)

bash scripts/00_prepare_grpo_data.sh          # + MLflow prepare_parquet run
bash scripts/01_sft_coldstart.sh              # optional cold start
bash scripts/02_gspo_train.sh                 # GSPO RL, logger=console+mlflow
bash scripts/03_eval.sh --max_samples 30
```

Full details: **`grpo/README.md`**.

## Layout

| Path | Role |
|------|------|
| `grpo/` | GSPO prep, reward, launch, MLflow helpers |
| `verl/` | Local VERL clone (`loss_mode=gspo` built-in) |
| `data/grpo/` | train/val/test parquet |
| `outputs/gspo_qwen3_0.6b_ie/` | GSPO checkpoints |
| `configs/gspo_qwen3_0.6b.yaml` | Defaults |

## Algorithm note

**GSPO** (sequence-level importance ratio + group-relative advantages), not token-level GRPO clipping. Paper: [arXiv:2507.18071](https://arxiv.org/abs/2507.18071). Clips default `3e-4` / `4e-4`.

---

Legacy encoder-style multi-task sketches live under `src/` / `configs/default.yaml` (IndicBERT plug); generative path is under `grpo/`.

## Security

Never commit `HF_TOKEN`. Use environment variables only.
