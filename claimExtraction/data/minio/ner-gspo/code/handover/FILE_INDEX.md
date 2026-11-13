# File index (non-VERL)

## Active GSPO path

| File | Purpose |
|------|---------|
| `grpo/prepare_parquet.py` | Dataset → VERL parquet + MLflow |
| `grpo/reward_ie.py` | `compute_score` for VERL |
| `grpo/mlflow_utils.py` | MLflow URI/experiment helpers |
| `grpo/verify_gspo_math.py` | Sequence-ratio unit tests |
| `grpo/run_sft_coldstart.py` | SFT cold start + MLflow |
| `grpo/run_gspo_qwen3_0.6b.sh` | VERL GSPO launch |
| `grpo/eval_rollouts.py` | Offline eval + MLflow |
| `grpo/README.md` | Operator docs |
| `configs/gspo_qwen3_0.6b.yaml` | Hyperparameter reference |
| `scripts/00_*.sh` … `03_*.sh` | One-command pipeline |
| `requirements.txt` | Python deps note |
| `README.md` | Project entry |

## Data / outputs

| Path | Purpose |
|------|---------|
| `data/grpo/train.parquet` | Smoke train |
| `data/grpo/val.parquet` | Smoke val |
| `data/grpo/test.parquet` | Smoke test |
| `data/grpo/stats.json` | Prep stats |
| `data/grpo/example_*.json` | Schema examples |
| `outputs/sft_qwen3_0.6b_ie/final` | SFT weights |
| `outputs/gspo_qwen3_0.6b_ie/` | GSPO target (empty) |

## VERL (do not fork lightly)

| Path | Purpose |
|------|---------|
| `verl/verl/trainer/ppo/core_algos.py` | `compute_policy_loss_gspo` |
| `verl/examples/gspo_trainer/` | Reference launch scripts |
| `verl/verl/utils/tracking.py` | MLflow adapter |

## Legacy (secondary)

| Path | Purpose |
|------|---------|
| `src/architecture/*` | IndicBERT multi-task plug (abandoned as primary) |
| `configs/default.yaml` | Encoder multi-task defaults |
| `configs/causal.yaml` | Earlier causal Gemma notes |

## Source labels

| Path | Purpose |
|------|---------|
| `../NER_TOPIC_Modelling/data/ner_topic_llm_dataset.jsonl` | Full IE gold |
| `../NER_TOPIC_Modelling/taxonomy/` | Entity/topic taxonomies |
