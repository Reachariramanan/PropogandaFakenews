# Handover for Claude — NER_Training GSPO IE

**Date:** 2026-08-10  
**Owner context:** Grok session → hand to Claude Code  
**Project root:** `/home/hariramanan/NTRO/NER_Training`  
**Dataset:** `/home/hariramanan/NTRO/NER_TOPIC_Modelling/data/ner_topic_llm_dataset.jsonl` (~80.6k docs)

---

## Mission (what the user wants)

1. **Causal** free-form generation of IE JSON (`entities` + `events`) — **not** locked BIO/classification heads.
2. Model: **`Qwen/Qwen3-0.6B`** ([HF](https://huggingface.co/Qwen/Qwen3-0.6B)).
3. RL: **GSPO** (Group Sequence Policy Optimization, [arXiv:2507.18071](https://arxiv.org/abs/2507.18071)) via **VERL** (`./verl` clone).
4. **Not** IndicBERT for generation (bidirectional; wrong tool for `.generate()` / RL rollouts).
5. Multiple JSON samples per doc; **group-relative** rewards so only worse-than-group-mean completions are penalized.
6. Wire everything to **MLflow** at `http://<INTERNAL_IP>:5000`.
7. Labels / subtypes (e.g. military ranks) exist in dataset as `subtype`; no `army_officer` literal — ranks like `colonel`, `general`, `soldier_trooper`.

---

## Environment

```bash
source /home/hariramanan/NTRO/Entity_extraction/.venv/bin/activate
export MLFLOW_TRACKING_URI=http://<INTERNAL_IP>:5000
export MLFLOW_EXPERIMENT_NAME=gspo-ie-qwen3-0.6b
cd /home/hariramanan/NTRO/NER_Training
export PYTHONPATH="${PWD}/verl:${PWD}:${PYTHONPATH:-}"
```

| Item | Value |
|------|--------|
| GPU | 1× NVIDIA L40S 49GB |
| Python | 3.12.3 (Entity_extraction venv) |
| torch | 2.13.0+cu130 (was working) |
| transformers | may be **5.x** after `pip install -e ./verl` (conflict with sentence-transformers) |
| MLflow | 3.15.1 → server **http://<INTERNAL_IP>:5000** |
| Experiment | **`gspo-ie-qwen3-0.6b`** (id **27**) |
| VERL | local clone at `NER_Training/verl`, commit ~`b75f0d0`, installed editable |

---

## Architecture decisions (locked)

### GSPO (not token-level GRPO)

- **Advantages:** `algorithm.adv_estimator=grpo` (group-relative outcome rewards).
- **Policy loss:** `actor_rollout_ref.actor.policy_loss.loss_mode=gspo`  
  Implementation already in VERL: `verl/trainer/ppo/core_algos.py` → `compute_policy_loss_gspo`.
- **Sequence ratio:**  
  \(s_i=\exp(\mathrm{mean}_{t\in\text{response}}[\log\pi_\theta-\log\pi_{\mathrm{old}}])\)  
  one clip per sequence: paper-scale **`clip_ratio_low=3e-4`**, **`clip_ratio_high=4e-4`** (not GRPO’s 0.2).
- **Agg:** `loss_agg_mode=seq-mean-token-mean`.
- Do **not** reimplement GSPO unless unit tests show paper mismatch; `grpo/verify_gspo_math.py` covers core ratio math.

### Reward (rule-based vs gold JSON)

`grpo/reward_ie.py` → `compute_score(data_source, solution_str, ground_truth, extra_info)`  

Weighted: format 0.15 + entity F1 0.40 + subtype 0.10 + topic F1 0.25 + action F1 0.10.

### Generation target (soft schema)

Same shape as dataset labels:

```json
{
  "entities": [{"text","type","subtype","score","normalized","evidence_text",...}],
  "events": [{"actor1","actor2","other_actors","topics","actions","evidence_text",...}]
}
```

No constrained decoding — free causal generation + reward.

---

## Repo map (primary path)

```
NER_Training/
  handover/                 ← you are here
  grpo/                     ← main GSPO stack
    prepare_parquet.py      # JSONL → VERL parquet + MLflow
    reward_ie.py            # VERL custom reward
    mlflow_utils.py         # tracking helpers
    verify_gspo_math.py     # GSPO formula checks
    run_sft_coldstart.py    # SFT before RL + MLflow
    run_gspo_qwen3_0.6b.sh  # VERL GSPO launch + MLflow
    eval_rollouts.py        # offline generate + reward + MLflow
    README.md
  scripts/
    00_prepare_grpo_data.sh
    01_sft_coldstart.sh
    02_gspo_train.sh
    03_eval.sh
  configs/gspo_qwen3_0.6b.yaml
  data/grpo/                # smoke parquet (200-doc prep)
  outputs/sft_qwen3_0.6b_ie/final/   # SFT smoke checkpoint EXISTS
  outputs/gspo_qwen3_0.6b_ie/        # empty / no successful GSPO ckpt yet
  verl/                     # upstream VERL clone
  src/                      # LEGACY IndicBERT multi-task sketches — not the active path
```

---

## Smoke status (as of handover)

| Step | Status | Notes |
|------|--------|--------|
| Reward unit test | **PASS** | gold-self 1.0, garbage 0.0 |
| GSPO math tests | **PASS** | VERL import needs `PYTHONPATH=verl` / installed package |
| Prepare 200 docs + MLflow | **PASS** | run `prepare_parquet` on exp 27 |
| SFT smoke (50 samples, ~15 steps) | **PASS** | `outputs/sft_qwen3_0.6b_ie/final`, MLflow run logged, loss ~1.3 |
| GSPO smoke | **BLOCKED** | see below |
| Full 80k parquet | **NOT RUN** | only 200-doc smoke |

### MLflow runs already present (exp 27)

- `prepare_parquet` — data stats + artifacts  
- `sft_coldstart_Qwen3-0.6B` — SFT smoke  
- `launch_gspo_*` — launch metadata only (training did not finish)

UI: http://<INTERNAL_IP>:5000/#/experiments/27

### GSPO failure trail (fix these in order)

1. **Missing `ray`** → installed `ray 2.56.1`.
2. **Missing `verl` package** → `pip install -e ./verl`.
3. **V1 trainer needs `transfer_queue`** (not on PyPI as `transfer-queue`).  
   Fix applied in script: **`trainer.use_v1=False`** (legacy TaskRunner).
4. **No `vllm`** → smoke used `INFER_BACKEND=hf` (slower; OK for smoke).
5. **Latest failure (v0 path):**  
   ```
   ImportError: FlashAttention2 has been toggled on, but ... doesn't seem to be installed.
   ```  
   Model load forces FA2. Fix options for Claude:
   - Install flash-attn (may be heavy), **or**
   - Force attention impl in config, e.g.  
     `actor_rollout_ref.model.override_config.attn_implementation=sdpa`  
     or `eager` / `sdpa` via VERL model kwargs (inspect `verl` config for exact key — often `actor_rollout_ref.model.override_config` or `+actor_rollout_ref.model.attn_implementation=sdpa`).
   - Or set env / patch `from_pretrained(..., attn_implementation="sdpa")`.

**GSPO has not completed a single training step yet.** Highest priority after resume.

---

## Commands to resume

```bash
source /home/hariramanan/NTRO/Entity_extraction/.venv/bin/activate
export MLFLOW_TRACKING_URI=http://<INTERNAL_IP>:5000
export MLFLOW_EXPERIMENT_NAME=gspo-ie-qwen3-0.6b
cd /home/hariramanan/NTRO/NER_Training
export PYTHONPATH="${PWD}/verl:${PWD}:${PYTHONPATH:-}"

# 1) Unblock GSPO model load (pick one)
# pip install flash-attn --no-build-isolation   # if feasible
# OR add attn_implementation=sdpa to run_gspo_qwen3_0.6b.sh

# 2) Minimal GSPO smoke
MODEL_PATH=$PWD/outputs/sft_qwen3_0.6b_ie/final \
MAX_STEPS=3 ROLLOUT_N=2 TRAIN_BATCH_SIZE=4 PPO_MINI_BATCH_SIZE=4 \
MAX_PROMPT_LENGTH=1024 MAX_RESPONSE_LENGTH=512 \
INFER_BACKEND=hf TOTAL_EPOCHS=1 TEST_FREQ=100 \
bash grpo/run_gspo_qwen3_0.6b.sh

# 3) After GSPO works — full data
bash scripts/00_prepare_grpo_data.sh   # full 80k
bash scripts/01_sft_coldstart.sh
bash scripts/02_gspo_train.sh
```

---

## MLflow wiring (already done)

| Component | Mechanism |
|-----------|-----------|
| Prep / SFT / eval | `grpo/mlflow_utils.py` (`setup_mlflow`, `mlflow_run`, `log_metrics`, `log_artifact`) |
| GSPO train | `export MLFLOW_TRACKING_URI=...` + VERL `trainer.logger='["console","mlflow"]'` + `trainer.project_name=gspo-ie-qwen3-0.6b` |
| VERL source | `verl/utils/tracking.py` reads `MLFLOW_TRACKING_URI` |

Do not hardcode HF tokens. User previously shared a HF token in chat — use env only if needed for gated models; Qwen3-0.6B is ungated.

---

## Dataset notes for Claude

- Top-level entity types: `PERSON`, `ORGANIZATION`, `LOCATION`, `PRODUCT`, `EVENT` (+ rare noise HDL/PR/MEN/CLM/HTG filtered in prep).
- PERSON subtypes include military ranks (`general`, `colonel`, `captain`, `soldier_trooper`, …) — **not** a single `army_officer` label.
- Events have topics (CAMEO-style `01.1`…) and actions (`13`–`20`).
- `char_start`/`char_end` are null in source labels; generative path does not need spans.
- Sample mid datapoint (earlier): `data/sample_mid_datapoint.json` if present; else `data/grpo/example_gt_pretty.json`.

---

## Explicit non-goals / do not reopen unless asked

- IndicBERT-v3 as causal generator  
- Locked BIO multi-task as primary product path  
- Reimplementing GSPO from scratch when VERL `loss_mode=gspo` exists  
- Committing secrets / HF tokens  

---

## Success criteria for “done”

- [ ] GSPO smoke ≥3 steps without OOM / ImportError  
- [ ] Checkpoints under `outputs/gspo_qwen3_0.6b_ie/`  
- [ ] Metrics in MLflow experiment 27 (reward, clipfrac, etc.)  
- [ ] Optional: full parquet + longer SFT + multi-epoch GSPO  
- [ ] `eval_rollouts.py` logs entity/topic F1 to MLflow  

---

## Suggested first message for Claude Code

> Continue GSPO IE training under `/home/hariramanan/NTRO/NER_Training`. Read `handover/HANDOVER.md` and `handover/STATUS.md`. Unblock GSPO (FlashAttention2 / use sdpa). Smoke 3 GSPO steps with SFT ckpt at `outputs/sft_qwen3_0.6b_ie/final`, log to MLflow `http://<INTERNAL_IP>:5000` experiment `gspo-ie-qwen3-0.6b`. Then full data prep if smoke passes.

Also read: `grpo/README.md`, `configs/gspo_qwen3_0.6b.yaml`, last error log under session terminal if needed.
