# Plan: Causal GSPO IE with Qwen3-0.6B + VERL

## Goal

Train **`Qwen/Qwen3-0.6B`** (true causal LM) with **Group Relative Policy Optimization (GRPO)** via **[verl](https://github.com/verl-project/verl)** on `ner_topic_llm_dataset.jsonl`, so the model **freely generates** rich JSON (`entities` + `events`) rather than locked BIO/classification heads.

IndicBERT-v3 is **out of scope** for this path (bidirectional encoder; not for `.generate()` / GRPO rollouts).

Hardware: **1× NVIDIA L40S 49GB** (fits 0.6B GRPO comfortably).
Tracking: **MLflow** `http://<INTERNAL_IP>:5000` (experiment `causal-grpo-ie-qwen3-0.6b`).

---

## Why this stack

| Piece             | Choice                                                   | Reason                                                                         |
| ----------------- | -------------------------------------------------------- | ------------------------------------------------------------------------------ |
| Model             | [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | Causal, small, ungated, VERL/Qwen3-ready                                       |
| RL algo           | **GRPO**                                           | Critic-less; group-relative advantages; good for verifiable structured rewards |
| Framework         | `NER_Training/verl` (already cloned)                   | Production GRPO + FSDP + vLLM/HF rollout + custom rewards                      |
| “Theta capping” | `clip_ratio` / dual-clip                               | Trust-region clamp on policy ratio (see below)                                 |
| Labels            | LLM JSONL entities+events                                | Soft schema via reward, not constrained decoding                               |

### GRPO in one picture

```
prompt (doc) ──► policy samples n completions (group)
                      │
                      ▼
              custom reward R_i  (JSON quality vs ground truth)
                      │
                      ▼
         advantage A_i = (R_i − mean_group) / std_group
                      │
                      ▼
         clipped policy gradient + optional KL to ref
```

Knobs (VERL):

- `algorithm.adv_estimator=grpo`
- `actor_rollout_ref.rollout.n` ≥ 2 (group size; use **4–8** on L40S)
- **Theta / clip capping** → `actor_rollout_ref.actor.clip_ratio` (default 0.2)
  - Asymmetric: `clip_ratio_low` / `clip_ratio_high`
  - Dual-clip floor: `clip_ratio_c` (>1, default 3.0)
  - Tighter θ (e.g. 0.1) = more conservative updates; looser (0.3) = faster but riskier

### KL / stability

- `actor_rollout_ref.actor.use_kl_loss=True`, `kl_loss_coef≈0.001`, `kl_loss_type=low_var_kl`
- `algorithm.use_kl_in_reward=False` (standard GRPO recipe)

---

## Target generation format (soft, not locked)

Model is prompted to emit **only JSON**, same shape as the dataset:

```json
{
  "entities": [
    {
      "text": "...",
      "type": "PERSON|ORGANIZATION|LOCATION|PRODUCT|EVENT",
      "subtype": "...",
      "score": 0.0-1.0,
      "normalized": "...",
      "evidence_text": "..."
    }
  ],
  "events": [
    {
      "actor1": "...",
      "actor2": "...",
      "other_actors": [{"text": "...", "entity_type": null}],
      "topics": [{"code": "01.1", "name": "...", "confidence": 0.9}],
      "actions": [{"code": "19", "name": "Fight", "confidence": 0.9}],
      "evidence_text": "..."
    }
  ]
}
```

No constrained grammar at decode time — **reward** teaches structure. Extra fields / missing optional fields allowed with partial credit.

---

## Reward design (`grpo/reward_ie.py`)

VERL signature:

```python
def compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float | dict:
    ...
```

Wired via:

```
custom_reward_function.path=.../reward_ie.py
custom_reward_function.name=compute_score
```

### Score components (weighted sum → scalar in [0, 1+])

| Component                  | Weight (start) | Signal                                                                 |
| -------------------------- | -------------- | ---------------------------------------------------------------------- |
| **Format**           | 0.15           | Parseable JSON; has`entities`/`events` lists                       |
| **Entity F1**        | 0.40           | Set F1 on`(norm_text, type)` vs gold (noise types HDL/PR/… ignored) |
| **Subtype accuracy** | 0.10           | Among matched entities, subtype match rate (optional soft)             |
| **Topic F1**         | 0.25           | Multi-label F1 on union of event topic codes                           |
| **Action F1**        | 0.10           | Multi-label F1 on action codes 13–20                                  |
| **Length penalty**   | −small        | Empty dump or absurd over-generation                                   |

`ground_truth` stored as JSON string of cleaned gold `{entities, events}`.

Also return dict with breakdown keys when VERL supports metric logging (`score`, `format`, `entity_f1`, `topic_f1`, …) so MLflow/console can show “is the model good” beyond a single number.

### Validation without training forever

- Hold-out val parquet; `trainer.test_freq` runs reward metrics on val rollouts.
- Periodic offline eval script: greedy/sample generate → same reward + entity/topic reports on fixed mid-band sample + full val subset.

---

## Data pipeline

**Source:** `../NER_TOPIC_Modelling/data/ner_topic_llm_dataset.jsonl` (~80.6k)

**Output:** VERL parquet rows (same contract as GSM8k example):

```python
{
  "data_source": "ner_topic_ie",
  "prompt": [
    {"role": "system", "content": SYSTEM_IE},
    {"role": "user", "content": "Title: ...\nDocument:\n...\n\nExtract entities and events. Return ONLY JSON..."}
  ],
  "ability": "ie",
  "reward_model": {"style": "rule", "ground_truth": "<json string>"},
  "extra_info": {"doc_id": ..., "split": "train", "index": ...}
}
```

**Prep steps (`grpo/prepare_parquet.py`):**

1. Clean entities (drop HDL/PR/MEN/CLM/HTG; dedupe).
2. Clean events/topics/actions (verified-ish codes).
3. Truncate long docs (`max_source_chars` ~4k–6k for 0.6B context comfort).
4. Cap huge targets or drop outliers (`max_target_chars`).
5. Split 90/5/5 by `doc_id` → `data/grpo/train.parquet`, `val.parquet`, `test.parquet`.
6. Write `stats.json` + one `example_prompt.json` / `example_gt.json`.

**Optional smoke:** `--max_docs 500` for first GRPO dry run.

---

## Cold start (strongly recommended)

0.6B base often cannot emit valid IE JSON from scratch → **sparse rewards** kill GRPO.

**Phase 0 — light SFT (optional but default-on for first real run):**

- Same prompts; completion = gold JSON.
- Short SFT (1 epoch, subset or full) with HF Trainer or VERL SFT example.
- Save adapter/checkpoint under `outputs/sft_qwen3_0.6b_ie/`.
- GRPO `MODEL_PATH` points at that checkpoint (or base if user forces zero-SFT).

Plan default: **SFT cold start → GRPO**.

---

## Layout under `NER_Training/`

```
NER_Training/
  verl/                          # already cloned (git submodule-ish local clone)
  grpo/
    prepare_parquet.py           # JSONL → VERL parquet
    reward_ie.py                 # compute_score for IE
    run_grpo_qwen3_0.6b.sh       # single-GPU L40S GRPO launch
    run_sft_coldstart.sh         # optional SFT before GRPO
    eval_rollouts.py             # offline generate + reward report
    README.md                    # how to run / knobs
  configs/
    grpo_qwen3_0.6b.yaml         # documented defaults (shell still primary for VERL)
  data/grpo/                     # train/val/test.parquet
  outputs/
    sft_qwen3_0.6b_ie/
    grpo_qwen3_0.6b_ie/          # VERL checkpoints
  scripts/
    00_prepare_grpo_data.sh
    01_sft_coldstart.sh
    02_grpo_train.sh
    03_eval.sh
```

Do **not** commit HF tokens; use `HF_TOKEN` env.
Add `verl/` to `.gitignore` only if you prefer not tracking the full clone; otherwise keep clone as dependency path (document `pip install -e ./verl`).

---

## Single-GPU GRPO launch profile (L40S)

Adapt from `verl/examples/grpo_trainer/run_qwen3_4b_fsdp.sh`, downsized:

| Knob                           | Suggested start                                                                                                             |
| ------------------------------ | --------------------------------------------------------------------------------------------------------------------------- |
| `MODEL_PATH`                 | `Qwen/Qwen3-0.6B` or SFT ckpt                                                                                             |
| `trainer.n_gpus_per_node`    | **1**                                                                                                                 |
| `trainer.nnodes`             | 1                                                                                                                           |
| `INFER_BACKEND`              | `vllm` if installable; else `hf` if VERL allows                                                                         |
| `rollout.n` (group size)     | **4** (raise to 8 if memory OK)                                                                                       |
| `train_batch_size`           | **32–64** prompts                                                                                                    |
| `ppo_mini_batch_size`        | divide total traj = batch × n                                                                                              |
| `max_prompt_length`          | 1536–2048                                                                                                                  |
| `max_response_length`        | 1024–1536                                                                                                                  |
| `actor.optim.lr`             | **1e-6**                                                                                                              |
| `clip_ratio` (θ)            | **0.2** (expose env `CLIP_RATIO`)                                                                                   |
| `clip_ratio_c`               | 3.0                                                                                                                         |
| `kl_loss_coef`               | 0.001                                                                                                                       |
| `total_epochs`               | 1–3 on full data; smoke 20–50 steps                                                                                       |
| `gpu_memory_utilization`     | 0.4–0.5 (leave room for actor+ref)                                                                                         |
| `tensor_model_parallel_size` | **1**                                                                                                                 |
| Logger                         | `console` + **mlflow** (`trainer.logger=['console','mlflow']` if supported; else console + manual MLflow wrapper) |

Checkpoint dir: `outputs/grpo_qwen3_0.6b_ie/`
`trainer.save_freq` / `test_freq` set for recovery and “is it good” checks.

---

## Implementation steps

1. **Deps**

   - Document install: `pip install -e ./verl` + VERL core deps (torch, vllm/sglang optional, ray, transformers, datasets, mlflow).
   - Pin note for transformers/vLLM compatibility with Qwen3.
2. **`grpo/prepare_parquet.py`**

   - Build parquet; unit-print one mid-doc example.
3. **`grpo/reward_ie.py`**

   - Robust JSON extract (strip markdown fences).
   - Entity/topic/action F1 + format.
   - CLI self-test: score gold-against-gold ≈ high; empty ≈ low.
4. **`grpo/run_sft_coldstart.sh` + small SFT entry**

   - Optional 1-epoch SFT on train parquet completions.
5. **`grpo/run_grpo_qwen3_0.6b.sh`**

   - Env overrides: `MODEL_PATH`, `CLIP_RATIO`, `ROLLOUT_N`, `TRAIN_BATCH_SIZE`, MLflow URI.
   - Calls `python -m verl.trainer.main_ppo` with `algorithm.adv_estimator=grpo` + `custom_reward_function.*`.
6. **`grpo/eval_rollouts.py`**

   - Load ckpt; generate on val/sample; print mean reward + entity/topic F1; dump JSON predictions next to gold.
7. **README**

   - End-to-end commands; θ clipping explanation; “how to judge model quality”; link to MLflow UI.
8. **Smoke path**

   - 200–500 docs → prepare → reward unit test → 10–20 GRPO steps on 1 GPU → confirm reward moves and checkpoints land under `outputs/`.

---

## How you’ll know the model is “good”

| Check             | Healthy signal                                                      |
| ----------------- | ------------------------------------------------------------------- |
| Format rate       | % valid JSON rises toward gold                                      |
| Mean group reward | Up over steps; std not collapsed to 0                               |
| Entity F1 (val)   | Improves vs step-0 base/SFT                                         |
| Topic micro-F1    | Improves                                                            |
| KL / clipfrac     | Not exploding; clipfrac not stuck at 1.0                            |
| Spot reads        | Mid sample like Hormuz news: sensible PERSON/LOC + topics 05.x/06.x |

If rewards stay flat: lower difficulty (SFT longer), increase `rollout.n`, soften entity matching (fuzzy / evidence overlap), or raise format weight early.

---

## Out of scope (this plan)

- IndicBERT encoder / bidirectional repurposing
- Locked BIO multi-task heads as primary path
- Multi-node / 8-GPU recipes (can scale later from same shell)
- Generative reward model (GenRM); start rule-based vs gold JSON
- Full event_hash / event_id fidelity (optional noise; reward ignores ids)

---

## Success criteria

- [ ] VERL installable from `./verl` and imports `verl.trainer.main_ppo`
- [ ] Parquet train/val built from full (or smoke) JSONL
- [ ] `reward_ie.compute_score` gold-self ≥ ~0.9; garbage ≤ ~0.15
- [ ] GRPO smoke ≥10 steps completes on L40S without OOM
- [ ] Checkpoints under `outputs/grpo_qwen3_0.6b_ie/`
- [ ] Metrics visible (console and/or MLflow `http://<INTERNAL_IP>:5000`)
- [ ] Eval script prints entity/topic quality on held-out docs

---

## Default command flow (after implementation)

```bash
cd /home/hariramanan/NTRO/NER_Training
export HF_TOKEN=...   # if needed
export MLFLOW_TRACKING_URI=http://<INTERNAL_IP>:5000

# 1 data
python grpo/prepare_parquet.py --config configs/grpo_qwen3_0.6b.yaml
# smoke: add --max_docs 500

# 2 optional cold start
bash grpo/run_sft_coldstart.sh

# 3 GRPO (theta = CLIP_RATIO)
CLIP_RATIO=0.2 ROLLOUT_N=4 bash grpo/run_grpo_qwen3_0.6b.sh

# 4 eval
python grpo/eval_rollouts.py --checkpoint outputs/grpo_qwen3_0.6b_ie/... --split val
```

# Plan: Causal GSPO IE with Qwen3-0.6B + VERL

## Goal

Train **`Qwen/Qwen3-0.6B`** as a **causal free-form JSON generator** for NER + events on `ner_topic_llm_dataset.jsonl`, using **Group Sequence Policy Optimization (GSPO)** [[arXiv:2507.18071](https://arxiv.org/abs/2507.18071)] via the local **VERL** checkout.

**Why GSPO (not plain GRPO) for this task**

For one document you sample **G JSON completions**. Some are mostly right, some wrong — you should **not** apply token-level importance clipping that over-penalizes long structured outputs. GSPO:

1. Samples a **group** of responses per prompt (same as GRPO rollouts).
2. Scores each with a **scalar reward** (entity/topic/format quality).
3. Uses **group-relative advantages**: only *worse-than-group-mean* JSONs get negative A; better ones get positive A.
4. Uses **one sequence-level importance ratio** and **one clip decision per response**, so a single bad entity in a long JSON does not explode token-level ratios.

IndicBERT-v3 remains out of scope (bidirectional).

---

## Paper math (must match)

### Sequence importance ratio (length-normalized)

\[
s_i(\theta)=\left(\frac{\pi_\theta(y_i\mid x)}{\pi_{\theta_{\mathrm{old}}}(y_i\mid x)}\right)^{1/|y_i|}
=\exp\left(\frac{1}{|y_i|}\sum_t \big[\log\pi_\theta(y_{i,t}\mid\ldots)-\log\pi_{\theta_{\mathrm{old}}}(y_{i,t}\mid\ldots)\big]\right)
\]

Only **response** tokens; pad/prompt excluded via `response_mask`.

### Group-relative advantage (same as GRPO outcome)

For group \(y_1,\ldots,y_G\) with rewards \(r_1,\ldots,r_G\):

\[
A_i=\frac{r_i-\mathrm{mean}(r)}{\mathrm{std}(r)+\varepsilon}
\]

Broadcast the **same** \(A_i\) to all tokens of response \(i\). Do **not** recompute groups inside the policy-loss microbatch.

### Objective (maximize)

\[
J=\mathbb{E}\left[\frac{1}{G}\sum_i \min\big(s_i A_i,\ \mathrm{clip}(s_i,1-\varepsilon_{\mathrm{low}},1+\varepsilon_{\mathrm{high}})\,A_i\big)\right]
\]

VERL minimizes loss ⇒ return **negative** of that.

### GRPO vs GSPO

|                   | GRPO                 | GSPO                        |
| ----------------- | -------------------- | --------------------------- |
| Ratio             | per-token\(w_{i,t}\) | per-response\(s_i\)         |
| Clip              | per token            | **once per sequence** |
| Fits long JSON IE | Noisy / unstable     | Preferred                   |

Paper experimental clips for GSPO are **tiny**: \(\varepsilon_{\mathrm{low}}=3\times10^{-4}\), \(\varepsilon_{\mathrm{high}}=4\times10^{-4}\) — **not** GRPO’s 0.2. Configurable; do not hard-swap.

**GSPO-token** (paper § variant): numerical ratio = \(s_i\), gradient via stop-grad construction. Ordinary GSPO is the default target; VERL’s registered `"gspo"` uses the stop-grad token form that is gradient-equivalent under `seq-mean-token-mean` (verify with unit tests).

---

## Local VERL inspection (authoritative)

| Item             | Finding                                                                                                                                    |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| Clone            | `/home/hariramanan/NTRO/NER_Training/verl`                                                                                               |
| Commit           | `b75f0d0`                                                                                                                                |
| GSPO loss        | **Already present**: `@register_policy_loss("gspo")` → `compute_policy_loss_gspo` in `verl/trainer/ppo/core_algos.py` (~1545) |
| Example          | `examples/gspo_trainer/run_qwen3_8b_fsdp.sh`                                                                                             |
| Enable           | `actor_rollout_ref.actor.policy_loss.loss_mode=gspo`                                                                                     |
| Agg              | `loss_agg_mode=seq-mean-token-mean` (required)                                                                                           |
| Adv              | `algorithm.adv_estimator=grpo` (group-relative outcome adv)                                                                              |
| Clips in example | `clip_ratio_low=3e-4`, `clip_ratio_high=4e-4`                                                                                          |

### Existing GSPO implementation (summary)

```text
seq_log_ratio = sum((log_π − log_π_old) * mask) / |y|
# stop-grad construction (GSPO-token style, paper-aligned gradient with seq-mean-token-mean):
log_s_t = (log_π − sg(log_π)) + sg(seq_log_ratio)
s_t = exp(log_s_t)          # forward value = s_i for all t
L = max(−A*s, −A*clip(s))   # ≡ −min(A*s, A*clip(s))
agg = seq-mean-token-mean
```

### Verify-before-patch checklist (do **not** duplicate `gspo`)

Before any core_algos edit:

1. Forward: \(s=\exp(\mathrm{mean\_valid}(\log\pi-\log\pi_{\mathrm{old}}))\).
2. Clip is on **sequence** scalar (broadcast), not independent per-token ratios.
3. Autograd: \(\partial L/\partial\log\pi_t \propto -A\cdot s/|y|\) on the unclipped branch.
4. Padding invariance; length-normalization equality for same mean log-ratio.
5. Negative \(A\) clip semantics (`max` of two loss terms).

**Patch policy (minimal):**

- If tests pass vs paper → **reuse VERL GSPO as-is**; only add IE reward + data + single-GPU launch + metrics polish.
- If mismatch found → **minimal fix** inside `compute_policy_loss_gspo` only; leave `vanilla` / GRPO untouched; add unit tests under `grpo/tests/` or VERL tests path.

Do **not** invent a second `"gspo"` registry name unless forking a variant (e.g. `"gspo_strict_scalar"`).

---

## Full product stack (IE-specific)

### Model / hardware

- Model: `Qwen/Qwen3-0.6B` (causal, ungated)
- GPU: 1× L40S 49GB
- Optional cold-start SFT → then GSPO (recommended so rewards are not sparse)

### Data → VERL parquet

Source: `../NER_TOPIC_Modelling/data/ner_topic_llm_dataset.jsonl`

Row shape (GSM8k-compatible):

```python
{
  "data_source": "ner_topic_ie",
  "prompt": [
    {"role": "system", "content": SYSTEM_IE},
    {"role": "user", "content": "Document: ...\nReturn ONLY JSON with entities and events."}
  ],
  "ability": "ie",
  "reward_model": {"style": "rule", "ground_truth": "<gold JSON string>"},
  "extra_info": {"doc_id": ..., "split": "train"}
}
```

Gold = cleaned `{entities, events}` (drop noise types HDL/PR/…; keep subtypes when present). Soft schema — free generation; reward teaches structure.

### Reward (`grpo/reward_ie.py`)

```python
def compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float | dict:
```

| Term      | Weight | Meaning                       |
| --------- | ------ | ----------------------------- |
| Format    | 0.15   | Valid JSON + keys             |
| Entity F1 | 0.40   | `(norm_text, type)` vs gold |
| Subtype   | 0.10   | Soft credit on matches        |
| Topic F1  | 0.25   | Union of topic codes          |
| Action F1 | 0.10   | CAMEO 13–20                  |

Scalar reward feeds GRPO-style **group** \(A_i\): mediocre JSONs near group mean get ~0 advantage (little update); only relative winners/losers drive learning — matches “not everything needs penalisation.”

### GSPO launch knobs (1× L40S)

From `examples/gspo_trainer/run_qwen3_8b_fsdp.sh`, downsized:

```bash
algorithm.adv_estimator=grpo
actor_rollout_ref.actor.policy_loss.loss_mode=gspo
actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean
actor_rollout_ref.actor.clip_ratio_low=3e-4    # θ-low (paper-scale)
actor_rollout_ref.actor.clip_ratio_high=4e-4  # θ-high
actor_rollout_ref.rollout.n=4                 # group size G (8 if VRAM allows)
trainer.n_gpus_per_node=1
actor_rollout_ref.rollout.tensor_model_parallel_size=1
MODEL_PATH=Qwen/Qwen3-0.6B   # or SFT ckpt
custom_reward_function.path=.../reward_ie.py
custom_reward_function.name=compute_score
```

Expose env: `CLIP_RATIO_LOW`, `CLIP_RATIO_HIGH`, `ROLLOUT_N`, `TRAIN_BATCH_SIZE`, `MODEL_PATH`.

### MLflow

`http://<INTERNAL_IP>:5000`, experiment `gspo-ie-qwen3-0.6b` (console logger + mlflow if VERL build supports it).

### Checkpoints

`outputs/gspo_qwen3_0.6b_ie/` — VERL trainer `save_freq`.

---

## Layout

```
NER_Training/
  verl/                              # existing clone (b75f0d0)
  grpo/                              # keep name; algorithm is GSPO
    prepare_parquet.py
    reward_ie.py
    verify_gspo_math.py              # unit tests vs paper eqs
    run_sft_coldstart.sh             # optional
    run_gspo_qwen3_0.6b.sh           # main launch
    eval_rollouts.py
    README.md
  configs/gspo_qwen3_0.6b.yaml
  data/grpo/                         # train/val/test.parquet
  outputs/sft_qwen3_0.6b_ie/
  outputs/gspo_qwen3_0.6b_ie/
  scripts/00_prepare_*.sh … 03_eval.sh
```

Optional **minimal** VERL patch only if verification fails:

- File: `verl/verl/trainer/ppo/core_algos.py` (`compute_policy_loss_gspo`)
- Tests: `grpo/verify_gspo_math.py` (and/or `verl/tests/...`)
- Preserve GRPO/vanilla registries

---

## Implementation order

1. **GSPO audit + unit tests** (`verify_gspo_math.py`) against paper eqs (5)–(7); document pass/fail; patch only if needed.
2. **`prepare_parquet.py`** from IE JSONL; smoke `--max_docs 500`.
3. **`reward_ie.py`** + gold-self / garbage self-tests.
4. **SFT cold-start script** (1 epoch light) → better nonzero rewards.
5. **`run_gspo_qwen3_0.6b.sh`** single-GPU; wire custom reward + paper-scale clips + `loss_mode=gspo`.
6. **`eval_rollouts.py`** + MLflow metrics (mean reward, entity F1, topic F1, format rate, sequence clipfrac).
7. **README**: GSPO vs GRPO for multi-JSON IE; how group rewards avoid blanket penalisation; commands.

---

## How “multiple JSONs, some wrong” is handled

```
Doc x
  ├─ y1 JSON  reward 0.82  → A > 0  (reinforce sequence)
  ├─ y2 JSON  reward 0.80  → A ≈ 0  (near mean, weak update)
  ├─ y3 JSON  reward 0.21  → A < 0  (discourage sequence)
  └─ y4 JSON  reward 0.55  → A < 0 or small
```

- **No token-level clip** chopping mid-JSON because one field drifted.
- **Sequence clip** (\(3\mathrm{e}{-4}/4\mathrm{e}{-4}\)) stabilizes whole-response updates.
- Only **relative** quality within the group matters → wrong generations are not absolute death sentences if the whole group is weak (shared baseline).

---

## Success criteria

- [ ] GSPO math verification documented (existing VERL ≈ paper, or minimal patch + tests green)
- [ ] Parquet train/val from IE dataset
- [ ] Reward: gold-self high, empty low
- [ ] GSPO smoke ≥10 steps on L40S, `loss_mode=gspo`
- [ ] Checkpoints under `outputs/gspo_qwen3_0.6b_ie/`
- [ ] Val: format rate + entity/topic F1 reported
- [ ] GRPO token-level path left unchanged

---

## Default commands (post-implementation)

```bash
cd /home/hariramanan/NTRO/NER_Training
export MLFLOW_TRACKING_URI=http://<INTERNAL_IP>:5000
export HF_TOKEN=...   # if needed

python grpo/verify_gspo_math.py
python grpo/prepare_parquet.py --max_docs 500   # then full
bash grpo/run_sft_coldstart.sh                  # recommended
CLIP_RATIO_LOW=3e-4 CLIP_RATIO_HIGH=4e-4 ROLLOUT_N=4 \
  bash grpo/run_gspo_qwen3_0.6b.sh
python grpo/eval_rollouts.py --checkpoint outputs/gspo_qwen3_0.6b_ie/...
```

---

## Out of scope

- IndicBERT causal repurpose
- Locked BIO multi-task as primary path
- Implementing GSPO-token as a separate mode unless requested
- Changing VERL rollout/logprob infrastructure / Routing Replay
- Multi-node 8-GPU recipes (scale later from same shell)
