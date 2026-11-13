# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

GSPO (Group Sequence Policy Optimization) fine-tuning for causal, free-form information
extraction: given a document, generate a JSON object with `entities` and `events` keys directly
(no BIO tagging, no constrained decoding). Training uses **VERL** (vendored locally under `verl/`)
for the RL loop and **MLflow** for tracking.

Two model paths exist:
- **Qwen3.5-0.8B (current target)** — `grpo/run_gspo_qwen3_5_0.8b.sh`, wrapper
  `scripts/04_gspo_train_qwen3_5.sh`. Validated end-to-end (LoRA + GSPO + vLLM rollout). Requires a
  pre-built text-only checkpoint, see below.
- **Qwen3-0.6B (original)** — `grpo/run_gspo_qwen3_0.6b.sh`, wrapper `scripts/02_gspo_train.sh`.
  All infra fixes applied but not re-verified end-to-end since the switch.

Source labels live outside this repo at
`../NER_TOPIC_Modelling/data/ner_topic_llm_dataset.jsonl` (~80.6k docs). This repo consumes that
file; it does not produce it.

Read `handover/HANDOVER.md`, `handover/STATUS.md`, and `handover/NEXT_STEPS.md` before starting
work — they hold the point-in-time status (what's blocked, what's been tried) from the last
session and are more current than this file for day-to-day state. Update them (not just this
file) when you change status, hit a new blocker, or unblock something.

## Environment

```bash
source /home/hariramanan/NTRO/Entity_extraction/.venv/bin/activate
export MLFLOW_TRACKING_URI=http://<INTERNAL_IP>:5000
export MLFLOW_EXPERIMENT_NAME=gspo-ie-qwen3-0.6b
cd /home/hariramanan/NTRO/NER_Training
export PYTHONPATH="${PWD}/verl:${PWD}:${PYTHONPATH:-}"
```

There is no dedicated venv for this repo — it shares `Entity_extraction/.venv`. Installing VERL
editable (`pip install -e ./verl`) can pull `transformers` to 5.x, which may conflict with other
projects sharing that venv; be deliberate about upgrades here.

MLflow UI: http://<INTERNAL_IP>:5000, experiment `gspo-ie-qwen3-0.6b`. VERL reads
`MLFLOW_TRACKING_URI` directly (see `verl/verl/utils/tracking.py`); the `grpo/` scripts also read
it via `grpo/mlflow_utils.py`.

Never commit `HF_TOKEN` or other secrets — use env vars / `.env` (see `.env.example`). Qwen3-0.6B
itself is ungated, so a token usually isn't needed for the model.

## Commands

One-time setup:

```bash
pip install -r requirements.txt
pip install -e ./verl   # needs ray, etc.
```

Full pipeline, in order:

```bash
bash scripts/00_prepare_grpo_data.sh          # JSONL -> data/grpo/{train,val,test}.parquet + MLflow stats
bash scripts/01_sft_coldstart.sh              # optional but recommended SFT cold-start before RL
bash scripts/02_gspo_train.sh                 # GSPO RL (Qwen3-0.6B) via VERL, logger=console+mlflow
bash scripts/03_eval.sh --max_samples 30      # offline eval + MLflow (auto-picks best checkpoint)
```

Qwen3.5-0.8B path (current target):

```bash
python grpo/build_qwen3_5_text_only.py        # one-time: text-only checkpoint (REQUIRED, see below)
LORA_RANK=16 bash scripts/04_gspo_train_qwen3_5.sh
bash scripts/03_eval.sh --max_samples 30
```

`scripts/03_eval.sh` auto-selects the newest *merged* GSPO checkpoint, else the Qwen3.5 base, else
the Qwen3-0.6B SFT checkpoint; override with `MODEL_PATH=`/`DATA=`. VERL writes **LoRA-only**
checkpoints that `eval_rollouts.py` cannot load directly — merge first:

```bash
python -m verl.model_merger merge --backend fsdp \
  --local_dir outputs/gspo_qwen3_5_0.8b_ie/global_step_N/actor \
  --target_dir outputs/gspo_qwen3_5_0.8b_ie/global_step_N_merged
```

Each `scripts/0N_*.sh` is a thin wrapper (activates venv, sets `MLFLOW_TRACKING_URI`, `cd`s to
repo root) around the underlying `grpo/*.py` / `grpo/*.sh` — pass flags straight through, e.g.
`bash scripts/00_prepare_grpo_data.sh --max_docs 500`.

Smoke-test variants (fast, small, for verifying wiring before a full run):

```bash
bash scripts/00_prepare_grpo_data.sh --max_docs 500
bash scripts/01_sft_coldstart.sh --max_samples 200 --max_steps 20
MAX_STEPS=20 ROLLOUT_N=2 TRAIN_BATCH_SIZE=4 bash scripts/02_gspo_train.sh
```

Self-checks (no GPU training required):

```bash
PYTHONPATH=verl python grpo/verify_gspo_math.py   # GSPO sequence-ratio math vs paper formula
python grpo/reward_ie.py                          # reward function sanity checks
```

`grpo/run_gspo_qwen3_0.6b.sh` (invoked by `scripts/02_gspo_train.sh`) is entirely controlled via
env vars — key ones: `MODEL_PATH`, `ROLLOUT_N`, `CLIP_RATIO_LOW`/`CLIP_RATIO_HIGH`,
`TRAIN_BATCH_SIZE`, `PPO_MINI_BATCH_SIZE`, `MAX_PROMPT_LENGTH`/`MAX_RESPONSE_LENGTH`,
`INFER_BACKEND` (`vllm` if importable, else `hf`), `TOTAL_EPOCHS`, `MAX_STEPS` (caps
`trainer.total_training_steps` for smoke runs). It auto-prefers
`outputs/sft_qwen3_0.6b_ie/final` as `MODEL_PATH` when that directory exists.

## Architecture

### Why GSPO, not token-level GRPO clipping

For one document, sample **G** completions (`rollout_n`, default 4). Rewards are group-relative:
only better-than-group-mean completions get pushed up. The policy loss uses a **sequence-level**
importance ratio with a single clip decision per sequence, rather than per-token GRPO clipping —
this avoids clipping chaos on long JSON outputs. Paper: arXiv:2507.18071. Clip range is
paper-scale (`clip_ratio_low=3e-4`, `clip_ratio_high=4e-4`), **not** GRPO's typical `0.2`. In VERL
config terms this is `algorithm.adv_estimator=grpo` (group-relative advantages) combined with
`actor_rollout_ref.actor.policy_loss.loss_mode=gspo` (sequence-level ratio/clip), aggregated with
`loss_agg_mode=seq-mean-token-mean`. The GSPO loss implementation itself lives in VERL
(`verl/verl/trainer/ppo/core_algos.py::compute_policy_loss_gspo`) — don't reimplement it; extend
`grpo/verify_gspo_math.py` if you suspect a paper mismatch.

### Pipeline stages and where they live

1. **`grpo/prepare_parquet.py`** — reads the external NER/topic JSONL, cleans entities (drops
   noise types `HDL/PR/MEN/CLM/HTG`), cleans topics/events, renders the fixed `SYSTEM_IE` prompt +
   gold JSON target, writes VERL-format parquet to `data/grpo/{train,val,test}.parquet`, logs
   stats to MLflow. `NOISE_TYPES` is shared with `grpo/reward_ie.py` so prep and reward filtering
   stay consistent.
2. **`grpo/run_sft_coldstart.py`** — optional supervised warm-start on the gold JSON before RL;
   writes to `outputs/sft_qwen3_0.6b_ie/final` (plus `sft_meta.json`). GSPO scripts auto-detect
   and prefer this checkpoint as `MODEL_PATH` if present, else fall back to `Qwen/Qwen3-0.6B`.
3. **`grpo/run_gspo_qwen3_0.6b.sh`** — launches `python -m verl.trainer.main_ppo` with the GSPO
   flags above, custom reward, and MLflow logger. Writes checkpoints to
   `outputs/gspo_qwen3_0.6b_ie/`. Logs a `launch_*` MLflow run with hyperparams *before* invoking
   VERL, so launch metadata survives even if the VERL run itself fails to log.
4. **`grpo/eval_rollouts.py`** — offline generation + reward scoring against `data/grpo/val.parquet`,
   logs predictions/metrics to MLflow.
5. **`grpo/reward_ie.py`** — VERL custom reward, signature
   `compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float`.
   Weighted sum: format 0.15 + entity F1 0.40 + subtype 0.10 + topic F1 0.25 + action F1 0.10
   (`WEIGHTS` dict). `_extract_json` tolerates markdown code fences and picks the outermost
   `{...}` if strict JSON parsing fails — the model is not constrained to valid JSON at decode
   time, so this recovery path matters. It additionally **repairs output truncated mid-object**
   (`_repair_truncated_json`): without that, a response cut off by the token budget scored 0.0 —
   identical to complete garbage — so RL got no gradient distinguishing "nearly right but cut off"
   from "nonsense". Empirically 9/9 val generations were truncated, and repair recovers a mean
   score of 0.24 with real entity F1 (0.07–0.50) that was previously discarded. Repaired output is
   capped at `format=0.5` so valid JSON still strictly wins, and the returned metrics carry a
   `repaired` flag. Run `python grpo/reward_ie.py` for the self-tests covering these cases.
   Passed into VERL via
   `reward.custom_reward_function.path=grpo/reward_ie.py`, `reward.custom_reward_function.name=compute_score`
   (this vendored VERL commit moved these under a nested `reward.*` config; the old top-level
   `custom_reward_function.*` keys are silently ignored — see `handover/STATUS.md`).
6. **`grpo/mlflow_utils.py`** — every stage's MLflow plumbing (`setup_mlflow`, `mlflow_run`
   context manager, `log_metrics`, `log_artifact`, `log_dict_as_json`). Default tracking URI and
   experiment name are hardcoded here as the fallback if env vars are unset.

### Qwen3.5-0.8B: text-only checkpoint is mandatory

`Qwen/Qwen3.5-0.8B` is a **multimodal hybrid** (`Qwen3_5ForConditionalGeneration`): 18 of its 24
decoder layers are Gated DeltaNet (linear attention), 6 are standard full attention (layers
3/7/11/15/19/23, per `full_attention_interval=4`), plus a vision tower and an MTP head this project
never uses. You cannot point `MODEL_PATH` at the hub repo — run
`python grpo/build_qwen3_5_text_only.py` to produce `outputs/qwen3_5_0.8b_text_only/`, which does
two things that both matter:

1. Resolves the nested `text_config` and stamps `architectures: ["Qwen3_5ForCausalLM"]`. VERL picks
   its auto-class from the config (`verl/verl/utils/model.py::get_hf_auto_model_class`), and the
   composite `Qwen3_5Config` is registered under `AutoModelForImageTextToText` — leave it alone and
   you load the whole VLM.
2. Strips a `language_model.` weight-key segment. transformers serializes this class as
   `model.language_model.layers.N…`, but vLLM's native `Qwen3_5ForCausalLM` expects
   `model.layers.N…`. Skip this and the FSDP actor loads fine while the vLLM rollout server dies
   with `ValueError: There is no module or parameter named 'language_model' in Qwen3_5Model`.

Both the HF actor and vLLM read the same on-disk directory (vLLM is launched as `serve <path>` with
no config-override hook), so this has to be fixed in the checkpoint, not via a flag.

### Two constraints that fail silently — do not "optimize" them away

- **`use_remove_padding=False` is mandatory for Qwen3.5** unless `flash-linear-attention` is
  installed. VERL's Qwen3.5 monkey-patch (`verl/verl/models/transformers/qwen3_5.py`) decides
  whether the chunked delta-rule kernel honours `cu_seqlens` with a `**kwargs`-presence heuristic
  (`_call_accepts_kwarg`), which wrongly returns True for transformers'
  `torch_chunk_gated_delta_rule` fallback — that fallback swallows `cu_seqlens` and never resets
  DeltaNet's recurrent state at packed-sequence boundaries. With padding-free packing on, state
  **leaks across samples with no error**: a silent correctness bug, not a crash. To re-enable, install
  `flash-linear-attention` + `causal-conv1d` and verify packed vs. per-sample GDN outputs match.
- **`rollout.layered_summon=False` on a single GPU** (both launch scripts force this). It is a
  multi-GPU optimization; with `NO_SHARD` FSDP it collects nothing, falls back to
  `summon_full_params(offload_to_cpu=True)`, and raises
  `NotImplementedError: offload_to_cpu=True and NO_SHARD is not supported yet` during LoRA weight sync.

### Prompt/reward schema alignment (`--schema full|minimal`)

`reward_ie.py` only ever reads **entity `text`/`type`/`subtype`** and **event `topics[].code` /
`actions[].code`**. Every other field the full prompt requests (`score`, `normalized`,
`evidence_text`, `resolution_method`, `actor1/actor2/other_actors`) earns exactly zero reward while
consuming response-token budget — measured at ~74% of a mean gold target (1257 → 332 tokens). The
overflow then truncates the JSON before it closes, which used to zero the reward outright.

`prepare_parquet.py --schema minimal` emits only the scored fields and uses a matching prompt.
Verified **reward-equivalent** to `full` across all 100 sample docs and all four pred/gold schema
combinations, and it takes gold targets exceeding a 2048-token budget from 5.4% of docs to 0%.
Since the upstream source JSONL is currently missing, `grpo/reschema_parquet.py` converts an
already-prepared split in place (`*.minimal.parquet`; `--in_place` keeps a `.full-schema.bak`).
`scripts/04_gspo_train_qwen3_5.sh` prefers `*.minimal.parquet` automatically when present.

`char_start`/`char_end` (entities and events) and event `confidence` are null in 100% of source
labels and are now **never** emitted in either schema.

### Target JSON schema (what the model must generate)

```json
{
  "entities": [{"text": "...", "type": "PERSON|ORGANIZATION|LOCATION|PRODUCT|EVENT", "subtype": "...", "score": 0.0, "normalized": "...", "evidence_text": "..."}],
  "events": [{"actor1": "...", "actor2": "...", "other_actors": [...], "topics": ["01.1"], "actions": ["13"], "evidence_text": "..."}]
}
```

Notes on the source data that shape this schema:
- `PERSON` subtypes include military ranks (`general`, `colonel`, `captain`, `soldier_trooper`,
  …) — there is no single `army_officer` literal.
- Event `topics` are CAMEO-style codes (`NN.N`, e.g. `01.1`); `actions` are CAMEO action codes
  (`13`–`20`).
- `char_start`/`char_end` are null in the source labels — the generative path doesn't use spans.
- Example shapes: `data/grpo/example_gt_pretty.json`, `data/grpo/example_prompt.json`,
  `data/sample_mid_datapoint.json`.

### Interrupting and resuming a run (local VERL patch)

Long runs are interruptible without losing progress. Upstream VERL only checkpoints on
`save_freq` multiples, the last step, or cloud-instance expiration — so killing mid-run discards
everything since the last periodic save. A local patch in
`verl/verl/trainer/ppo/ray_trainer.py` (search `LOCAL PATCH`) adds two control channels, polled
once per training step:

```bash
bash scripts/train_ctl.sh status   # running? which step is saved? pending requests?
bash scripts/train_ctl.sh save     # checkpoint after the current step, keep training
bash scripts/train_ctl.sh stop     # checkpoint after the current step, then exit cleanly
bash scripts/train_ctl.sh kill     # hard kill; loses everything since the last checkpoint
```

Mechanism: `save`/`stop` drop a `SAVE_NOW`/`STOP_NOW` sentinel file in the run's
`trainer.default_local_dir`. Files are the primary channel **because `fit()` runs inside a Ray
actor — a different process from the `python -m verl.trainer.main_ppo` launcher — so a signal sent
to the launcher never reaches the training loop.** `SIGUSR1` (save) / `SIGUSR2` (stop) sent
directly to the actor process work too. Each request fires once (the file is deleted when
consumed) and is honored at the *end* of the step in flight, so allow ~1 step for the checkpoint
to appear. The forced save deliberately bypasses the `save_freq > 0` guard, so it works even with
periodic saving disabled.

**Resuming needs no special flag.** `trainer.resume_mode=auto` is VERL's default and reads
`latest_checkpointed_iteration.txt` from `default_local_dir` — relaunch the same command and it
continues from the saved step. This cuts both ways: it is why a "fresh" run reusing an existing
checkpoint dir will silently resume old weights instead of starting clean. **Point
`CHECKPOINT_DIR` at a new directory when you intend to start from scratch.**

To evaluate any saved checkpoint (including while training continues — merging only reads):

```bash
bash scripts/05_infer_latest.sh --max_samples 100   # newest checkpoint
STEP=50 bash scripts/05_infer_latest.sh             # a specific step
```

This merges the LoRA-only checkpoint into a standalone HF model first, which `eval_rollouts.py`
requires (see the `model_merger` note under Commands).

### `verl/`

Vendored upstream VERL clone, installed editable (`pip install -e ./verl`). Treat it as
third-party: prefer config flags over patching it, and check `verl/CLAUDE.md` /
`verl/examples/gspo_trainer/` before modifying anything inside it. `verl/CLAUDE.md`'s
contribution rules are written for upstream PRs to `verl-project/verl` and don't apply to how you
work in `NER_Training` itself.

Local patches applied (grep `LOCAL PATCH` / the noted symbol to find them):
- `verl/verl/trainer/ppo/ray_trainer.py` — on-demand checkpoint + graceful stop (above).
- `verl/verl/utils/attention_utils.py` — falls back to VERL's own pure-PyTorch
  `npu_flash_attn_utils` reimplementation when `flash_attn` isn't installed (it is not, here).

### Legacy / inactive path

`old_methods/src/` (IndicBERT-based bidirectional multi-task NER, encoder + classification heads)
and `configs/default.yaml`/`configs/causal.yaml` are earlier abandoned approaches, kept for
reference only. **Do not** revive IndicBERT-as-causal-generator or a locked BIO/classification
head as the primary path without an explicit user request — bidirectional encoders are the wrong
tool for `.generate()`-based RL rollouts, which is exactly why the project moved to Qwen3 + VERL.

## Known constraints / gotchas

- GSPO training can fail to load the model if FlashAttention2 gets toggled on without being
  installed. Check `handover/STATUS.md` / `handover/NEXT_STEPS.md` for the current fix in
  progress (forcing `attn_implementation=sdpa`, or installing `flash-attn`).
- `trainer.use_v1=False` is set deliberately in `grpo/run_gspo_qwen3_0.6b.sh` because VERL's v1
  trainer needs a `transfer_queue` package that isn't on PyPI — don't "fix" this by flipping it
  back without resolving that dependency first.
- Rollout backend auto-selects `vllm` if importable, else falls back to `hf` (slower). Force with
  `INFER_BACKEND=hf|vllm`.
- Hydra scatters per-run output dirs under `outputs/<date>/<time>/` for every VERL launch
  (including failed ones) — these are ephemeral run artifacts, not checkpoints; the real
  checkpoints are under `outputs/gspo_qwen3_0.6b_ie/` and `outputs/sft_qwen3_0.6b_ie/`.
