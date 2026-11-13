# NEXT STEPS (Claude) — ordered

## Done (2026-08-12, later session) — data unblocked, reward crash fixed, runs now interruptible

**The P0 external blocker below is RESOLVED.** The full dataset exists locally at
`data/ner_topic_llm_dataset.jsonl` (80,642 docs, 710 MB). Prepared with the minimal schema into
`data/grpo_full/`: **72,574 train / 4,031 val / 4,033 test** (4 docs skipped, no labels; 0 empty,
0 over-length). Verified: prompt matches `SYSTEM_IE_MINIMAL`, entity keys are exactly
text/type/subtype, `extra_info.answer == ground_truth`, gold-self reward 1.0 on 2,000 sampled rows.
Only 2.2% of prompts exceed the 1536 prompt budget (auto-filtered) and **0%** of gold targets
exceed the 2048 response budget. Also wrote `data/grpo_full/val_small.parquet` (200 rows) —
in-training validation against the full 4,031-row val split would cost >1 h per checkpoint.

**Reward function crashed a live training run — fixed.** A real run died at step 15 with
`TypeError: 'bool' object is not iterable` from `_action_codes`. Root cause: fields were read as
`ev.get("actions") or []`, which only guards *falsy* values; the model emitted `"actions": true`,
and `True or []` → `True`, so the loop tried to iterate a bool. The exception propagated out of
`compute_score` → `RewardLoopWorker` → Ray and **killed the whole run over one malformed rollout
out of ~480**. Fixes in `grpo/reward_ie.py`:
- `_as_list()` at all 14 field-read sites — wrong-typed values score as empty (a lone scalar/dict
  is read as a one-element list). Format credit still requires a real `list`, so correct output
  stays strictly better.
- `_score_safe()` wraps scoring so *any* unforeseen shape returns 0.0 with a warning instead of
  raising. A reward fed model-generated text must be a total function; the space of strings an LLM
  can emit is not enumerable in advance.
- Verified: prior scores byte-identical, gold-self still 1.0 over 2,000 full-data rows,
  **0 exceptions in 4,000 fuzzed mutations**, 11 malformed shapes added to the self-tests.

**Runs are now interruptible and resumable** (local patch, `LOCAL PATCH` in
`verl/verl/trainer/ppo/ray_trainer.py`; see CLAUDE.md "Interrupting and resuming a run"):
`bash scripts/train_ctl.sh {status|save|stop|kill}` and `bash scripts/05_infer_latest.sh` to merge
+ evaluate any checkpoint. Control is via `SAVE_NOW`/`STOP_NOW` sentinel files in the checkpoint
dir (plus SIGUSR1/SIGUSR2) — **files are primary because `fit()` runs in a Ray actor, a different
process from the launcher, so signalling the launcher does nothing.** Unit-tested all six paths
(save, stop, one-shot consumption, both signals, missing-dir no-op). **Not yet tested live on GPU**
— the machine's root-owned vLLM server reclaimed 42 GB of 48 GB right after our run stopped.

Also fixed earlier in the session: SFT cold-start had **no prompt masking** (`labels = input_ids`),
so ~70% of supervised tokens (up to 91%) were the system prompt + document — the cold-start was
mostly learning to reproduce its own input. Now masks the prompt via a token-level common-prefix
boundary, with a guard against an all-`-100` row producing NaN loss. And `reschema_parquet.py` left
a stale full-schema `extra_info.answer` in converted splits (the SFT target fallback), which would
have trained minimal prompts against verbose targets.

### Baseline to beat
**0.1934** mean reward on `val_small.parquet` (200 rows), from the clean base checkpoint with the
fixed reward. The old "0.0 on everything" evals were artifacts of the two reward bugs, not model
quality.

### State of training
Nothing is trained. A full-data run reached step 6/400 before being stopped on request; no
checkpoint was written (upstream saves only at `save_freq` multiples — which is exactly why the
control patch above now exists). `outputs/gspo_qwen3_5_0.8b_full/` is empty.
**Do not start training without an explicit request.**

### When training resumes
1. Use `CHECKPOINT_DIR` pointing at a **new** dir for a clean start — `resume_mode=auto` silently
   resumes any existing checkpoint in the dir (this already caused one run to unknowingly continue
   pre-fix LoRA weights from `global_step_10`).
2. Set `SAVE_FREQ` low (~10) early on, so an interruption isn't a total loss.
3. Check the GPU first — the root-owned vLLM server contends for it and will cause an OOM.
4. Sizing measured at batch 16 + `rollout_n=4`: **~110 s/step**, ~85–105k tokens/step, actor peak
   ~25 GB. 400 steps ≈ 12 h and covers only ~9% of one epoch over 72.5k rows.

## Done (2026-08-12) — code fixes, no training run

Root-caused the 0.0 eval scores to **two code defects, not infrastructure**:

1. **`reward_ie.py` had a zero-gradient cliff.** Output truncated by the token budget failed to
   parse, so it scored 0.0 — *identical to complete garbage*. In RL that means no gradient
   separating "nearly correct but cut off" from "nonsense". Verified **9/9** val generations were
   truncated. Added `_repair_truncated_json` (rewinds to the last cleanly-closed element and closes
   open containers); re-scoring the real predictions recovers mean **0.000 → 0.239** with genuine
   entity F1 (0.07–0.50) that was being discarded. Repaired output is capped at `format=0.5` so
   valid JSON still strictly wins, and metrics now carry a `repaired` flag. Self-tests extended to
   cover truncated / unsalvageable / already-valid cases.
2. **Prompt and reward disagreed about the schema.** `SYSTEM_IE` demanded 9 entity fields + event
   `confidence`, but the reward only ever reads entity `text`/`type`/`subtype` and event
   `topics[].code`/`actions[].code` — **~74% of every generation was unscorable filler** (1257 → 332
   mean tokens) that then overflowed the budget and zeroed the reward. Added
   `prepare_parquet.py --schema {full,minimal}` with a matching prompt, proven **reward-equivalent**
   over all 100 sample docs × 4 pred/gold combinations. Gold targets exceeding a 2048-token budget
   drop from 5.4% of docs to 0%. Also stopped emitting `char_start`/`char_end` (null in 100% of 1899
   entities / 451 events) and event `confidence` in both schemas.

Supporting fixes:
- `grpo/reschema_parquet.py` (new) — converts an already-prepared split to the minimal schema, since
  the source JSONL is missing and re-prep isn't possible. Ran non-destructively:
  `data/grpo/*.minimal.parquet` now exist (originals untouched), 0 reward mismatches.
- `grpo/build_qwen3_5_text_only.py` (new) — the text-only checkpoint build was previously a
  header comment; the weight-key remap step is easy to omit and fails deep inside vLLM startup.
  Verified it reproduces the working checkpoint **bit-for-bit** (320/320 keys, 0 tensors differing).
- `eval_rollouts.py` — `--max_new_tokens` default 1024 → 2048 (the old default was *below* the mean
  gold target, guaranteeing truncation); `subtype` was computed but silently dropped from reporting;
  added `hit_token_cap` / `truncated_repaired` / `unparseable` diagnostics plus a warning, so
  "budget-bound" is no longer indistinguishable from "bad extraction".
- `grpo/verify_gspo_math.py` — **this advertised self-check had been silently broken** by VERL API
  drift (`ActorConfig` gained required fields and became a frozen dataclass, and signalled via
  `AssertionError`, which the existing `except TypeError` missed). It crashed instead of verifying.
  Now constructs the config correctly and the VERL GSPO check **passes**.
- `scripts/04_gspo_train_qwen3_5.sh` (new) wrapper; `scripts/03_eval.sh` rewritten to auto-select
  the newest merged checkpoint and matching data schema instead of hardcoding Qwen3-0.6B paths.
- `prepare_parquet.py` now fails fast with a clear message on the dead default `--source`.
- `CLAUDE.md` updated: both model paths, the mandatory text-only checkpoint, the two silent-failure
  constraints, and the schema-alignment section.

All self-checks pass (`python grpo/reward_ie.py`, `PYTHONPATH=verl python grpo/verify_gspo_math.py`);
every launch script passes `bash -n`; both schemas verified end-to-end through real data prep.
**Nothing was trained** — next run should start from the minimal-schema data to get the benefit.

## Done (previous session, 2026-08-11)

- FA2 blocker, rollout-registry gap (`vllm` installed), reward-dispatch nesting, `layered_summon`
  single-GPU incompatibility — all fixed in `grpo/run_gspo_qwen3_0.6b.sh` (see STATUS.md Green).
- Pivoted the active model target to **Qwen3.5-0.8B** (text-only extraction) per explicit user
  request, despite the larger unresolved-risk profile flagged earlier. Built and validated a
  working GSPO+LoRA pipeline for it end-to-end: smoke run (3 steps) and a real run (11 further
  steps, checkpoint saved and merged, evaluated). See STATUS.md for the full list of Qwen3.5-specific
  fixes (text-only checkpoint construction + weight-key remap, `use_remove_padding=False`,
  LoRA scoped to layer 23, `layered_summon=False`).
- Both evaluated checkpoints (baseline and trained) score 0.0 on val — diagnosed as an expected
  outcome of the training scale (11 steps, 1 layer, GSPO's microscopic clip range), not a pipeline
  defect. Full diagnosis in STATUS.md Amber/watch.

## P0 — RESOLVED 2026-08-12 (was: missing source dataset)

~~`../NER_TOPIC_Modelling/data/ner_topic_llm_dataset.jsonl` (~80.6k docs) doesn't exist.~~
The dataset is now available **inside this repo** at `data/ner_topic_llm_dataset.jsonl` (80,642
docs) and is prepared into `data/grpo_full/`. Note the path differs from the old default in
`grpo/prepare_parquet.py --source`, so pass `--source data/ner_topic_llm_dataset.jsonl` explicitly.
The remaining ceiling on real training is GPU availability, not data.

## P1 — If continuing on Qwen3.5-0.8B once full data is available

1. **Install `flash-linear-attention` + `causal-conv1d`**, then re-enable
   `actor_rollout_ref.model.use_remove_padding=True` in `grpo/run_gspo_qwen3_5_0.8b.sh` — this
   should meaningfully cut the ~115-150s/step cost seen in the real run (padding waste dominates
   `update_actor` time). **Before trusting it**, write a small script that runs a packed batch and
   an unpadded/per-sample batch through the same GDN layer and confirms outputs match — the
   correctness bug we found (`cu_seqlens` silently ignored by the torch fallback) only manifests
   without these packages, but verify the fix rather than assuming installation alone fixes it.
2. **Do a real SFT cold-start on Qwen3.5-0.8B** (`outputs/qwen3_5_0.8b_text_only/` as the base,
   not the Qwen3-0.6B SFT checkpoint which is irrelevant here) before the next GSPO run — right
   now GSPO is starting from the raw base model with no IE-specific warm-start at all.
3. **Consider trimming the reward/prompt schema's per-entity verbosity** (9 fields: text/type/
   subtype/score/normalized/evidence_text/resolution_method/char_start/char_end) or raising
   `MAX_RESPONSE_LENGTH` further — the diagnosed failure mode (JSON truncated mid-entity before
   closing) will keep suppressing reward signal regardless of training progress until addressed.
4. Full prep once data exists: `bash scripts/00_prepare_grpo_data.sh` (no `--max_docs`), pointed at
   whatever the restored source file path turns out to be.
5. Real GSPO run with a *realistic* time budget this time — the 1800s cap I used for the "real"
   run this session was based on smoke-scale step timing and undershot actual cost by ~4-5x. Size
   `timeout` (or run without one, foreground-monitored) based on measured per-step cost once
   `flash-linear-attention` is in place, not the smoke-scale number.
6. Eval with `grpo/eval_rollouts.py` (bug already fixed — numpy-array prompt handling) against
   `val.parquet` / `test.parquet`, comparing to the baseline runs already logged in MLflow
   (`eval_qwen3_5_baseline_val`, `eval_qwen3_5_lora_step10_val`) as a before/after reference point.

## P2 — If reverting to Qwen3-0.6B instead

All the general fixes (FA2, vllm, reward dispatch, `layered_summon`) are already applied to
`grpo/run_gspo_qwen3_0.6b.sh` and were smoke-tested individually, but the *combined* script was not
re-run end-to-end after the model pivot happened mid-session. Rerun the smoke command once before
trusting it for real training:
```bash
source /home/hariramanan/NTRO/Entity_extraction/.venv/bin/activate
export MLFLOW_TRACKING_URI=http://<INTERNAL_IP>:5000
cd /home/hariramanan/NTRO/NER_Training
export PYTHONPATH=$PWD/verl:$PWD
MODEL_PATH=$PWD/outputs/sft_qwen3_0.6b_ie/final \
MAX_STEPS=3 ROLLOUT_N=2 TRAIN_BATCH_SIZE=4 PPO_MINI_BATCH_SIZE=4 \
INFER_BACKEND=vllm TOTAL_EPOCHS=1 TEST_FREQ=100 \
bash grpo/run_gspo_qwen3_0.6b.sh 2>&1 | tee /tmp/gspo_smoke.log
```

## P3 — Other

- Resolve V1 trainer / `transfer_queue` if team wants current VERL default path (not blocking,
  `use_v1=False` workaround stands).
- Check whether the `numpy`/`pydantic` version shift from installing `vllm` this session actually
  broke anything in the sibling `Entity_extraction` project (basic import works; not functionally
  tested beyond that).
- Log sample generations to MLflow artifacts (VERL validation already can) for qualitative review
  alongside the aggregate metrics.

## Do not

- Switch back to IndicBERT for causal GSPO without explicit user request.
- Hardcode HF tokens into git.
- Duplicate GSPO loss if VERL's is paper-correct after smoke.
- Re-enable `use_remove_padding=True` for Qwen3.5 without first installing
  `flash-linear-attention`/`causal-conv1d` and verifying packed-vs-unpadded parity — the silent
  correctness bug is real and won't announce itself with an error.
