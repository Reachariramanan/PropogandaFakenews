# STATUS — point-in-time

**Updated:** 2026-08-11 (Claude Code — Qwen3.5-0.8B + LoRA + GSPO pipeline validated end-to-end)

## Green

- [x] VERL cloned to `NER_Training/verl`
- [x] GSPO present in VERL (`loss_mode=gspo`)
- [x] Reward function + unit tests
- [x] Parquet prep pipeline + MLflow log
- [x] Smoke parquet: train 167 / val 9 / test 10 (from 200 source docs; 14 skipped long)
- [x] SFT cold-start smoke → `outputs/sft_qwen3_0.6b_ie/final`
- [x] MLflow experiment `gspo-ie-qwen3-0.6b` (id 27) receiving runs — was soft-deleted at some
  point outside this session; restored via `MlflowClient.restore_experiment('27')`. If runs stop
  landing again, check `lifecycle_stage` first before assuming a code bug.
- [x] Launch scripts force `trainer.use_v1=False` (avoid missing `transfer_queue`)
- [x] **Qwen3-0.6B path**: FlashAttention2 fixed (`+actor_rollout_ref.model.override_config.attn_implementation=sdpa`),
  `vllm` installed (unblocks the rollout registry gap — see NEXT_STEPS.md history), reward dispatch
  moved to `reward.custom_reward_function.*` (nested config in this VERL commit), `layered_summon`
  forced `False` on single-GPU (see below). Not re-verified end-to-end after the pivot to
  Qwen3.5-0.8B — the fixes are in `grpo/run_gspo_qwen3_0.6b.sh` but the last full re-run was
  superseded by the model switch.
- [x] **Qwen3.5-0.8B + LoRA + GSPO pipeline — fully validated, smoke AND real run.**
  - Built a self-contained **text-only** checkpoint at `outputs/qwen3_5_0.8b_text_only/`
    (`architectures: ["Qwen3_5ForCausalLM"]`, 0.752B params, no vision tower, no MTP weights).
    Two build steps required, both now scripted in `grpo/run_gspo_qwen3_5_0.8b.sh`'s header
    comment: (1) load via `AutoConfig.from_pretrained(...).get_text_config()` +
    `AutoModelForCausalLM.from_pretrained(repo, config=text_config)`, then `save_pretrained()`;
    (2) **post-process the saved `model.safetensors` to strip a `language_model.` key segment**
    (`model.language_model.layers.X...` → `model.layers.X...`) — transformers' own canonical
    serialization for this class doesn't match what vLLM's native `Qwen3_5ForCausalLM` weight
    loader expects (vLLM has no `language_model` wrapper for the standalone class; only
    `Qwen3_5ForConditionalGeneration` does). Confirmed both transformers and vLLM load the
    corrected checkpoint correctly.
  - `actor_rollout_ref.model.use_remove_padding` **forced `False`** for this model — see Red/watch
    below, this is load-bearing, not a style choice.
  - LoRA scoped to the final decoder block only (layer 23 of 24 — confirmed via
    `config.json`'s `layer_types` to be a standard full-attention layer, not Gated DeltaNet, so
    plain `q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj` naming applies). Target regex
    lives in `grpo/run_gspo_qwen3_5_0.8b.sh`'s `LORA_TARGET_MODULES` default. Hydra needs the
    regex wrapped in **literal embedded quotes** (`target_modules='...'` as part of the argv
    string, not just shell-level quoting) — parens/pipes aren't valid in Hydra's bare-value
    grammar.
  - `actor_rollout_ref.rollout.layered_summon` **forced `False`** — this is a multi-GPU memory
    optimization (gather sharded FSDP params one unit at a time); on a single GPU FSDP uses
    `NO_SHARD`, and the mechanism's fallback path (`summon_full_params(offload_to_cpu=True)`)
    raises `NotImplementedError: offload_to_cpu=True and NO_SHARD is not supported yet`. Applies to
    **both** launch scripts now, not just the Qwen3.5 one.
  - 3-step smoke run: clean pass, real reward computed, checkpoint saved, script exited 0.
  - Real run: `TRAIN_BATCH_SIZE=16, ROLLOUT_N=4, TOTAL_EPOCHS=3` — auto-resumed from the smoke
    checkpoint (`global_step_3`) since both runs share `CHECKPOINT_DIR`; completed steps 4-14
    before the 1800s budget I set ran out (steps cost ~115-150s each with padding, not the ~25s
    smoke-scale steps — see Red/watch). `global_step_10` is the latest **saved** checkpoint
    (`SAVE_FREQ=10`).
  - Merged the LoRA-only `global_step_10` checkpoint into a standalone HF model via
    `python -m verl.model_merger merge --backend fsdp --local_dir .../global_step_10/actor
    --target_dir .../global_step_10_merged` — loads cleanly as plain `Qwen3_5ForCausalLM`.
  - Evaluated both the untrained baseline and the merged step-10 checkpoint against
    `data/grpo/val.parquet` (9 samples, greedy) via `grpo/eval_rollouts.py`. **Both score 0.0 on
    every metric** — see Amber/watch, this is an expected/diagnosed outcome, not a pipeline bug.

## Red / blocked

- [ ] **Full source dataset missing.** `../NER_TOPIC_Modelling/data/ner_topic_llm_dataset.jsonl`
  (~80.6k docs) still doesn't exist — only a 100-line `ner_topic_llm_dataset_sample.jsonl` remains
  in that project. `prepare_parquet.py`'s default `--source` still points at the missing path; a
  full (non-`--max_docs`) data prep will fail immediately. External to `NER_Training` — needs the
  `NER_TOPIC_Modelling` owner to restore/regenerate it. **This is the actual ceiling on training
  scale right now**, not anything in this repo's pipeline.
- [ ] `transfer_queue` not installable from PyPI (blocked V1 trainer — unaffected, `use_v1=False`
  workaround stands)

## Amber / watch

- **`use_remove_padding=True` is UNSAFE for Qwen3.5 without `flash-linear-attention` installed.**
  VERL's own Qwen3.5 monkey-patch (`verl/verl/models/transformers/qwen3_5.py`) threads `cu_seqlens`
  into `torch_chunk_gated_delta_rule` (the pure-PyTorch fallback used when `flash-linear-attention`
  isn't installed), but that fallback silently ignores `cu_seqlens` instead of resetting Gated
  DeltaNet's recurrent state at packed-sequence boundaries — confirmed via direct code reading, not
  just inferred. This is a **silent correctness bug** (contaminated training signal), not a crash,
  so it wouldn't be caught by just running the pipeline and checking exit codes. Fixed today by
  forcing `use_remove_padding=False` (padded batches, no packing). To safely re-enable it for
  speed: install `flash-linear-attention` + `causal-conv1d`, then verify packed-vs-unpadded GDN
  outputs match per-sample before trusting it.
- **Real-run step cost is dominated by padding waste.** With `use_remove_padding=False`, a step
  with `TRAIN_BATCH_SIZE=16, ROLLOUT_N=4` (64 generated sequences/step, padded to
  `max_response_length=2048`) took ~115-150s — roughly 5-6x the smoke-scale (`batch=4, rollout=2`)
  step cost, not the ~2x you'd expect from batch size alone. `timing_s/update_actor` dominates
  (~80-90s/step) since every micro-batch pays for full-length padding regardless of actual content
  length. Installing `flash-linear-attention` (see above) would let `use_remove_padding=True` pack
  sequences and cut this substantially — worth doing before any larger-scale run.
- **Eval score is 0.0 for both baseline and trained checkpoint — diagnosed, not a bug.** Root
  cause: the model's un-tuned output style lists ~9 fields per entity (text/type/subtype/score/
  normalized/evidence_text/...), which for documents with many entities runs past the
  `max_new_tokens=2048` budget before the JSON closes — `Unterminated string` on `json.loads`,
  `reward_ie.py`'s `_extract_json` can't recover a truncated object, format score (and everything
  gated on parseable JSON) comes out 0. The trained checkpoint's step-10 LoRA delta hasn't shifted
  greedy-decode output measurably from baseline (verified: sample 0 predictions are essentially
  identical) — expected given only 11 real training steps at GSPO's deliberately microscopic clip
  range (3e-4/4e-4, designed for large-scale runs) on a single decoder layer. Training's own
  sampling-based validation reward (`critic/score/mean`) did vary 0.08-0.18 across steps, so the
  reward signal itself is real and responsive — greedy eval at this training scale just isn't where
  it'd show up yet. **Not fixable by more debugging** — needs either (a) the full 80.6k-doc
  dataset + many more steps, or (b) a real SFT cold-start on Qwen3.5 first (current
  `sft_qwen3_0.6b_ie/final` is for the *other* model and irrelevant here), or (c) tightening the
  reward/prompt to reduce per-entity field verbosity so outputs fit the response budget.
- **`grpo/eval_rollouts.py` had a real bug, now fixed**: `row["prompt"]` comes back from parquet as
  a `numpy.ndarray` of dicts, not a JSON string or plain list — the script only handled the
  JSON-string case, so Qwen3.5's Jinja chat template hit `ValueError: The truth value of an array
  with more than one element is ambiguous` on the raw ndarray. Fixed with a `.tolist()` branch.
  This would have silently been fine for models whose templates happen not to trip on this, so
  it's worth keeping an eye out for similar array-vs-list assumptions elsewhere in this script.
- transformers upgraded by VERL install (5.x, confirmed 5.10.4) — may break other packages in same
  venv; also now confirmed to include native `qwen3_5`/`qwen3_5_moe` support, no dev-branch install
  needed despite the model card's README suggesting otherwise (that guidance was accurate only at
  the model's original release time)
- SFT checkpoint at `outputs/sft_qwen3_0.6b_ie/final` is a **12-step smoke artifact** for
  **Qwen3-0.6B** — not applicable to the Qwen3.5-0.8B path at all
- Hydra outputs litter `outputs/2026-08-10/` from failed GSPO starts (Qwen3-0.6B era, harmless)
- Installing `vllm` this session also bumped `numpy` (2.4.4→2.3.5) and `pydantic` (2.10.3→2.13.4)
  in the shared `Entity_extraction` venv, which pip flags as incompatible with the sibling
  `entity-extraction` package's exact pins. Basic import of that package still succeeds; full
  functional impact on the `Entity_extraction` project has not been tested — flag if anything looks
  off over there.

## Artifacts

| Path | Meaning |
|------|---------|
| `data/grpo/*.parquet` | Smoke RL data (167/9/10 rows) |
| `outputs/sft_qwen3_0.6b_ie/final` | Qwen3-0.6B SFT smoke checkpoint (not used by the Qwen3.5 path) |
| `outputs/qwen3_5_0.8b_text_only/` | Base Qwen3.5-0.8B, text-only extraction, corrected weight keys — the `MODEL_PATH` for `run_gspo_qwen3_5_0.8b.sh` |
| `outputs/gspo_qwen3_0.6b_ie/` | Qwen3-0.6B GSPO checkpoints (empty — never completed a run) |
| `outputs/gspo_qwen3_5_0.8b_ie/global_step_10/` | Qwen3.5-0.8B LoRA checkpoint, VERL's native (LoRA-only + FSDP) format |
| `outputs/gspo_qwen3_5_0.8b_ie/global_step_10_merged/` | Same, merged into a standalone loadable HF model |
| `outputs/eval_rollouts/qwen3_5_baseline/`, `.../qwen3_5_trained_step10/` | Eval predictions + summary for both checkpoints |
| MLflow exp 27 | Tracking UI — restored this session after being found soft-deleted |
