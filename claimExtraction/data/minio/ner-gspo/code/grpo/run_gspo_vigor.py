#!/usr/bin/env python3
# Copyright (c) NER_Training. Licensed under Apache-2.0, same as the vendored VERL.
"""GSPO training driven by VIGOR variance-guided progressive rollout allocation.

What this is
------------
A drop-in replacement for ``python -m verl.trainer.main_ppo trainer.use_v1=False``
that keeps *everything* VERL does (RayPPOTrainer, FSDP actor, vLLM rollout,
GSPO sequence-level loss, GRPO group-relative advantages, checkpointing,
validation, MLflow) and swaps out exactly one thing: instead of drawing a fixed
``rollout.n`` completions for every prompt in the batch, each training step runs
the VIGOR schedule -- several *rounds* of generation against a frozen policy
snapshot, progressively concentrating the rollout budget on the prompts whose
cumulative reward variance is highest.

Why a custom loop rather than a patch to ``ray_trainer.py``
-----------------------------------------------------------
``verl/trainer/main_ppo_v0.py::TaskRunner.run`` is just::

    trainer = RayPPOTrainer(...)
    trainer.init_workers()
    trainer.fit()

All three are plain public calls. This file mirrors ``TaskRunner.run`` verbatim
up to ``init_workers()`` and then substitutes :func:`vigor_fit` for ``fit()``,
calling the trainer's already-public building blocks
(``_get_gen_batch`` / ``async_rollout_manager.generate_sequences`` /
``_compute_old_log_prob`` / ``compute_advantage`` / ``_update_actor`` /
``_validate`` / ``_save_checkpoint``). Nothing under ``verl/`` is modified.

The VIGOR algorithm
-------------------
Reimplemented here from ``rollout_allocators/rollout_allocators/vigor_allocator.py``
(read for its round/select/expand/budget structure) but operating on
``DataProto`` batches keyed by VERL ``uid`` instead of that library's ``Rollout``
dataclasses -- the two interfaces are incompatible, so nothing is imported from
it. Per training step, with ``B`` prompts:

  * round 0: every prompt is active, ``m = m0`` new rollouts each.
  * round t>0: keep ``ceil(alpha * len(active))`` prompts with the highest
    *population variance of their cumulative rewards so far* (all rounds, not
    just the last), give each survivor ``m <- round(m * gamma)`` new rollouts.
  * rollouts are **merged**, never replaced: a prompt that survives to round 3
    contributes ``m0 + m0*gamma + m0*gamma^2 + m0*gamma^3`` sequences to the
    training batch, and a prompt eliminated after round 0 still contributes its
    ``m0``.
  * budget: ``B * m0 * T`` rows, which is exactly the paper's budget-conserving
    total when ``alpha*gamma == 1`` (the default alpha=0.5 / gamma=2). Before
    each generate we shrink ``m`` -- and, if even one rollout per active prompt
    does not fit, truncate ``active`` (which is already ordered by descending
    variance) -- rather than overspend.

The resulting batch has **variable group sizes per uid**. That is fine, and is
the reason no custom advantage code is needed: VERL's
``compute_grpo_outcome_advantage`` (core_algos.py:268) groups purely via
``defaultdict(list)`` over ``non_tensor_batch["uid"]`` and never assumes a
uniform count per uid. We hand it a batch where different uids repeat different
numbers of times and let it normalize.

Known gap vs. ``fit()``
-----------------------
Profiler hooks (``global_profiler.steps``), REMAX baselines, critic/GAE,
rollout-correction / bypass mode, and distillation are intentionally not
reimplemented -- this loop targets the project's GSPO+GRPO+no-critic setup and
raises if it is pointed at a config that needs them. The on-demand
``SAVE_NOW`` / ``STOP_NOW`` control channel (the LOCAL PATCH in
``ray_trainer.py``) *is* wired in, via the trainer's own ``_poll_control()``;
if a future VERL bump drops that patch the loop logs that control is
unavailable rather than pretending it works.
"""

from __future__ import annotations

import math
import os
import socket
import uuid
from importlib import import_module
from pprint import pprint
from typing import Any

import hydra
import numpy as np
import ray
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.main_ppo import run_ppo
from verl.trainer.main_ppo_v0 import BaseTaskRunner
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.reward import extract_reward
from verl.trainer.ppo.utils import create_rl_dataset, create_rl_sampler, need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.debug import marked_timer
from verl.utils.device import auto_set_device
from verl.utils.metric import reduce_metrics
from verl.utils.skip.skip_manager import SkipManager
from verl.utils.tracking import Tracking

# ---------------------------------------------------------------------------
# VIGOR defaults. These mirror the values validated in the standalone
# rollout_allocators library; alpha*gamma == 1 is the budget-conserving regime.
# ---------------------------------------------------------------------------
DEFAULT_M0 = 2
DEFAULT_T = 4
DEFAULT_GAMMA = 2.0
DEFAULT_ALPHA = 0.5
DEFAULT_SCORE_SOURCE = "auto"
DEFAULT_SCORE_FN = "grpo.reward_ie.compute_score"


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def population_variance(values) -> float:
    """Population (ddof=0) variance, matching the standalone allocator."""
    if values is None or len(values) == 0:
        return 0.0
    return float(np.var(np.asarray(values, dtype=np.float64)))


def estimate_budget_conserving_total(b: int, m0: int, t: int, gamma: float, alpha: float) -> float:
    """C_total = B*m0 * sum_{k=0}^{T-1} (alpha*gamma)^k (paper Part 5).

    Collapses to ``B*m0*T`` in the budget-conserving regime ``alpha*gamma == 1``.
    """
    ag = alpha * gamma
    if abs(ag - 1.0) < 1e-12:
        return float(b * m0 * t)
    return b * m0 * (1.0 - ag**t) / (1.0 - ag)


def _load_callable(path: str):
    """Import ``pkg.mod.attr`` and return the attribute."""
    module_path, _, attr = path.rpartition(".")
    if not module_path:
        raise ValueError(f"vigor.score_fn must be a dotted path like 'grpo.reward_ie.compute_score', got {path!r}")
    return getattr(import_module(module_path), attr)


def choose_effective_mini_batch_size(num_rows: int, desired: int, dp_size: int) -> int:
    """Pick a mini-batch size that actually divides a variable-size VIGOR batch.

    See the long note in :func:`_update_actor_variable_groups` for why this is
    required at all. Constraints VERL imposes:

      * ``num_rows % mini_batch_size == 0`` (tensordict_utils.py:588)
      * ``mini_batch_size % dp_size == 0``  (engine_workers.py:266)

    Among the divisors of ``num_rows`` that satisfy those, return the one
    closest to ``desired`` (ties go to the larger). A floor of ``desired // 4``
    keeps a prime-ish row count from collapsing to a mini-batch of 1 -- in that
    case ``num_rows`` itself (a single full-batch update) is always a
    candidate and is far closer to the intended effective batch size.
    """
    if num_rows % dp_size != 0:
        raise ValueError(
            f"VIGOR produced {num_rows} rollout rows, which is not divisible by the actor data-parallel "
            f"size {dp_size}. VERL dispatches the batch across DP ranks by equal chunks, so this cannot "
            f"be scheduled. Choose train_batch_size / vigor.m0 / vigor.T so that the per-step budget "
            f"(train_batch_size * m0 * T) is a multiple of {dp_size}, or run single-GPU."
        )
    floor = min(num_rows, max(dp_size, desired // 4))
    candidates = [
        d
        for d in range(dp_size, num_rows + 1, dp_size)
        if num_rows % d == 0 and d >= floor
    ]
    # num_rows itself always satisfies every constraint, so this is never empty.
    return min(candidates, key=lambda d: (abs(d - desired), -d))


class _TempConfigOverride:
    """Temporarily set dotted config keys, restoring them on exit.

    Used only around ``_update_actor`` so the mini-batch size it derives matches
    the real (variable) row count. Purely driver-side: worker configs were
    already shipped at ``init_workers()`` time.
    """

    def __init__(self, config, overrides: dict[str, Any]):
        self._config = config
        self._overrides = overrides
        self._saved: dict[str, Any] = {}

    @staticmethod
    def _resolve(config, dotted: str):
        parts = dotted.split(".")
        node = config
        for p in parts[:-1]:
            node = node[p]
        return node, parts[-1]

    def __enter__(self):
        for dotted, value in self._overrides.items():
            node, leaf = self._resolve(self._config, dotted)
            self._saved[dotted] = node[leaf]
            node[leaf] = value
        return self

    def __exit__(self, *exc):
        for dotted, value in self._saved.items():
            node, leaf = self._resolve(self._config, dotted)
            node[leaf] = value
        return False


# ---------------------------------------------------------------------------
# reward scoring for the VIGOR selection step
# ---------------------------------------------------------------------------
class RolloutScorer:
    """Produces one scalar reward per rollout row, and guarantees ``rm_scores``.

    ``rm_scores`` is what ``extract_reward`` (and therefore
    ``token_level_scores`` -> ``token_level_rewards`` -> advantages) reads. The
    agent loop already fills it in using ``reward.custom_reward_function``
    (i.e. ``grpo/reward_ie.py::compute_score``) -- see
    ``verl/experimental/agent_loop/agent_loop.py:1086-1092``. When it is
    present we reuse it, which is not a shortcut but the stronger guarantee:
    the reward VIGOR ranks on is then bit-identical to the reward that trains
    the policy. When it is absent (or ``vigor.score_source=direct``) we call
    ``compute_score`` here on the driver and write ``rm_scores`` ourselves, in
    the same last-response-token layout the agent loop uses.

    The ground truth for the direct path comes from ``prompt_batch``, **not**
    from ``gen_out``. ``_get_gen_batch`` (ray_trainer.py:573) keeps
    ``reward_model`` / ``data_source`` / ``extra_info`` on the prompt-side
    DataProto, and the agent loop only merges its input non-tensors into its
    output when ``reward_loop_worker_handles is None``
    (agent_loop.py:1097). With no reward model configured,
    ``enable_agent_reward_loop = not use_rm`` is True (ray_trainer.py:953),
    those handles exist, and ``gen_out`` therefore carries **no**
    ``reward_model`` column at all -- reading it from there yields
    ``ground_truth=None`` for every row and a meaningless variance ranking.
    """

    def __init__(self, tokenizer, score_source: str, score_fn_path: str):
        if score_source not in ("auto", "direct", "rm"):
            raise ValueError(f"vigor.score_source must be one of auto|direct|rm, got {score_source!r}")
        self.tokenizer = tokenizer
        self.score_source = score_source
        self.score_fn_path = score_fn_path
        self._score_fn = None
        self.used_source: str | None = None

    @property
    def score_fn(self):
        if self._score_fn is None:
            self._score_fn = _load_callable(self.score_fn_path)
        return self._score_fn

    def score(self, gen_out: DataProto, prompt_batch: DataProto) -> np.ndarray:
        """Score ``gen_out``'s rows, using ``prompt_batch`` for the ground truth.

        ``prompt_batch`` must be the prompt-side rows this generation came from,
        aligned row-for-row with ``gen_out`` (i.e. ``batch.select_idxs(row_index)``
        for the same ``row_index`` that produced ``gen_out``).

        On the direct path this *mutates* ``gen_out.batch`` by adding
        ``rm_scores``. The caller must therefore call this **before** unioning
        ``gen_out`` into the training batch -- ``union`` copies tensors into the
        destination TensorDict, so a key added afterwards never propagates.
        """
        has_rm = gen_out.batch is not None and "rm_scores" in gen_out.batch.keys()

        if self.score_source == "rm" and not has_rm:
            raise RuntimeError(
                "vigor.score_source=rm but the rollout returned no 'rm_scores'. Set "
                "reward.custom_reward_function.path/name (see grpo/run_gspo_vigor.sh) or use "
                "vigor.score_source=direct."
            )
        if has_rm and self.score_source != "direct":
            self.used_source = "rm_scores"
            return gen_out.batch["rm_scores"].sum(dim=-1).detach().cpu().numpy().astype(np.float64)

        self.used_source = "direct"
        scores = self._score_direct(gen_out, prompt_batch)
        self._write_rm_scores(gen_out, scores)
        return scores

    @staticmethod
    def _pick_non_tensor(key: str, prompt_batch: DataProto, gen_out: DataProto):
        """Prefer the prompt-side column; fall back to the rollout output.

        The prompt side is authoritative (see the class docstring). The fallback
        only matters for configurations where the agent loop *does* echo its
        input non-tensors back (no reward loop workers).
        """
        val = prompt_batch.non_tensor_batch.get(key, None)
        if val is None:
            val = gen_out.non_tensor_batch.get(key, None)
        return val

    def _score_direct(self, gen_out: DataProto, prompt_batch: DataProto) -> np.ndarray:
        responses = gen_out.batch["responses"]
        texts = self.tokenizer.batch_decode(responses, skip_special_tokens=True)

        reward_model = self._pick_non_tensor("reward_model", prompt_batch, gen_out)
        data_source = self._pick_non_tensor("data_source", prompt_batch, gen_out)
        extra_info = self._pick_non_tensor("extra_info", prompt_batch, gen_out)

        if reward_model is None:
            raise RuntimeError(
                "vigor direct scoring found no 'reward_model' column on either the prompt batch or the "
                "rollout output, so every ground_truth would be None and the reward (and the variance "
                "ranking built on it) would be meaningless. Check that the parquet built by "
                "grpo/prepare_parquet.py carries a 'reward_model' column, or use vigor.score_source=rm."
            )
        if len(reward_model) != len(texts):
            raise RuntimeError(
                f"vigor direct scoring got {len(reward_model)} ground-truth rows for {len(texts)} "
                "generations; the prompt batch passed to score() must be row-aligned with gen_out."
            )

        out = np.zeros(len(texts), dtype=np.float64)
        for i, text in enumerate(texts):
            rm_i = reward_model[i]
            gt = rm_i.get("ground_truth") if isinstance(rm_i, dict) else None
            out[i] = float(
                self.score_fn(
                    data_source=None if data_source is None else data_source[i],
                    solution_str=text,
                    ground_truth=gt,
                    extra_info=None if extra_info is None else extra_info[i],
                )
            )
        return out

    @staticmethod
    def _write_rm_scores(gen_out: DataProto, scores: np.ndarray) -> None:
        """Mirror agent_loop._postprocess: put the scalar on the last response token."""
        response_mask = gen_out.batch["response_mask"]
        attention_mask = gen_out.batch["attention_mask"]
        prompt_length = gen_out.batch["prompts"].size(1)
        last_idx = (attention_mask[:, prompt_length:].sum(dim=1) - 1).clamp(min=0)
        rm_scores = torch.zeros_like(response_mask, dtype=torch.float32)
        rm_scores[torch.arange(response_mask.size(0)), last_idx] = torch.tensor(
            scores, dtype=torch.float32, device=rm_scores.device
        )
        gen_out.batch["rm_scores"] = rm_scores


# ---------------------------------------------------------------------------
# the VIGOR round schedule for one training step
# ---------------------------------------------------------------------------
def run_vigor_rounds(
    trainer: RayPPOTrainer,
    batch: DataProto,
    gen_batch: DataProto,
    scorer: RolloutScorer,
    m0: int,
    total_rounds: int,
    gamma: float,
    alpha: float,
    timing_raw: dict,
) -> tuple[DataProto, dict]:
    """Run up to ``total_rounds`` generate/score/select rounds on a frozen policy.

    Args:
        batch:     prompt-side DataProto, one row per prompt, ``uid`` assigned.
        gen_batch: the matching generation payload from ``trainer._get_gen_batch``.

    Returns:
        (merged_batch, diagnostics). ``merged_batch`` has one row per rollout,
        with ``uid`` repeated a *variable* number of times per prompt.
    """
    config = trainer.config
    n_prompts = len(batch)
    uids = list(batch.non_tensor_batch["uid"])

    # Budget in rollout-rows for this step. B*m0*T reproduces the paper's
    # budget-conserving total for alpha*gamma == 1 and is the ceiling used by
    # the shrink-before-generate rule below in every other case too.
    budget_target = n_prompts * m0 * total_rounds

    # Agent-loop dispatch chunks the request across workers and requires an
    # exactly divisible row count (DataProto.chunk asserts it). Round row counts
    # are arbitrary under VIGOR, so pad/unpad exactly like `_validate` does.
    size_divisor = max(1, int(config.actor_rollout_ref.rollout.agent.num_workers))

    active: list[int] = list(range(n_prompts))
    cumulative: list[list[float]] = [[] for _ in range(n_prompts)]
    collected: list[DataProto] = []
    rounds_log: list[dict] = []
    budget_used = 0
    m = m0

    for t in range(total_rounds):
        if not active or m < 1 or budget_used >= budget_target:
            break

        # Shrink-before-generate (allocator Part 7): never overspend. Shrink m
        # first; if not even one rollout per active prompt fits, truncate the
        # active list, which is already sorted by descending variance so the
        # most informative prompts survive.
        remaining = budget_target - budget_used
        if len(active) * m > remaining:
            per = remaining // len(active)
            if per >= 1:
                m = int(per)
            else:
                active = active[:remaining]
                m = 1
        if not active or m < 1:
            break

        row_index = np.repeat(np.asarray(active, dtype=np.int64), m)

        round_gen = gen_batch.select_idxs(row_index)
        # select_idxs returns the *same* meta_info object; give this round its
        # own copy so a union below cannot mutate gen_batch/batch meta_info.
        round_gen.meta_info = dict(gen_batch.meta_info)

        with marked_timer("gen", timing_raw, color="red"):
            padded, pad_size = pad_dataproto_to_divisor(round_gen, size_divisor)
            gen_out = trainer.async_rollout_manager.generate_sequences(padded)
            # ---------------------------------------------------------------
            # VIGOR SAME-POLICY-SNAPSHOT INVARIANT -- do not add a weight sync here.
            #
            # Every round of this loop must sample from one frozen policy
            # snapshot, otherwise the cumulative per-prompt reward variance that
            # drives selection would mix samples from different policies and the
            # GRPO group statistics computed later over the merged group would be
            # off-policy with respect to `old_log_probs`.
            #
            # The invariant holds structurally: actor weights reach the rollout
            # replicas through exactly one call,
            # `self.checkpoint_manager.update_weights(...)`
            # (verl/checkpoint_engine/base.py:486 -- the only path that runs
            # actor_wg.update_weights + rollout.update_weights). In `fit()` that
            # call site sits *after* `_update_actor` (ray_trainer.py ~line 1779),
            # plus once before the loop starts and inside the critic-warmup
            # branch. This loop calls it in exactly the same places -- never
            # between rounds -- so the vLLM engines hold identical weights for
            # every round of a step. `_compute_old_log_prob` likewise runs once,
            # after all rounds, against the same unmodified actor.
            #
            # `sleep_replicas()` is deliberately deferred until after the last
            # round: it frees rollout GPU memory for training, and calling it
            # between rounds would force an extra wake-up round-trip.
            # ---------------------------------------------------------------
            gen_out = unpad_dataproto(gen_out, pad_size=pad_size)
            gen_out.meta_info.pop("timing", None)

        round_batch = batch.select_idxs(row_index)
        round_batch.meta_info = dict(batch.meta_info)

        # Score BEFORE the union, and hand the scorer the prompt-side rows.
        # Both halves matter:
        #  * On the direct path the scorer writes `rm_scores` into
        #    `gen_out.batch`. `union` copies gen_out's tensors *into*
        #    round_batch's TensorDict, so a key added to gen_out after the
        #    union would never reach the training batch and `extract_reward`
        #    (verl/trainer/ppo/reward.py:164) would KeyError on step 1.
        #  * The ground truth (`reward_model`) lives on the prompt-side batch,
        #    not on gen_out -- see RolloutScorer's docstring.
        round_scores = scorer.score(gen_out, round_batch)

        round_batch = round_batch.union(gen_out)
        collected.append(round_batch)
        budget_used += len(row_index)

        # Accumulate this round's rewards onto each prompt's cumulative group.
        for pos, prompt_i in enumerate(row_index.tolist()):
            cumulative[prompt_i].append(float(round_scores[pos]))

        # Utility = population variance over the FULL cumulative reward set for
        # the prompt (allocator Parts 3/9), not just this round's rewards.
        utility = {i: population_variance(cumulative[i]) for i in active}

        rounds_log.append(
            {
                "round": t,
                "active_count": len(active),
                "m": m,
                "new_rollouts": int(len(row_index)),
                "budget_used": budget_used,
                "mean_reward": float(np.mean(round_scores)) if len(round_scores) else 0.0,
                "mean_utility": float(np.mean(list(utility.values()))) if utility else 0.0,
            }
        )

        if t == total_rounds - 1:
            break

        keep = max(1, math.ceil(alpha * len(active)))
        # Descending variance, deterministic tie-break by uid. Selection is
        # relative to the CURRENT active set only.
        active = sorted(active, key=lambda i: (-utility[i], str(uids[i])))[:keep]
        m = max(1, int(round(m * gamma)))

    if not collected:
        raise RuntimeError("VIGOR produced no rollouts for this step; check vigor.m0 / vigor.T.")

    merged = DataProto.concat(collected)

    group_sizes = np.array([len(c) for c in cumulative if c], dtype=np.float64)
    diagnostics = {
        "rounds": rounds_log,
        "budget_target": budget_target,
        "budget_used": budget_used,
        "budget_theoretical": estimate_budget_conserving_total(n_prompts, m0, total_rounds, gamma, alpha),
        "final_selected_prompts": len(active),
        "prompts_with_rollouts": int(group_sizes.size),
        "group_size_mean": float(group_sizes.mean()) if group_sizes.size else 0.0,
        "group_size_max": float(group_sizes.max()) if group_sizes.size else 0.0,
        "group_size_min": float(group_sizes.min()) if group_sizes.size else 0.0,
        "total_rows": len(merged),
        "score_source": scorer.used_source,
    }
    return merged, diagnostics


def _update_actor_variable_groups(trainer: RayPPOTrainer, batch: DataProto) -> tuple[DataProto, int]:
    """``trainer._update_actor`` with a mini-batch size that fits a ragged batch.

    ppo_mini_batch_size investigation (findings item "RISK")
    --------------------------------------------------------
    ``RayPPOTrainer._update_actor`` (ray_trainer.py ~1326) computes::

        ppo_mini_batch_size = actor.ppo_mini_batch_size * rollout.n

    which bakes in the "every prompt got exactly rollout.n rollouts" assumption.
    Two separate questions follow.

    1. Can mini-batch splitting *silently corrupt* the GSPO objective by
       separating a prompt's rollouts across mini-batches?  **No.**
       ``compute_policy_loss_gspo`` (core_algos.py:1546) is entirely per-row:
       the sequence ratio is ``exp(mean_t(log_prob - old_log_prob))`` computed
       within each row, the clip decision is per row, and the only cross-row
       operation is ``agg_loss(..., "seq-mean-token-mean")``, which averages
       finished per-sequence losses. It never touches ``uid`` and never
       re-derives group membership from row position. Group membership is used
       exactly once, on the driver, inside
       ``compute_grpo_outcome_advantage`` (core_algos.py:268), which groups by
       the ``uid`` column via a ``defaultdict(list)`` -- before any
       mini-batching happens. Advantages are already baked into the rows by the
       time ``update_actor`` runs, so shuffling or splitting rows is safe.
       ``_balance_batch`` in ``fit()`` already relies on exactly this property.

    2. Does the stale product *break* anything?  **Yes, loudly, and it needs
       this workaround.** ``tu.make_iterator`` asserts
       ``batch_size[0] % mini_batch_size == 0``
       (verl/utils/tensordict_utils.py:588), and ``train_mini_batch`` asserts
       ``mini_batch_size % dp_size == 0``
       (verl/workers/engine_workers.py:266). A VIGOR batch has
       ``B*m0*T`` rows, which is generally *not* a multiple of
       ``ppo_mini_batch_size * rollout.n``, so the step would crash. Related,
       the same number is handed to the loss as ``global_batch_size``, the
       denominator of ``seq-mean-token-mean`` (core_algos.py:1198); a value
       that does not match the real mini-batch row count would silently
       mis-scale the gradient.

    Fix: treat ``ppo_mini_batch_size * rollout.n`` as the *desired* row count
    per mini-batch (the launch script defaults ``rollout.n`` to ``m0*T``, the
    equivalent uniform rollouts-per-prompt, so this is the same number a
    fixed-rollout_n run would use), snap it to a value that divides the actual
    row count via :func:`choose_effective_mini_batch_size`, then run
    ``_update_actor`` with ``rollout.n`` temporarily pinned to 1 and
    ``ppo_mini_batch_size`` set to that value -- so the product it recomputes is
    exactly the number we chose and the ``seq-mean-token-mean`` denominator
    stays consistent with the real mini-batch.
    """
    config = trainer.config
    num_rows = len(batch)
    desired = int(config.actor_rollout_ref.actor.ppo_mini_batch_size) * int(config.actor_rollout_ref.rollout.n)
    dp_size = max(1, int(getattr(trainer.actor_rollout_wg, "world_size", 1)))

    effective = choose_effective_mini_batch_size(num_rows, desired, dp_size)
    if effective != desired:
        print(
            f"[vigor] ppo_mini_batch_size {desired} does not divide {num_rows} ragged rollout rows; "
            f"using {effective} for this step (dp_size={dp_size})."
        )

    with _TempConfigOverride(
        config,
        {
            "actor_rollout_ref.actor.ppo_mini_batch_size": effective,
            "actor_rollout_ref.rollout.n": 1,
        },
    ):
        actor_output = trainer._update_actor(batch)
    return actor_output, effective


# ---------------------------------------------------------------------------
# the training loop
# ---------------------------------------------------------------------------
def vigor_fit(trainer: RayPPOTrainer) -> None:
    """VERL's ``RayPPOTrainer.fit()`` with the VIGOR rollout schedule."""
    config = trainer.config

    if trainer.use_critic:
        raise NotImplementedError(
            "run_gspo_vigor.py targets critic-free group-relative RL (algorithm.adv_estimator=grpo). "
            "Use the stock trainer for GAE/critic runs."
        )
    if config.algorithm.adv_estimator not in (AdvantageEstimator.GRPO, "grpo"):
        print(
            f"[vigor] WARNING: adv_estimator={config.algorithm.adv_estimator}. VIGOR's variable group "
            "sizes are only known to be handled correctly by the GRPO estimator (it groups by uid)."
        )
    # `algorithm.rollout_correction` is a composed config node
    # (verl/trainer/config/ppo_trainer.yaml:40 -> algorithm/rollout_correction.yaml),
    # so it is *never* None -- testing the node itself would warn on every run.
    # At the shipped defaults (rollout_is/rollout_rs both null) stock `fit()`'s
    # `compute_rollout_correction_and_add_to_batch` is a no-op, so there is
    # nothing to warn about. Only the two switches that actually turn it on
    # matter here.
    rollout_correction = config.algorithm.get("rollout_correction", None) or {}
    enabled_rc = [
        k for k in ("rollout_is", "rollout_rs") if rollout_correction.get(k, None) is not None
    ]
    if enabled_rc:
        print(
            f"[vigor] WARNING: algorithm.rollout_correction enables {'/'.join(enabled_rc)}, but this loop "
            "does not apply rollout correction -- those settings will be silently ignored. Use the stock "
            "trainer if you need IS/RS correction."
        )
    if getattr(trainer, "use_prefix_grouper", False) and config.trainer.balance_batch:
        # _balance_batch's PrefixGrouper path partitions *uid groups* with
        # equal_size=True (seqlen_balancing.py:611), which only yields equal
        # per-rank sample counts when every group has the same size. VIGOR
        # groups are ragged by construction, so this would silently unbalance
        # the DP dispatch. Refuse rather than corrupt the batch.
        raise NotImplementedError(
            "trainer.balance_batch with PrefixGrouper assumes uniform rollout group sizes, which VIGOR "
            "violates. Set trainer.balance_batch=False or disable the prefix grouper."
        )

    vigor_cfg = config.get("vigor", {}) or {}
    m0 = int(vigor_cfg.get("m0", DEFAULT_M0))
    total_rounds = int(vigor_cfg.get("T", DEFAULT_T))
    gamma = float(vigor_cfg.get("gamma", DEFAULT_GAMMA))
    alpha = float(vigor_cfg.get("alpha", DEFAULT_ALPHA))
    score_source = str(vigor_cfg.get("score_source", DEFAULT_SCORE_SOURCE))
    score_fn_path = str(vigor_cfg.get("score_fn", DEFAULT_SCORE_FN))

    if m0 < 1 or total_rounds < 1:
        raise ValueError(f"vigor.m0 and vigor.T must be >= 1, got m0={m0} T={total_rounds}")
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"vigor.alpha must be in (0, 1], got {alpha}")
    if gamma <= 0:
        raise ValueError(f"vigor.gamma must be > 0, got {gamma}")

    print(
        f"[vigor] schedule: m0={m0} T={total_rounds} gamma={gamma} alpha={alpha} "
        f"(alpha*gamma={alpha * gamma:.3f}; ==1 is the budget-conserving regime) "
        f"score_source={score_source}"
    )
    print(
        f"[vigor] per-step budget = train_batch_size*m0*T = "
        f"{config.data.train_batch_size}*{m0}*{total_rounds} = "
        f"{config.data.train_batch_size * m0 * total_rounds} rollout rows"
    )

    scorer = RolloutScorer(trainer.tokenizer, score_source, score_fn_path)

    logger = Tracking(
        project_name=config.trainer.project_name,
        experiment_name=config.trainer.experiment_name,
        default_backend=config.trainer.logger,
        config=OmegaConf.to_container(config, resolve=True),
    )

    if hasattr(trainer, "_dump_executor") and trainer._dump_executor._shutdown:
        trainer._init_dump_executor()

    trainer.global_steps = 0

    # LOCAL PATCH channel (scripts/train_ctl.sh). Present only while the
    # SAVE_NOW/STOP_NOW patch lives in ray_trainer.py; if a VERL bump drops it,
    # say so rather than pretending the control script works.
    has_ctl = hasattr(trainer, "_poll_control") and hasattr(trainer, "_install_signal_control")
    if has_ctl:
        trainer._install_signal_control()
    else:
        print("[vigor] NOTE: trainer has no _poll_control(); scripts/train_ctl.sh save/stop is NOT available.")

    trainer._load_checkpoint()
    trainer.checkpoint_manager.update_weights(trainer.global_steps)

    current_epoch = trainer.global_steps // len(trainer.train_dataloader)
    SkipManager.init(config)

    last_val_metrics = None
    if config.trainer.get("val_before_train", True):
        val_metrics = trainer._validate()
        assert val_metrics, f"{val_metrics=}"
        pprint(f"Initial validation metrics: {val_metrics}")
        logger.log(data=val_metrics, step=trainer.global_steps)
        if config.trainer.get("val_only", False):
            _shutdown(trainer)
            return

    progress_bar = tqdm(total=trainer.total_training_steps, initial=trainer.global_steps, desc="VIGOR Training")
    trainer.global_steps += 1
    trainer.max_steps_duration = 0
    SkipManager.set_step(trainer.global_steps)

    for epoch in range(current_epoch, config.trainer.total_epochs):
        for batch_dict in trainer.train_dataloader:
            if hasattr(trainer.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                trainer.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
            metrics: dict[str, Any] = {}
            timing_raw: dict[str, float] = {}

            batch: DataProto = DataProto.from_single_dict(batch_dict)
            batch.meta_info["temperature"] = config.actor_rollout_ref.rollout.temperature

            # Same uid convention as ray_trainer.py:1533 -- one uuid per prompt,
            # repeated across every rollout of that prompt.
            batch.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
            )

            gen_batch = trainer._get_gen_batch(batch)
            gen_batch.meta_info["global_steps"] = trainer.global_steps

            is_last_step = trainer.global_steps >= trainer.total_training_steps
            ctl_save, ctl_stop = trainer._poll_control() if has_ctl else (False, False)

            with marked_timer("step", timing_raw):
                # ---- VIGOR rounds (policy frozen for the whole block) ----
                with marked_timer("vigor_rounds", timing_raw, color="red"):
                    batch, vigor_diag = run_vigor_rounds(
                        trainer=trainer,
                        batch=batch,
                        gen_batch=gen_batch,
                        scorer=scorer,
                        m0=m0,
                        total_rounds=total_rounds,
                        gamma=gamma,
                        alpha=alpha,
                        timing_raw=timing_raw,
                    )
                    # Free rollout GPU memory now that all rounds are done --
                    # the single sleep that `fit()` does right after its one
                    # generate_sequences call.
                    trainer.checkpoint_manager.sleep_replicas()

                _print_vigor_diagnostics(trainer.global_steps, vigor_diag)
                metrics.update(_vigor_metrics(vigor_diag))

                # Fail early and clearly: every worker-group call below chunks
                # the batch across DP ranks and needs an exact split. Checking
                # here (rather than letting `_compute_old_log_prob` die with a
                # chunk assertion) points at the real cause -- the VIGOR budget.
                dp_size = max(1, int(getattr(trainer.actor_rollout_wg, "world_size", 1)))
                if len(batch) % dp_size != 0:
                    raise ValueError(
                        f"VIGOR produced {len(batch)} rollout rows at step {trainer.global_steps}, not "
                        f"divisible by the actor world size {dp_size}. Adjust data.train_batch_size / "
                        f"vigor.m0 / vigor.T so the per-step budget is a multiple of {dp_size}."
                    )

                if "response_mask" not in batch.batch.keys():
                    batch.batch["response_mask"] = compute_response_mask(batch)
                if config.trainer.balance_batch:
                    # Reorders rows. Safe: advantages are computed from the uid
                    # column, and the GSPO loss is per-row (see the long note in
                    # _update_actor_variable_groups).
                    trainer._balance_batch(batch, metrics=metrics)

                batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                images_seqlens_all: list = []
                for multi_modal_input in batch.non_tensor_batch.get("multi_modal_inputs", []):
                    if isinstance(multi_modal_input, dict) and "image_grid_thw" in multi_modal_input:
                        images_seqlens_all.extend(multi_modal_input["images_seqlens"].tolist())
                batch.meta_info["images_seqlens"] = images_seqlens_all

                with marked_timer("reward", timing_raw, color="yellow"):
                    reward_tensor, reward_extra_infos_dict = extract_reward(batch)

                with marked_timer("old_log_prob", timing_raw, color="blue"):
                    old_log_prob, old_log_prob_mfu = trainer._compute_old_log_prob(batch)
                    entropys = old_log_prob.batch["entropys"]
                    actor_config = config.actor_rollout_ref.actor
                    entropy_agg = agg_loss(
                        loss_mat=entropys,
                        loss_mask=batch.batch["response_mask"],
                        loss_agg_mode=actor_config.loss_agg_mode,
                        loss_scale_factor=actor_config.loss_scale_factor,
                    )
                    metrics.update(
                        {
                            "actor/entropy": entropy_agg.detach().item(),
                            "perf/mfu/actor_infer": old_log_prob_mfu,
                        }
                    )
                    old_log_prob.batch.pop("entropys")
                    batch = batch.union(old_log_prob)

                assert "old_log_probs" in batch.batch, f'"old_log_probs" not in {batch.batch.keys()=}'

                if trainer.use_reference_policy:
                    with marked_timer("ref", timing_raw, color="olive"):
                        batch = batch.union(trainer._compute_ref_log_prob(batch))

                with marked_timer("adv", timing_raw, color="brown"):
                    batch.batch["token_level_scores"] = reward_tensor
                    if reward_extra_infos_dict:
                        batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                    if config.algorithm.use_kl_in_reward:
                        batch, kl_metrics = apply_kl_penalty(
                            batch, kl_ctrl=trainer.kl_ctrl_in_reward, kl_penalty=config.algorithm.kl_penalty
                        )
                        metrics.update(kl_metrics)
                    else:
                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                    # Identical kwargs to the stock call site (ray_trainer.py
                    # ~1712). `num_repeat` is unused by the GRPO estimator, which
                    # derives every group purely from the uid column -- which is
                    # exactly why variable VIGOR group sizes need no custom
                    # advantage code.
                    batch = compute_advantage(
                        batch,
                        adv_estimator=config.algorithm.adv_estimator,
                        gamma=config.algorithm.gamma,
                        lam=config.algorithm.lam,
                        num_repeat=1,
                        norm_adv_by_std_in_grpo=config.algorithm.get("norm_adv_by_std_in_grpo", True),
                        config=config.algorithm,
                    )

                with marked_timer("update_actor", timing_raw, color="red"):
                    actor_output, effective_mini_bs = _update_actor_variable_groups(trainer, batch)
                metrics["vigor/effective_ppo_mini_batch_size"] = float(effective_mini_bs)

                should_save = (
                    config.trainer.save_freq > 0
                    and (is_last_step or trainer.global_steps % config.trainer.save_freq == 0)
                ) or ctl_save or ctl_stop
                if should_save:
                    if ctl_save or ctl_stop:
                        print(f"[train-ctl] force saving checkpoint at step {trainer.global_steps}")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        trainer._save_checkpoint()

                # The one and only weight sync: actor -> rollout replicas. Keeping
                # it here (and nowhere inside run_vigor_rounds) is what makes the
                # VIGOR same-policy-snapshot invariant hold.
                with marked_timer("update_weights", timing_raw, color="red"):
                    trainer.checkpoint_manager.update_weights(trainer.global_steps)

                metrics.update(reduce_metrics(actor_output.meta_info["metrics"]))

                rollout_data_dir = config.trainer.get("rollout_data_dir", None)
                if rollout_data_dir:
                    trainer._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

            if config.trainer.test_freq > 0 and (
                is_last_step or trainer.global_steps % config.trainer.test_freq == 0
            ):
                with marked_timer("testing", timing_raw, color="green"):
                    val_metrics = trainer._validate()
                    if is_last_step:
                        last_val_metrics = val_metrics
                metrics.update(val_metrics)

            trainer.max_steps_duration = max(trainer.max_steps_duration, timing_raw["step"])
            metrics.update({"training/global_step": trainer.global_steps, "training/epoch": epoch})
            metrics.update(compute_data_metrics(batch=batch, use_critic=False))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            n_gpus = trainer.resource_pool_manager.get_n_gpus()
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

            logger.log(data=metrics, step=trainer.global_steps)
            progress_bar.update(1)
            trainer.global_steps += 1
            SkipManager.set_step(trainer.global_steps)

            if is_last_step or ctl_stop:
                if ctl_stop:
                    print(
                        f"[train-ctl] stopping cleanly after step {trainer.global_steps - 1}; "
                        f"checkpoint written under {config.trainer.default_local_dir}. "
                        "Relaunch the same command to resume from here."
                    )
                if hasattr(trainer.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    trainer.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                _shutdown(trainer)
                pprint(f"Final validation metrics: {last_val_metrics}")
                progress_bar.close()
                return

            if hasattr(trainer.train_dataset, "on_batch_end"):
                trainer.train_dataset.on_batch_end(batch=batch)

    _shutdown(trainer)
    progress_bar.close()


def _shutdown(trainer: RayPPOTrainer) -> None:
    if hasattr(trainer, "_shutdown_dump_executor"):
        trainer._shutdown_dump_executor()


def _print_vigor_diagnostics(step: int, diag: dict) -> None:
    for r in diag["rounds"]:
        print(
            f"[vigor] step {step} round {r['round']}: active={r['active_count']} m={r['m']} "
            f"new_rollouts={r['new_rollouts']} budget_used={r['budget_used']} "
            f"mean_reward={r['mean_reward']:.4f} mean_var={r['mean_utility']:.5f}"
        )
    print(
        f"[vigor] step {step} summary: budget_used={diag['budget_used']}/{diag['budget_target']} "
        f"(theoretical {diag['budget_theoretical']:.1f}) rows={diag['total_rows']} "
        f"final_selected_prompts={diag['final_selected_prompts']} "
        f"prompts_with_rollouts={diag['prompts_with_rollouts']} "
        f"group_size mean/min/max={diag['group_size_mean']:.2f}/"
        f"{diag['group_size_min']:.0f}/{diag['group_size_max']:.0f} "
        f"score_source={diag['score_source']}"
    )


def _vigor_metrics(diag: dict) -> dict[str, float]:
    out: dict[str, float] = {
        "vigor/budget_used": float(diag["budget_used"]),
        "vigor/budget_target": float(diag["budget_target"]),
        "vigor/budget_utilization": (
            float(diag["budget_used"]) / float(diag["budget_target"]) if diag["budget_target"] else 0.0
        ),
        "vigor/total_rows": float(diag["total_rows"]),
        "vigor/final_selected_prompts": float(diag["final_selected_prompts"]),
        "vigor/prompts_with_rollouts": float(diag["prompts_with_rollouts"]),
        "vigor/group_size_mean": diag["group_size_mean"],
        "vigor/group_size_max": diag["group_size_max"],
        "vigor/group_size_min": diag["group_size_min"],
        "vigor/rounds_run": float(len(diag["rounds"])),
    }
    for r in diag["rounds"]:
        t = r["round"]
        out[f"vigor/round_{t}/active_count"] = float(r["active_count"])
        out[f"vigor/round_{t}/new_rollouts"] = float(r["new_rollouts"])
        out[f"vigor/round_{t}/m"] = float(r["m"])
        out[f"vigor/round_{t}/mean_reward"] = r["mean_reward"]
        out[f"vigor/round_{t}/mean_variance"] = r["mean_utility"]
    return out


# ---------------------------------------------------------------------------
# Ray task runner -- mirrors verl/trainer/main_ppo_v0.py::TaskRunner.run
# up to init_workers(), then swaps fit() for vigor_fit().
# ---------------------------------------------------------------------------
@ray.remote
class VigorTaskRunner(BaseTaskRunner):
    """Same construction path as VERL's v0 TaskRunner; different training loop."""

    def run(self, config):
        print(f"VigorTaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)
        self.add_reward_model_resource_pool(config)
        self.add_teacher_model_resource_pool(config)
        self.add_ref_policy_worker(config, actor_rollout_cls)

        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(config),
            use_critic=need_critic(config),
        )

        from verl.utils.config import omega_conf_to_dataclass
        from verl.workers.config import HFModelConfig

        model_config: HFModelConfig = omega_conf_to_dataclass(config.actor_rollout_ref.model)
        tokenizer = model_config.tokenizer
        processor = model_config.processor

        resource_pool_manager = self.init_resource_pool_mgr(config)

        from verl.utils.dataset.rl_dataset import collate_fn

        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            is_train=True,
            max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_rl_dataset(
            config.data.val_files,
            config.data,
            tokenizer,
            processor,
            is_train=False,
            max_samples=config.data.get("val_max_samples", -1),
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)

        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        trainer.init_workers()

        # <- the only divergence from TaskRunner.run: vigor_fit instead of fit
        vigor_fit(trainer)


@hydra.main(config_path="vigor_config", config_name="vigor_ppo_trainer", version_base=None)
def main(config):
    """Entry point. Accepts every override the stock GSPO launch script uses,
    plus ``vigor.m0`` / ``vigor.T`` / ``vigor.gamma`` / ``vigor.alpha``."""
    auto_set_device(config)

    if config.trainer.use_v1:
        raise ValueError(
            "run_gspo_vigor.py builds on the v0 RayPPOTrainer path. Pass trainer.use_v1=False "
            "(grpo/run_gspo_vigor.sh already does)."
        )

    validate_config(
        config=config,
        use_reference_policy=need_reference_policy(config),
        use_critic=need_critic(config),
    )
    run_ppo(config, task_runner_class=VigorTaskRunner)


if __name__ == "__main__":
    main()
