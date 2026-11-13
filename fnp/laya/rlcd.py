"""Stage B: RLCD fine-tune with a KL anchor to the stage-A checkpoint.

`laya.train` supports `loss="rlcd"` natively (Gaussian-noise REINFORCE on logits, proper-scoring
reward, soft-CE term) but has NO KL/anchor term, no custom reward and no per-item reward target.
This module is a minimal custom loop on top of the same model/collate/forward helpers:

  policy      reported distribution q = softmax(logits + eps),  eps ~ N(0, sigma^2) projected on the
              option simplex, G samples per item
  reward      r = log-score(q, t) + w_sph * spherical(q, t)      (strictly proper; t = reward target:
              one-hot gold where available, else Gemma consensus)
              - lam * q[INCONCLUSIVE]  for verdict items with strong evidence (n_eff>=2 &
              ds_uncertainty<0.3) whose reward target puts < 0.5 on INCONCLUSIVE
  baseline    group mean over the G samples of the same item; advantage std-normalised per batch
  loss        -E[adv * log N(eps)] + ce_coef * softCE(logits, soft target) + kl_coef * KL(anchor || pi)
  anchor      stage-A logits are precomputed once (no second model on the GPU)
"""
from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from fnp.schema import LABELS

from .train import (build_items, calibrate_save_card, freeze_embeddings, free_cuda, gpu_free_gb,
                    is_oom, next_version, oom_ladder, plan_resources, _count)

INC = LABELS.index("INCONCLUSIVE")
_MASKED = -1e4


@dataclass
class RLConfig:
    epochs: int = 2
    micro_batch: Optional[int] = None
    grad_accum: int = 8
    max_len: Optional[int] = None
    encoder_lr: float = 1e-5
    head_lr: float = 5e-5
    group: int = 8              # G noisy samples per item (group-mean baseline)
    sigma_start: float = 0.4
    sigma_end: float = 0.1
    w_sph: float = 0.75
    lam_inconclusive: float = 0.5   # assumed
    kl_coef: float = 0.1            # assumed
    ce_coef: float = 0.25           # assumed
    grad_clip: float = 1.0
    seed: int = 0
    device: str = "auto"
    freeze_emb: bool = True
    target_error: float = 0.10
    log_every: int = 25


def rlcd_anchor_loss(logits, soft_target, reward_target, mask, qtype, inc_pen, anchor_logits, *,
                     sigma: float, cfg: RLConfig):
    """Returns (loss, parts). All tensors [N,K] / [N]; `inc_pen` is 1.0 for strong-evidence verdict items."""
    import torch
    from laya.common import proper_reward
    from laya.train import soft_ce_loss

    fmask = mask.float()
    k = fmask.sum(-1, keepdim=True)
    G = cfg.group
    eps = torch.randn((G,) + logits.shape, device=logits.device) * sigma * fmask
    eps = (eps - eps.sum(-1, keepdim=True) / k) * fmask
    z = logits.detach().unsqueeze(0) + eps
    q = torch.softmax(z.masked_fill(~mask, _MASKED), -1)
    with torch.no_grad():
        r = proper_reward(q, reward_target.unsqueeze(0), qtype, fmask, w_sph=cfg.w_sph)      # [G,N]
        if q.shape[-1] > INC:      # a batch of only noul items has K=2
            r = r - cfg.lam_inconclusive * inc_pen.unsqueeze(0) * q[..., INC]
        adv = r - r.mean(0, keepdim=True)                                                     # group-mean baseline
        adv = adv / (adv.std() + 1e-6)
    logp = -((eps ** 2) * fmask).sum(-1) / (2 * sigma ** 2)
    pg = -(adv * logp).mean()
    ce = soft_ce_loss(logits, soft_target, mask)
    logp_pi = torch.log_softmax(logits.masked_fill(~mask, _MASKED), -1)
    p_a = torch.softmax(anchor_logits.masked_fill(~mask, _MASKED), -1)
    kl = (p_a * (torch.log(p_a.clamp_min(1e-9)) - logp_pi) * fmask).sum(-1).mean()
    loss = pg + cfg.ce_coef * ce + cfg.kl_coef * kl
    return loss, {"pg": float(pg), "ce": float(ce), "kl": float(kl), "reward": float(r.mean())}


def _anchor_logits(model, tok, items, dev, max_len, head_max_len, parallel):
    from laya.train import calibration_records
    recs = calibration_records(model, tok, items, dev, max_len, head_max_len, batch_size=4, parallel=parallel)
    return [np.asarray(r[1], dtype=np.float32) for r in recs]


def _run_stage_b_once(rows: dict, init_dir: Path, out_dir: Path, cfg: RLConfig, plan: dict) -> dict:
    import torch
    from laya.train import (_forward, calibration_records, collate_items, encode_item, evaluate_records,
                            load_checkpoint, resolve_device, sigma_at, uses_parallel_layout)

    dev = resolve_device(plan["device"])
    model, tok, mcfg = load_checkpoint(str(init_dir))
    head_max_len = mcfg.get("head_max_len", 192)
    max_len = min(plan["max_len"], mcfg.get("max_len", plan["max_len"]))
    parallel = uses_parallel_layout(mcfg)
    tr, skip = build_items(tok, rows["train"], max_len, head_max_len)
    ca, _ = build_items(tok, rows["calib"], max_len, head_max_len)
    ev, _ = build_items(tok, rows.get("eval", []), max_len, head_max_len)
    if not tr:
        raise ValueError(f"no training items (skipped {skip})")
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model.to(dev).eval()
    anchors = _anchor_logits(model, tok, tr, dev, max_len, head_max_len, parallel)
    for it, a in zip(tr, anchors):
        it["anchor"] = a
    erecs0 = calibration_records(model, tok, [dict(i, target=i["rtarget"]) for i in ev], dev, max_len,
                                 head_max_len, batch_size=4, parallel=parallel) if ev else []

    n_frozen = freeze_embeddings(model) if cfg.freeze_emb else 0
    amp = bool(plan["amp"]) and dev.type == "cuda"
    if hasattr(model.encoder, "gradient_checkpointing_enable") and amp:
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.head_checkpointing = amp
    model.train()
    enc = [p for n, p in model.named_parameters() if n.startswith("encoder.") and p.requires_grad]
    head = [p for n, p in model.named_parameters() if not n.startswith("encoder.") and p.requires_grad]
    opt = torch.optim.AdamW([{"params": enc, "lr": cfg.encoder_lr}, {"params": head, "lr": cfg.head_lr}],
                            weight_decay=0.01)
    mb = plan["micro_batch"]
    updates = math.ceil(math.ceil(len(tr) / mb) / cfg.grad_accum) * cfg.epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, updates), eta_min=1e-6)
    torch.manual_seed(cfg.seed)
    params = enc + head
    hist, t0, n_micro = [], time.time(), 0
    for epoch in range(cfg.epochs):
        order = list(tr)
        random.Random(cfg.seed + epoch).shuffle(order)
        sigma = sigma_at(epoch, cfg.epochs, cfg.sigma_start, cfg.sigma_end)
        agg = {"pg": 0.0, "ce": 0.0, "kl": 0.0, "reward": 0.0}
        opt.zero_grad(set_to_none=True)
        n_steps = 0
        for s in range(0, len(order), mb):
            chunk = order[s:s + mb]
            enc_items = [encode_item(tok, it, max_len, head_max_len, parallel=parallel) for it in chunk]
            batch = collate_items([enc_items], tok.pad_token_id)
            logits = _forward(model, batch, dev, amp, False)
            mask = batch["marker_mask"].to(dev)
            kmax = mask.shape[1]
            soft = batch["target"].to(dev)
            rt = torch.zeros_like(soft)
            an = torch.zeros_like(soft)
            pen = torch.zeros(len(chunk), device=dev)
            for i, it in enumerate(chunk):
                rt[i, :it["k"]] = torch.tensor(it["rtarget"], device=dev)
                an[i, :it["k"]] = torch.tensor(it["anchor"], device=dev)
                if it["qid"] == "verdict" and it["strong"] and it["rtarget"][INC] < 0.5:
                    pen[i] = 1.0
            loss, parts = rlcd_anchor_loss(logits, soft, rt, mask, batch["qtype"].to(dev), pen, an,
                                           sigma=sigma, cfg=cfg)
            (loss / cfg.grad_accum).backward()
            n_steps += 1
            n_micro += 1
            for k_ in agg:
                agg[k_] += parts[k_]
            if n_steps % cfg.grad_accum == 0 or s + mb >= len(order):
                torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
            if cfg.log_every and n_steps % cfg.log_every == 0:
                print(f"[stage B] epoch {epoch + 1} step {n_steps} loss {loss.item():.4f} {parts}", flush=True)
        hist.append({k_: v / max(1, n_steps) for k_, v in agg.items()})
        print(f"[stage B] epoch {epoch + 1}/{cfg.epochs} {hist[-1]}", flush=True)
    wall = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9 if dev.type == "cuda" else None
    from .train import verdict_metrics
    card = {
        "stage": "B", "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "parent": str(init_dir),
        "config": asdict(cfg), "plan": plan, "frozen_params": n_frozen,
        "counts": {"train_rows": len(rows["train"]), "calib_rows": len(rows["calib"]),
                   "eval_rows": len(rows.get("eval", [])), "train_items": len(tr), "calib_items": len(ca),
                   "eval_items": len(ev), "skipped": skip,
                   "train_by_src": _count([r.get("src") for r in rows["train"]])},
        "epoch_stats": hist, "wall_s": round(wall, 1),
        "step_time_s_per_microbatch": round(wall / max(1, n_micro), 3),
        "gpu_peak_alloc_gb": None if peak is None else round(peak, 3),
        "eval_parent_uncalibrated": evaluate_records(erecs0) if erecs0 else None,
        "eval_parent_verdict_uncalibrated": verdict_metrics(erecs0) if erecs0 else None,
        "assumptions": {"lam_inconclusive": cfg.lam_inconclusive, "kl_coef": cfg.kl_coef, "ce_coef": cfg.ce_coef},
    }
    mcfg = dict(mcfg)
    mcfg.pop("temperature_by_options", None)       # stage-A temperatures no longer apply; refit below
    mcfg["temperature"] = [1.0, 1.0, 1.0]
    (mcfg.get("training") or {}).pop("abstention_thresholds", None)
    card = calibrate_save_card(model, tok, mcfg, ca, ev, Path(out_dir), dev=dev, max_len=max_len,
                               head_max_len=head_max_len, card=card, target_error=cfg.target_error,
                               parallel=parallel)
    del model, opt
    free_cuda()
    return card


def train_stage_b(rows: dict, init_dir, out_dir, cfg: Optional[RLConfig] = None) -> dict:
    cfg = cfg or RLConfig()
    plan = plan_resources(cfg.device, cfg.micro_batch, cfg.max_len)
    ladder = oom_ladder(plan)
    last = None
    for i, p in enumerate(ladder):
        try:
            print(f"[stage B] plan {p}", flush=True)
            card = _run_stage_b_once(rows, Path(init_dir), Path(out_dir), cfg, p)
            card["oom_retries"] = i
            Path(out_dir, "training_card.json").write_text(json.dumps(card, indent=2, default=str))
            return card
        except Exception as e:
            if not is_oom(e):
                raise
            print(f"[stage B] OOM with {p}; retrying cheaper", flush=True)
            last = e
            free_cuda()
    raise RuntimeError("RL stage does not fit even on CPU") from last
