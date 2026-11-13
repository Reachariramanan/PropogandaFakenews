"""Stage A (supervised distillation) and shared plumbing for Laya fine-tuning under a small GPU budget.

Built on the primitives of `laya.train` rather than `laya.train.finetune`, because `finetune`
(a) takes its calibration slice out of the training file, and we want separate calib/eval splits,
(b) has no OOM ladder, and (c) cannot hold an anchor for stage B.

Memory plan for the 322M multilingual checkpoint (197M of it is the 250k-token embedding matrix):
embeddings are frozen (no grads / Adam state for them), activation checkpointing + autocast on,
micro-batch and max_len picked from `torch.cuda.mem_get_info()` at run time with an OOM ladder.
"""
from __future__ import annotations

import gc
import json
import math
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from fnp.config import LAYA_SUBFOLDER, MODELS_DIR
from fnp.schema import LABELS

BASE_MODEL = LAYA_SUBFOLDER   # 'multilingual' alias resolved by laya.train.resolve_checkpoint_dir


# ------------------------------------------------------------------------------ resources
def gpu_free_gb() -> float:
    import torch
    if not torch.cuda.is_available():
        return 0.0
    try:
        return torch.cuda.mem_get_info()[0] / 1e9
    except Exception:
        return 0.0


def plan_resources(device: str = "auto", micro_batch: Optional[int] = None,
                   max_len: Optional[int] = None) -> dict:
    """Pick device / micro_batch / max_len from the memory that is free right now.

    Ladder (free GB -> settings), measured on the L40S with ~6.5 GB free, see
    docs/review/03_RESOURCE_CONTENTION.md. User overrides win; CPU only when nothing fits.
    """
    free = gpu_free_gb()
    if device == "auto":
        device = "cuda" if free >= 2.5 else "cpu"
    if device == "cpu":
        plan = {"device": "cpu", "micro_batch": 2, "max_len": 512, "amp": False}
    elif free >= 12:
        plan = {"device": "cuda", "micro_batch": 8, "max_len": 768, "amp": True}
    elif free >= 5.5:
        plan = {"device": "cuda", "micro_batch": 4, "max_len": 512, "amp": True}
    elif free >= 4:
        plan = {"device": "cuda", "micro_batch": 2, "max_len": 512, "amp": True}
    else:
        plan = {"device": "cuda", "micro_batch": 1, "max_len": 384, "amp": True}
    if micro_batch:
        plan["micro_batch"] = micro_batch
    if max_len:
        plan["max_len"] = max_len
    plan["free_gb_at_plan"] = round(free, 2)
    return plan


def oom_ladder(plan: dict) -> list[dict]:
    """Progressively cheaper settings to retry with after a CUDA OOM, ending on CPU."""
    out, p = [dict(plan)], dict(plan)
    while p["device"] == "cuda" and (p["micro_batch"] > 1 or p["max_len"] > 256):
        if p["micro_batch"] > 1:
            p = dict(p, micro_batch=p["micro_batch"] // 2)
        else:
            p = dict(p, max_len=max(256, p["max_len"] - 128))
        out.append(p)
    if plan["device"] == "cuda":
        out.append({"device": "cpu", "micro_batch": 2, "max_len": min(plan["max_len"], 512), "amp": False})
    return out


def is_oom(exc: BaseException) -> bool:
    import torch
    return isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower()


def free_cuda():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def freeze_embeddings(model) -> int:
    """Freeze the (huge) token-embedding matrix; returns the number of frozen parameters."""
    n = 0
    for name, p in model.encoder.named_parameters():
        if "tok_embeddings" in name or "word_embeddings" in name:
            p.requires_grad_(False)
            n += p.numel()
    return n


# ------------------------------------------------------------------------------ items
def build_items(tok, rows: Sequence[dict], max_len: int, head_max_len: int) -> tuple[list[dict], dict]:
    """Laya training items, one per (row, question), each tagged with qid / reward target / flags.

    `item["target"]` is the soft training target (gold blended with Gemma);
    `item["rtarget"]` is the reward / evaluation target (one-hot gold, else Gemma consensus; noul
    questions use their own target). Built one question at a time so the tags stay aligned.
    """
    from laya.train import items_from_rows

    items, skipped = [], {}
    for row in rows:
        for qid, qdef in row["questions"].items():
            g = (row.get("gold") or {}).get(qid)
            if g is None:
                continue
            one = {"state": row["state"], "questions": {qid: qdef}, "gold": {qid: g}}
            its, sk = items_from_rows(tok, [one], max_len, head_max_len)
            for k, v in sk.items():
                skipped[k] = skipped.get(k, 0) + v
            for it in its:
                it["qid"], it["row_id"] = qid, row.get("id")
                it["strong"] = bool(row.get("strong"))
                it["src"] = row.get("src")
                it["rtarget"] = _reward_target(it, row, qid)
                items.append(it)
    return items, skipped


def _reward_target(item: dict, row: dict, qid: str) -> list[float]:
    if qid == "verdict" and row.get("reward_target"):
        v = np.array([float(row["reward_target"].get(l, 0.0)) for l in LABELS])
        return (v / v.sum()).tolist()
    return list(item["target"])


def with_target(items: Sequence[dict], key: str = "rtarget") -> list[dict]:
    return [dict(it, target=it[key]) for it in items]


# ------------------------------------------------------------------------------ evaluation
def _softmax(z, t=1.0):
    z = np.asarray(z, float) / t
    e = np.exp(z - z.max())
    return e / e.sum()


def verdict_metrics(records, temperature: Optional[Sequence[float]] = None,
                    temperature_by_options: Optional[dict] = None) -> dict:
    """Accuracy, macro-F1, log-loss and ECE on the 5-class verdict records only (choice, k=5)."""
    from laya.common import ece_score, temp_bucket
    ys, ps = [], []
    for qt, logits, target, k in records:
        if qt != 0 or k != len(LABELS):
            continue
        t = (temperature_by_options or {}).get(temp_bucket(qt, k)) or (temperature[qt] if temperature else 1.0)
        ps.append(_softmax(logits[:k], t))
        ys.append(int(np.argmax(target)))
    if not ys:
        return {"n": 0}
    P, y = np.array(ps), np.array(ys)
    pred = P.argmax(1)
    f1s = []
    for c in range(len(LABELS)):
        tp = int(((pred == c) & (y == c)).sum())
        fp = int(((pred == c) & (y != c)).sum())
        fn = int(((pred != c) & (y == c)).sum())
        if tp + fp + fn:
            f1s.append(2 * tp / (2 * tp + fp + fn))
    return {"n": len(y), "accuracy": float((pred == y).mean()), "macro_f1": float(np.mean(f1s)),
            "log_loss": float(-np.log(np.clip(P[np.arange(len(y)), y], 1e-9, 1)).mean()),
            "ece": float(ece_score(P.max(1), pred == y)) if len(y) >= 2 else None,
            "per_class_support": {l: int((y == i).sum()) for i, l in enumerate(LABELS)}}


# ------------------------------------------------------------------------------ pipeline
@dataclass
class StageAConfig:
    base: str = BASE_MODEL
    epochs: int = 3
    micro_batch: Optional[int] = None
    grad_accum: int = 8
    max_len: Optional[int] = None
    encoder_lr: float = 2.5e-5
    head_lr: float = 1e-4
    seed: int = 0
    device: str = "auto"
    beta: float = 0.3                 # recorded; applied when rows were built (dataset.make_row)
    freeze_emb: bool = True
    target_error: float = 0.10
    log_every: int = 50


def next_version(models_dir: Path = MODELS_DIR, prefix: str = "laya_fnp_v") -> int:
    n = [int(p.name[len(prefix):]) for p in Path(models_dir).glob(prefix + "[0-9]*")
         if p.name[len(prefix):].isdigit()]
    return max(n, default=0) + 1


def calibrate_save_card(model, tok, cfg: dict, calib_items, eval_items, out_dir: Path, *, dev,
                        max_len: int, head_max_len: int, card: dict, target_error: float = 0.10,
                        parallel: bool = False) -> dict:
    """Fit temperatures + abstention thresholds on the calib split (gold targets), score the eval
    split, save the checkpoint (laya.load-able) with questions.json and training_card.json."""
    from laya.calibrate import fit_abstention_thresholds, fit_temperature_map
    from laya.train import (calibration_records, calibration_report, evaluate_records,
                            save_checkpoint)
    from .state_builder import QUESTIONS

    model.eval()
    crecs = calibration_records(model, tok, with_target(calib_items), dev, max_len, head_max_len, parallel=parallel)
    fitted = fit_temperature_map(crecs)
    report = calibration_report(crecs, fitted["temperature"])
    try:
        thr = fit_abstention_thresholds(crecs, temperature=fitted["temperature"],
                                        temperature_by_options=fitted["temperature_by_options"],
                                        target_error=target_error, min_bucket_n=10)
    except Exception as e:                                            # too few calib items
        warnings.warn(f"abstention thresholds not fitted: {e}")
        thr = {}
    erecs = calibration_records(model, tok, with_target(eval_items), dev, max_len, head_max_len, parallel=parallel) \
        if eval_items else []
    out_cfg = dict(cfg, max_len=max_len, head_max_len=head_max_len, fine_tuned=True,
                   temperature=fitted["temperature"])
    out_cfg.pop("temperature_by_options", None)
    if fitted["temperature_by_options"]:
        out_cfg["temperature_by_options"] = fitted["temperature_by_options"]
    if thr:
        out_cfg.setdefault("training", {})["abstention_thresholds"] = thr
    out_dir = Path(out_dir)
    save_checkpoint(model, tok, out_cfg, str(out_dir))
    (out_dir / "questions.json").write_text(json.dumps(QUESTIONS, indent=2))
    card.update({
        "temperature": fitted["temperature"], "temperature_by_options": fitted["temperature_by_options"],
        "calibration": report, "abstention_thresholds": thr,
        "eval_uncalibrated": evaluate_records(erecs) if erecs else None,
        "eval": evaluate_records(erecs, fitted["temperature"], fitted["temperature_by_options"]) if erecs else None,
        "eval_verdict": verdict_metrics(erecs, fitted["temperature"], fitted["temperature_by_options"]) if erecs else None,
        "eval_verdict_uncalibrated": verdict_metrics(erecs) if erecs else None,
        "calib_verdict": verdict_metrics(crecs, fitted["temperature"], fitted["temperature_by_options"]),
    })
    (out_dir / "training_card.json").write_text(json.dumps(card, indent=2, default=str))
    return card


def _run_stage_a_once(rows: dict, out_dir: Path, cfg: StageAConfig, plan: dict) -> dict:
    import torch
    from laya.train import (TrainConfig, calibration_records, evaluate_records, load_checkpoint,
                            resolve_device, train_model, uses_parallel_layout)

    dev = resolve_device(plan["device"])
    model, tok, mcfg = load_checkpoint(cfg.base)
    head_max_len = mcfg.get("head_max_len", 192)
    max_len = plan["max_len"]
    parallel = uses_parallel_layout(mcfg)
    tr, skip_tr = build_items(tok, rows["train"], max_len, head_max_len)
    ca, _ = build_items(tok, rows["calib"], max_len, head_max_len)
    ev, _ = build_items(tok, rows.get("eval", []), max_len, head_max_len)
    if not tr:
        raise ValueError(f"no training items (skipped {skip_tr})")
    n_frozen = freeze_embeddings(model) if cfg.freeze_emb else 0
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model.to(dev)
    base_recs = calibration_records(model, tok, with_target(ev), dev, max_len, head_max_len, batch_size=4,
                                    parallel=parallel) if ev else []
    tc = TrainConfig(epochs=cfg.epochs, micro_batch=plan["micro_batch"], grad_accum=cfg.grad_accum,
                     encoder_lr=cfg.encoder_lr, head_lr=cfg.head_lr, loss="soft-ce", seed=cfg.seed,
                     max_len=max_len, amp=plan["amp"] if dev.type == "cuda" else False,
                     gradient_checkpointing=plan["amp"] and dev.type == "cuda", log_every=cfg.log_every,
                     freeze_encoder=False)
    t0 = time.time()
    steps = {"n": 0}
    history = train_model(model, tok, tr, tc, dev, max_len, head_max_len, parallel=parallel,
                          on_epoch_end=lambda e, l: steps.__setitem__("n", e + 1))
    wall = time.time() - t0
    n_micro = math.ceil(len(tr) / plan["micro_batch"]) * cfg.epochs
    peak = torch.cuda.max_memory_allocated() / 1e9 if dev.type == "cuda" else None
    card = {
        "stage": "A", "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "base": cfg.base,
        "config": asdict(cfg), "plan": plan, "frozen_params": n_frozen,
        "counts": {"train_rows": len(rows["train"]), "calib_rows": len(rows["calib"]),
                   "eval_rows": len(rows.get("eval", [])), "train_items": len(tr),
                   "calib_items": len(ca), "eval_items": len(ev), "skipped": skip_tr,
                   "train_by_src": _count([r.get("src") for r in rows["train"]])},
        "epoch_loss": history, "wall_s": round(wall, 1),
        "step_time_s_per_microbatch": round(wall / max(1, n_micro), 3),
        "gpu_peak_alloc_gb": None if peak is None else round(peak, 3),
        "eval_base_zero_shot": evaluate_records(base_recs) if base_recs else None,
        "eval_base_verdict": verdict_metrics(base_recs) if base_recs else None,
        "assumptions": {"beta": cfg.beta, "note": "target=(1-beta)*gold+beta*gemma; beta assumed, see 02_ASSUMPTIONS_REGISTER"},
    }
    out_dir = Path(out_dir)
    card = calibrate_save_card(model, tok, mcfg, ca, ev, out_dir, dev=dev, max_len=max_len,
                               head_max_len=head_max_len, card=card, target_error=cfg.target_error,
                               parallel=parallel)
    del model
    free_cuda()
    return card


def _count(xs):
    d = {}
    for x in xs:
        d[x] = d.get(x, 0) + 1
    return d


def train_stage_a(rows: dict, out_dir: Path, cfg: Optional[StageAConfig] = None) -> dict:
    """rows = {"train": [...], "calib": [...], "eval": [...]} of Laya rows (fnp.laya.dataset).
    Retries down the OOM ladder; the card records the plan that finally ran."""
    cfg = cfg or StageAConfig()
    plan = plan_resources(cfg.device, cfg.micro_batch, cfg.max_len)
    last = None
    for p in oom_ladder(plan):
        try:
            print(f"[stage A] plan {p}", flush=True)
            card = _run_stage_a_once(rows, Path(out_dir), cfg, p)
            card["oom_retries"] = oom_ladder(plan).index(p)
            Path(out_dir, "training_card.json").write_text(json.dumps(card, indent=2, default=str))
            return card
        except Exception as e:
            if not is_oom(e):
                raise
            print(f"[stage A] OOM with {p}; retrying cheaper", flush=True)
            last = e
            free_cuda()
    raise RuntimeError("training does not fit even on CPU") from last
