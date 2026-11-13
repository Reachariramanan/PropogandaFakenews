"""Batched inference with a fine-tuned (or base) Laya checkpoint.

    dec = LayaDecider.load("data/models/laya_fnp_v1")
    dec.predict([state_text, ...])
    -> [{"probs": {label: p}, "label": str, "confidence": float, "margin": float, "entropy": float,
         "noul": {"is_propaganda": p, ...}, "act_probability": float, "truncated": bool}, ...]

`confidence` is max(p) of the TEMPERATURE-CALIBRATED verdict distribution (laya applies the
checkpoint's fitted temperature; the base checkpoints are over-confident, see 04_LAYA_LIMITS.md).
Runs on GPU only when enough memory is free; otherwise (or on OOM) falls back to CPU.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

from fnp.config import LAYA_REPO, LAYA_SUBFOLDER
from fnp.schema import LABELS

from .state_builder import NOUL_QUESTIONS, QUESTIONS


class LayaDecider:
    def __init__(self, agent, device: str, questions: Optional[dict] = None, batch_size: int = 8,
                 path: Optional[str] = None):
        self.agent, self.device, self.path = agent, device, path
        self.questions = questions or QUESTIONS
        self.batch_size = batch_size

    @classmethod
    def load(cls, path: Optional[str] = None, device: str = "auto", batch_size: int = 8,
             min_free_gb: float = 1.5) -> "LayaDecider":
        """`path=None` loads the shipped multilingual base (zero-shot, uncalibrated for FNP)."""
        import laya
        import torch

        if device == "auto":
            free = torch.cuda.mem_get_info()[0] / 1e9 if torch.cuda.is_available() else 0.0
            device = "cuda" if free >= min_free_gb else "cpu"
        try:
            agent = (laya.load(path, device=device) if path else
                     laya.load(LAYA_REPO, subfolder=LAYA_SUBFOLDER, device=device))
        except Exception as e:
            if device == "cuda" and ("out of memory" in str(e).lower() or isinstance(e, torch.OutOfMemoryError)):
                torch.cuda.empty_cache()
                device = "cpu"
                agent = (laya.load(path, device="cpu") if path else
                         laya.load(LAYA_REPO, subfolder=LAYA_SUBFOLDER, device="cpu"))
            else:
                raise
        return cls(agent, device, batch_size=batch_size, path=path)

    def _raw(self, states: Sequence[str]) -> list[dict]:
        import torch
        bs = self.batch_size
        try:
            return self.agent.predict_batch(list(states), self.questions, batch_size=bs)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs > 1:
                self.batch_size = max(1, bs // 2)
                return self._raw(states)
            raise

    @staticmethod
    def _decode(res: dict) -> dict:
        ans = res["answers"]
        v = ans["verdict"]
        probs = {l: float(v["probabilities"].get(l, 0.0)) for l in LABELS}
        s = sum(probs.values()) or 1.0
        probs = {l: p / s for l, p in probs.items()}      # laya rounds to 4dp; renormalise
        arr = np.array([probs[l] for l in LABELS])
        top = np.sort(arr)[::-1]
        ent = float(-(arr * np.log(np.clip(arr, 1e-12, 1))).sum() / math.log(len(LABELS)))
        return {
            "probs": probs, "label": LABELS[int(arr.argmax())], "confidence": float(top[0]),
            "margin": float(top[0] - top[1]), "entropy": ent,
            "noul": {q: float(ans[q]["noul"]) for q in NOUL_QUESTIONS if q in ans},
            "act_probability": float(v.get("action", {}).get("act_probability", float("nan"))),
            "truncated": bool(res.get("usage", {}).get("truncated", False)),
        }

    def predict(self, states: list[str]) -> list[dict]:
        if not states:
            return []
        return [self._decode(r) for r in self._raw(states)]

    @property
    def abstention_thresholds(self) -> dict:
        return dict(getattr(self.agent, "min_confidence_map", None) or {})
