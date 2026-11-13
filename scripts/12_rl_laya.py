#!/usr/bin/env python
"""Stage B: RLCD fine-tune of the stage-A Laya checkpoint (KL-anchored), then refit temperatures
and abstention thresholds on the calib split. Dry-run by default; `--apply` trains and writes
data/models/laya_fnp_v{N+1}/ (never overwrites). See fnp/laya/rlcd.py for the objective.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fnp.config import MODELS_DIR, ensure_dirs
from fnp.laya.dataset import read_jsonl, split_rows
from fnp.laya.rlcd import RLConfig, train_stage_b
from fnp.laya.synth import synth_records
from fnp.laya.train import next_version, plan_resources


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", required=True, help="stage-A checkpoint dir (the KL anchor)")
    ap.add_argument("--rows", help="Laya-format JSONL (same file as stage A, same deterministic split)")
    ap.add_argument("--synthetic", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--micro-batch", type=int)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-len", type=int)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--lam", type=float, default=0.5, help="INCONCLUSIVE penalty under strong evidence")
    ap.add_argument("--kl", type=float, default=0.1)
    ap.add_argument("--ce", type=float, default=0.25)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args(argv)
    ensure_dirs()
    rows = read_jsonl(a.rows) if a.rows else synth_records(a.synthetic) if a.synthetic else None
    if not rows:
        ap.error("give --rows or --synthetic N")
    parts = split_rows(rows)
    cfg = RLConfig(epochs=a.epochs, micro_batch=a.micro_batch, grad_accum=a.grad_accum, max_len=a.max_len,
                   group=a.group, lam_inconclusive=a.lam, kl_coef=a.kl, ce_coef=a.ce, device=a.device)
    print({k: len(v) for k, v in parts.items()}, "plan:", plan_resources(a.device, a.micro_batch, a.max_len), cfg)
    if not a.apply:
        print("dry-run only; pass --apply to train")
        return 0
    out = Path(a.out) if a.out else MODELS_DIR / f"laya_fnp_v{next_version()}"
    if out.exists() and any(out.iterdir()):
        ap.error(f"{out} exists; versions are never overwritten")
    card = train_stage_b(parts, a.init, out, cfg)
    print(json.dumps({k: card[k] for k in ("counts", "gpu_peak_alloc_gb", "step_time_s_per_microbatch", "wall_s",
                                           "eval_verdict", "eval_parent_verdict_uncalibrated") if k in card},
                     indent=1, default=str))
    print("saved", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
