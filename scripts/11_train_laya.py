#!/usr/bin/env python
"""Stage A: supervised distillation of Laya (multilingual 322M) on FNP verdict rows.

Dry-run by default (builds items, prints the resource plan and the package's dry-run stats).
`--apply` trains and writes data/models/laya_fnp_v{N}/ (checkpoint + training_card.json).

Input rows: `--rows file.jsonl` (Laya rows from fnp.laya.dataset.make_row) or `--synthetic N`.
Target = (1-beta)*gold one-hot + beta*Gemma probs (beta=0.3, assumed; see 02_ASSUMPTIONS_REGISTER).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fnp.config import DATASETS_DIR, MODELS_DIR, ensure_dirs
from fnp.laya.dataset import read_jsonl, split_rows, write_splits
from fnp.laya.synth import synth_records
from fnp.laya.train import StageAConfig, next_version, plan_resources, train_stage_a


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", help="Laya-format JSONL of all rows (split deterministically by id)")
    ap.add_argument("--synthetic", type=int, default=0, help="use N synthetic rows (smoke only)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--micro-batch", type=int)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-len", type=int)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--beta", type=float, default=0.3, help="recorded only; applied when rows are built")
    ap.add_argument("--out", help="output dir (default data/models/laya_fnp_v{N})")
    ap.add_argument("--apply", action="store_true", help="train and write (default: dry-run)")
    a = ap.parse_args(argv)
    ensure_dirs()

    rows = read_jsonl(a.rows) if a.rows else synth_records(a.synthetic) if a.synthetic else None
    if not rows:
        ap.error("give --rows or --synthetic N")
    parts = split_rows(rows)
    print({k: len(v) for k, v in parts.items()}, "resource plan:", plan_resources(a.device, a.micro_batch, a.max_len))
    cfg = StageAConfig(epochs=a.epochs, micro_batch=a.micro_batch, grad_accum=a.grad_accum, max_len=a.max_len,
                       device=a.device, beta=a.beta)
    if not a.apply:
        from laya.train import TrainConfig, dry_run
        d = DATASETS_DIR / "laya_fnp_dryrun"
        write_splits(rows, d)
        dry_run(str(d / "train.jsonl"), cfg.base, TrainConfig(epochs=cfg.epochs, micro_batch=a.micro_batch or 4,
                                                             grad_accum=cfg.grad_accum, max_len=a.max_len or 512,
                                                             loss="soft-ce", calib_frac=0.0))
        print("dry-run only; pass --apply to train")
        return 0
    out = Path(a.out) if a.out else MODELS_DIR / f"laya_fnp_v{next_version()}"
    if out.exists() and any(out.iterdir()):
        ap.error(f"{out} exists; versions are never overwritten")
    write_splits(rows, out / "data")
    card = train_stage_a(parts, out, cfg)
    print(json.dumps({k: card[k] for k in ("counts", "gpu_peak_alloc_gb", "step_time_s_per_microbatch", "wall_s",
                                           "eval_verdict", "eval_base_verdict") if k in card}, indent=1, default=str))
    print("saved", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
