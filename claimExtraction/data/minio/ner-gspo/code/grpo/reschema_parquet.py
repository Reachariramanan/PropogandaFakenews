#!/usr/bin/env python3
"""Re-render an existing VERL parquet split to the minimal (reward-scored) schema.

Why this exists: `prepare_parquet.py --schema minimal` only helps when you can
re-run data prep, and the upstream source JSONL
(`../NER_TOPIC_Modelling/data/ner_topic_llm_dataset.jsonl`) is currently missing
— see handover/STATUS.md. This converts already-prepared parquet in place so the
schema fix is usable without it.

The conversion is lossless with respect to reward: `grpo/reward_ie.py` reads only
entity text/type/subtype and event topics[].code / actions[].code, and the minimal
schema is a strict field-subset that preserves exactly those. Verified
reward-equivalent over the prepared split (see --verify).

Usage:
  python grpo/reschema_parquet.py --in_dir data/grpo            # writes *.minimal.parquet
  python grpo/reschema_parquet.py --in_dir data/grpo --in_place  # backs up, then overwrites
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grpo.prepare_parquet import SCORED_ENTITY_FIELDS, SYSTEM_IE_MINIMAL
from grpo.reward_ie import compute_score_detailed


def slim_target(gt: dict) -> dict:
    return {
        "entities": [
            {k: v for k, v in e.items() if k in SCORED_ENTITY_FIELDS}
            for e in (gt.get("entities") or [])
        ],
        "events": [
            {
                "topics": [{"code": t.get("code")} for t in (ev.get("topics") or [])],
                "actions": [{"code": a.get("code")} for a in (ev.get("actions") or [])],
            }
            for ev in (gt.get("events") or [])
            if (ev.get("topics") or ev.get("actions"))
        ],
    }


def convert(df, verify: bool = True):
    import pandas as pd  # noqa: F401  (ensured by caller)

    n_rows = 0
    n_events_dropped = 0
    mismatches = 0

    prompts = []
    reward_models = []
    extra_infos = []
    for _, row in df.iterrows():
        prompt = row["prompt"]
        if hasattr(prompt, "tolist"):
            prompt = prompt.tolist()
        prompt = [dict(m) for m in prompt]
        for msg in prompt:
            if msg.get("role") == "system":
                msg["content"] = SYSTEM_IE_MINIMAL

        rm = row["reward_model"]
        if isinstance(rm, str):
            rm = json.loads(rm)
        rm = dict(rm)
        gt_full = json.loads(rm["ground_truth"])
        gt_min = slim_target(gt_full)
        n_events_dropped += len(gt_full.get("events") or []) - len(gt_min["events"])

        if verify:
            a = compute_score_detailed(
                solution_str=rm["ground_truth"], ground_truth=rm["ground_truth"]
            )["score"]
            b = compute_score_detailed(
                solution_str=json.dumps(gt_min, ensure_ascii=False),
                ground_truth=json.dumps(gt_min, ensure_ascii=False),
            )["score"]
            # Also cross-check: minimal prediction against the ORIGINAL full gold.
            c = compute_score_detailed(
                solution_str=json.dumps(gt_min, ensure_ascii=False),
                ground_truth=rm["ground_truth"],
            )["score"]
            if not (round(a, 9) == round(b, 9) == round(c, 9)):
                mismatches += 1

        gt_min_str = json.dumps(gt_min, ensure_ascii=False)
        rm["ground_truth"] = gt_min_str

        # extra_info.answer is prepare_parquet.py's copy of the same gold string,
        # used as the SFT target fallback. Leaving the full-schema copy here while
        # the prompt and ground_truth are minimal is a live trap: SFT would train
        # minimal prompts against verbose targets, reintroducing the truncation
        # that the minimal schema exists to remove. Keep the counts honest too.
        ei = row.get("extra_info")
        if isinstance(ei, str):
            try:
                ei = json.loads(ei)
            except json.JSONDecodeError:
                ei = {}
        ei = dict(ei) if ei is not None else {}
        if "answer" in ei:
            ei["answer"] = gt_min_str
        if "n_entities" in ei:
            ei["n_entities"] = len(gt_min["entities"])
        if "n_events" in ei:
            ei["n_events"] = len(gt_min["events"])

        prompts.append(prompt)
        reward_models.append(rm)
        extra_infos.append(ei)
        n_rows += 1

    out = df.copy()
    out["prompt"] = prompts
    out["reward_model"] = reward_models
    if "extra_info" in out.columns:
        out["extra_info"] = extra_infos
    return out, {
        "rows": n_rows,
        "events_dropped_unscorable": n_events_dropped,
        "reward_mismatches": mismatches,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default=str(ROOT / "data" / "grpo"))
    ap.add_argument("--splits", nargs="*", default=["train", "val", "test"])
    ap.add_argument(
        "--in_place",
        action="store_true",
        help="Overwrite <split>.parquet (a .full-schema.bak copy is kept first).",
    )
    ap.add_argument("--no_verify", action="store_true")
    args = ap.parse_args()

    import pandas as pd
    from transformers import AutoTokenizer

    in_dir = Path(args.in_dir).resolve()
    tok = None
    ckpt = ROOT / "outputs" / "qwen3_5_0.8b_text_only"
    if ckpt.is_dir():
        tok = AutoTokenizer.from_pretrained(str(ckpt))

    for split in args.splits:
        src = in_dir / f"{split}.parquet"
        if not src.is_file():
            print(f"skip {split}: {src} not found")
            continue
        df = pd.read_parquet(src)
        out, stats = convert(df, verify=not args.no_verify)

        if tok is not None:
            def _tok_len(rm):
                if isinstance(rm, str):
                    rm = json.loads(rm)
                return len(tok(rm["ground_truth"])["input_ids"])

            before = sum(_tok_len(r) for r in df["reward_model"]) / max(len(df), 1)
            after = sum(_tok_len(r) for r in out["reward_model"]) / max(len(out), 1)
            stats["mean_gold_tokens"] = f"{before:.0f} -> {after:.0f}"

        if args.in_place:
            bak = in_dir / f"{split}.full-schema.bak.parquet"
            if not bak.exists():
                shutil.copy2(src, bak)
            dst = src
        else:
            dst = in_dir / f"{split}.minimal.parquet"
        out.to_parquet(dst, index=False)
        print(f"{split}: {stats}  ->  {dst.name}")

    print(
        "\nreward_mismatches must be 0 — a nonzero value means the minimal schema "
        "changed what the reward scores, which would be a bug."
    )


if __name__ == "__main__":
    main()
