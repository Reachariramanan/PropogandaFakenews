#!/usr/bin/env python3
"""Convert ner_topic_llm_dataset.jsonl → VERL parquet for GSPO.

Also logs dataset stats to MLflow (http://<INTERNAL_IP>:5000 by default).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grpo.mlflow_utils import log_artifact, log_dict_as_json, log_metrics, mlflow_run
from grpo.reward_ie import NOISE_TYPES

_SYSTEM_PREFIX = (
    "You are an information extraction model for geopolitical news and social text. "
    "Given the document, extract named entities and events. Reply with ONLY a JSON "
    "object with keys \"entities\" and \"events\". Do not wrap in markdown. "
)

# The rich product schema. Note `confidence` is deliberately no longer requested:
# it is null in 100% of source event labels, so asking for it only trains the
# model to spend response tokens emitting nulls.
SYSTEM_IE_FULL = _SYSTEM_PREFIX + (
    "Generate as completely as the text supports: entity text/type/subtype/score/"
    "normalized/evidence_text, and event actor1/actor2/other_actors/topics/actions/"
    "evidence_text."
)

# Only the fields grpo/reward_ie.py actually scores. Everything the full schema
# adds beyond this earns exactly zero reward while consuming response-token
# budget — measured at ~74% of a mean gold target (1257 -> 332 tokens), and the
# overflow truncates the JSON before it closes, which zeroes the reward outright.
# Use this when you want the response budget spent only on what is rewarded.
SYSTEM_IE_MINIMAL = _SYSTEM_PREFIX + (
    "Emit ONLY these fields and nothing else: for each entity text/type/subtype; "
    "for each event topics and actions, each a list of objects with a \"code\" key. "
    "Omit all other fields."
)

# Back-compat: existing importers expect `SYSTEM_IE`.
SYSTEM_IE = SYSTEM_IE_FULL

# Fields the reward function reads. Kept next to the prompts so the two cannot
# drift apart silently again.
SCORED_ENTITY_FIELDS = ("text", "type", "subtype")

# Null in 100% of source labels (verified over 1899 entities / 451 events in the
# prepared smoke split), so they carry no information and are always dropped.
DEAD_ENTITY_FIELDS = ("char_start", "char_end")
DEAD_EVENT_FIELDS = ("char_start", "char_end", "confidence")


def clean_entity(e: dict[str, Any], minimal: bool = False) -> dict[str, Any] | None:
    et = (e.get("type") or "").upper().strip()
    if et in NOISE_TYPES or not et:
        return None
    text = (e.get("text") or "").strip()
    if not text:
        return None
    out = {
        "text": text,
        "type": et,
        "score": e.get("score"),
        "subtype": e.get("subtype"),
        "normalized": e.get("normalized") or text.lower(),
        "evidence_text": e.get("evidence_text"),
        "resolution_method": e.get("resolution_method"),
    }
    # DEAD_ENTITY_FIELDS are intentionally never emitted (always null upstream).
    if minimal:
        out = {k: v for k, v in out.items() if k in SCORED_ENTITY_FIELDS}
    return out


def clean_topic(t: dict[str, Any]) -> dict[str, Any] | None:
    code = str(t.get("code") or "").strip()
    if not re.match(r"^\d+\.\d+$", code):
        return None
    if t.get("code_verified") is False:
        return None
    return {
        "code": code,
        "name": t.get("name"),
        "confidence": t.get("confidence"),
        "code_verified": t.get("code_verified"),
    }


def clean_action(a: dict[str, Any]) -> dict[str, Any] | None:
    code = str(a.get("code") or "").strip()
    if not re.match(r"^\d{1,2}$", code):
        return None
    try:
        if not (13 <= int(code) <= 20):
            return None
    except ValueError:
        return None
    return {
        "code": code,
        "name": a.get("name"),
        "confidence": a.get("confidence"),
        "code_verified": a.get("code_verified"),
    }


def clean_event(ev: dict[str, Any], minimal: bool = False) -> dict[str, Any]:
    topics = [ct for t in (ev.get("topics") or []) if (ct := clean_topic(t))]
    actions = [ca for a in (ev.get("actions") or []) if (ca := clean_action(a))]
    # Only `code` is scored; drop the rest so the target matches what earns reward.
    if minimal:
        return {
            "topics": [{"code": t["code"]} for t in topics],
            "actions": [{"code": a["code"]} for a in actions],
        }
    other = []
    for o in ev.get("other_actors") or []:
        if isinstance(o, dict):
            other.append({"text": o.get("text"), "entity_type": o.get("entity_type")})
        elif o:
            other.append({"text": str(o), "entity_type": None})
    # DEAD_EVENT_FIELDS are intentionally never emitted (always null upstream).
    return {
        "actor1": ev.get("actor1"),
        "actor2": ev.get("actor2"),
        "other_actors": other,
        "topics": topics,
        "actions": actions,
        "evidence_text": ev.get("evidence_text"),
    }


def build_target(rec: dict[str, Any], minimal: bool = False) -> dict[str, Any]:
    entities = []
    seen = set()
    for e in rec.get("entities") or []:
        ce = clean_entity(e, minimal=minimal)
        if not ce:
            continue
        key = (ce["text"].lower(), ce["type"], ce.get("subtype"))
        if key in seen:
            continue
        seen.add(key)
        entities.append(ce)
    events = []
    for ev in rec.get("events") or []:
        cev = clean_event(ev, minimal=minimal)
        # In minimal schema the actor/evidence fields are gone, so keep any event
        # that still carries a scorable code rather than dropping all of them.
        keep = (
            bool(cev.get("topics") or cev.get("actions"))
            if minimal
            else bool(cev.get("evidence_text") or cev.get("actor1") or cev.get("topics"))
        )
        if keep:
            events.append(cev)
    return {"entities": entities, "events": events}


def build_user(text: str, title: str | None) -> str:
    parts = []
    if title and str(title).strip():
        parts.append(f"Title: {str(title).strip()}")
    parts.append("Document:")
    parts.append(text.strip())
    parts.append(
        '\nExtract entities and events. Return ONLY a JSON object with keys '
        '"entities" and "events".'
    )
    return "\n".join(parts)


def split_ids(ids: list[Any], train_r: float, val_r: float, seed: int):
    rng = random.Random(seed)
    ids = list(ids)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * train_r)
    n_val = int(n * val_r)
    return set(ids[:n_train]), set(ids[n_train : n_train + n_val]), set(ids[n_train + n_val :])


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--source",
        default=str(ROOT / ".." / "NER_TOPIC_Modelling" / "data" / "ner_topic_llm_dataset.jsonl"),
    )
    p.add_argument("--out_dir", default=str(ROOT / "data" / "grpo"))
    p.add_argument(
        "--schema",
        choices=("full", "minimal"),
        default="full",
        help=(
            "full: rich product schema (default, unchanged behaviour). "
            "minimal: emit only the fields grpo/reward_ie.py scores "
            "(entity text/type/subtype, event topics/actions codes). Cuts mean "
            "target size ~74%% (1257->332 tok) and takes gold targets exceeding a "
            "2048-token response budget from 5.4%% to 0%% of docs, so generations "
            "stop truncating and losing all reward."
        ),
    )
    p.add_argument("--max_docs", type=int, default=None)
    p.add_argument("--max_source_chars", type=int, default=6000)
    p.add_argument("--max_target_chars", type=int, default=8000)
    p.add_argument("--train_ratio", type=float, default=0.90)
    p.add_argument("--val_ratio", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mlflow", action="store_true", default=True)
    p.add_argument("--no_mlflow", action="store_true")
    p.add_argument("--mlflow_uri", default=None)
    p.add_argument("--experiment", default="gspo-ie-qwen3-0.6b")
    args = p.parse_args(argv)

    use_mlflow = args.mlflow and not args.no_mlflow
    source = Path(args.source).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    minimal = args.schema == "minimal"
    system_prompt = SYSTEM_IE_MINIMAL if minimal else SYSTEM_IE_FULL

    rows: list[dict[str, Any]] = []
    skipped = {"empty": 0, "no_labels": 0, "too_long": 0}

    if not source.is_file():
        raise SystemExit(
            f"Source dataset not found: {source}\n"
            "Pass --source explicitly. The historical default "
            "(../NER_TOPIC_Modelling/data/ner_topic_llm_dataset.jsonl) no longer "
            "exists — see handover/STATUS.md."
        )

    print(f"Reading {source}  (schema={args.schema})")
    with open(source, encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f, desc="prepare-parquet")):
            if args.max_docs is not None and i >= args.max_docs:
                break
            rec = json.loads(line)
            text = (rec.get("text") or "").strip()
            if not text:
                skipped["empty"] += 1
                continue
            if len(text) > args.max_source_chars:
                text = text[: args.max_source_chars] + "\n…[truncated]"

            target = build_target(rec, minimal=minimal)
            if not target["entities"] and not target["events"]:
                skipped["no_labels"] += 1
                continue
            gt = json.dumps(target, ensure_ascii=False)
            if len(gt) > args.max_target_chars:
                skipped["too_long"] += 1
                continue

            user = build_user(text, rec.get("title"))
            rows.append(
                {
                    "data_source": "ner_topic_ie",
                    "prompt": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user},
                    ],
                    "ability": "ie",
                    "reward_model": {"style": "rule", "ground_truth": gt},
                    "extra_info": {
                        "doc_id": rec.get("doc_id"),
                        "source_platform": rec.get("source_platform"),
                        "split": None,
                        "index": i,
                        "n_entities": len(target["entities"]),
                        "n_events": len(target["events"]),
                        # gold completion for optional SFT
                        "answer": gt,
                    },
                }
            )

    train_ids, val_ids, test_ids = split_ids(
        [r["extra_info"]["doc_id"] for r in rows],
        args.train_ratio,
        args.val_ratio,
        args.seed,
    )
    splits: dict[str, list] = {"train": [], "val": [], "test": []}
    for r in rows:
        did = r["extra_info"]["doc_id"]
        if did in train_ids:
            split = "train"
        elif did in val_ids:
            split = "val"
        else:
            split = "test"
        r["extra_info"]["split"] = split
        splits[split].append(r)

    try:
        import datasets as hf_datasets
    except ImportError:
        # fallback: write jsonl + require pyarrow later
        hf_datasets = None

    for name, part in splits.items():
        path = out_dir / f"{name}.parquet"
        if hf_datasets is not None:
            ds = hf_datasets.Dataset.from_list(part)
            ds.to_parquet(str(path))
        else:
            import pandas as pd

            pd.DataFrame(part).to_parquet(path, index=False)
        print(f"  {name}: {len(part)} → {path}")

    # examples for inspection
    if rows:
        mid = rows[len(rows) // 2]
        (out_dir / "example_prompt.json").write_text(
            json.dumps(mid["prompt"], indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (out_dir / "example_gt.json").write_text(
            mid["reward_model"]["ground_truth"], encoding="utf-8"
        )
        # pretty
        try:
            (out_dir / "example_gt_pretty.json").write_text(
                json.dumps(json.loads(mid["reward_model"]["ground_truth"]), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except json.JSONDecodeError:
            pass

    stats = {
        "num_rows": len(rows),
        "splits": {k: len(v) for k, v in splits.items()},
        "skipped": skipped,
        "source": str(source),
        "avg_entities": sum(r["extra_info"]["n_entities"] for r in rows) / max(len(rows), 1),
        "avg_events": sum(r["extra_info"]["n_events"] for r in rows) / max(len(rows), 1),
    }
    stats_path = out_dir / "stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))

    if use_mlflow:
        with mlflow_run(
            run_name="prepare_parquet",
            tags={"stage": "data_prep", "task": "gspo_ie"},
            params={
                "source": str(source),
                "out_dir": str(out_dir),
                "max_docs": args.max_docs or -1,
                "max_source_chars": args.max_source_chars,
                "seed": args.seed,
            },
            tracking_uri=args.mlflow_uri,
            experiment_name=args.experiment,
        ):
            log_metrics(
                {
                    "num_rows": stats["num_rows"],
                    "num_train": stats["splits"]["train"],
                    "num_val": stats["splits"]["val"],
                    "num_test": stats["splits"]["test"],
                    "skipped_empty": skipped["empty"],
                    "skipped_no_labels": skipped["no_labels"],
                    "skipped_too_long": skipped["too_long"],
                    "avg_entities": stats["avg_entities"],
                    "avg_events": stats["avg_events"],
                }
            )
            log_artifact(stats_path)
            if (out_dir / "example_gt_pretty.json").exists():
                log_artifact(out_dir / "example_gt_pretty.json")
            if (out_dir / "example_prompt.json").exists():
                log_artifact(out_dir / "example_prompt.json")
            log_dict_as_json(stats, "prepare_stats.json")
            print(f"MLflow: logged prepare_parquet → experiment {args.experiment}")

    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
