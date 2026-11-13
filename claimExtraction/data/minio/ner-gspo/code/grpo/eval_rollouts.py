#!/usr/bin/env python3
"""Offline generate + reward eval; logs metrics/artifacts to MLflow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grpo.mlflow_utils import log_artifact, log_dict_as_json, log_metrics, mlflow_run
from grpo.reward_ie import compute_score, compute_score_detailed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--data", default=str(ROOT / "data/grpo/val.parquet"))
    ap.add_argument("--max_samples", type=int, default=50)
    # Must match the training-time response budget. The old default of 1024 was
    # below the mean gold target (~1257 tok under the full schema), so eval
    # truncated essentially every generation and scored 0.0 across the board.
    ap.add_argument("--max_new_tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--out_dir", default=str(ROOT / "outputs/eval_rollouts"))
    ap.add_argument("--mlflow_uri", default=None)
    ap.add_argument("--experiment", default="gspo-ie-qwen3-0.6b")
    ap.add_argument("--no_mlflow", action="store_true")
    ap.add_argument("--run_name", default=None)
    args = ap.parse_args()

    import pandas as pd
    from transformers import AutoModelForCausalLM, AutoTokenizer

    df = pd.read_parquet(args.data).head(args.max_samples)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
        device_map="auto" if device.type == "cuda" else None,
    )
    if device.type != "cuda":
        model = model.to(device)
    model.eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    preds = []
    # `subtype` was previously computed by the reward but silently dropped here.
    # `repaired` counts generations that only parsed after truncation repair —
    # a high value means the response budget is the bottleneck, not extraction
    # quality, which is otherwise invisible in the aggregate scores.
    sums = {
        "score": 0.0,
        "format": 0.0,
        "entity_f1": 0.0,
        "subtype": 0.0,
        "topic_f1": 0.0,
        "action_f1": 0.0,
        "repaired": 0.0,
    }
    diag = {"unparseable": 0, "truncated_repaired": 0, "hit_token_cap": 0}
    n = 0

    use_mlflow = not args.no_mlflow
    run_name = args.run_name or f"eval_{Path(args.model).name}"
    ctx = (
        mlflow_run(
            run_name=run_name,
            tags={"stage": "eval", "model": args.model},
            params={
                "model": args.model,
                "data": args.data,
                "max_samples": args.max_samples,
                "temperature": args.temperature,
            },
            tracking_uri=args.mlflow_uri,
            experiment_name=args.experiment,
        )
        if use_mlflow
        else _nullcontext()
    )

    with ctx:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="eval"):
            prompt = row["prompt"]
            if isinstance(prompt, str):
                prompt = json.loads(prompt)
            elif hasattr(prompt, "tolist"):
                # parquet nested list-of-struct columns come back as numpy
                # arrays; chat templates choke on their truthiness checks.
                prompt = prompt.tolist()
            rm = row["reward_model"]
            if isinstance(rm, str):
                rm = json.loads(rm)
            gt = rm.get("ground_truth", "{}")

            if hasattr(tok, "apply_chat_template"):
                text = tok.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            else:
                text = "\n".join(f"{m['role']}: {m['content']}" for m in prompt) + "\nassistant:"

            inputs = tok(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                gen = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.temperature > 0,
                    temperature=max(args.temperature, 1e-5),
                    top_p=0.9,
                    pad_token_id=tok.pad_token_id,
                )
            new_tokens = gen[0, inputs["input_ids"].shape[-1] :]
            solution = tok.decode(new_tokens, skip_special_tokens=True)
            detail = compute_score_detailed(solution_str=solution, ground_truth=gt)
            for k in sums:
                sums[k] += float(detail.get(k, 0.0))
            n += 1

            # Diagnostics: distinguish "extraction was wrong" from "ran out of
            # tokens", which look identical in the aggregate score otherwise.
            hit_cap = int(new_tokens.shape[-1]) >= args.max_new_tokens
            if hit_cap:
                diag["hit_token_cap"] += 1
            if detail.get("repaired"):
                diag["truncated_repaired"] += 1
            elif detail["format"] == 0.0:
                diag["unparseable"] += 1

            preds.append(
                {
                    "doc_id": (row.get("extra_info") or {}).get("doc_id")
                    if isinstance(row.get("extra_info"), dict)
                    else None,
                    "score": detail["score"],
                    "metrics": detail,
                    "response_tokens": int(new_tokens.shape[-1]),
                    "hit_token_cap": hit_cap,
                    "prediction": solution[:4000],
                }
            )
            if use_mlflow and n % 10 == 0:
                log_metrics({f"eval_running/{k}": sums[k] / n for k in sums}, step=n)

        means = {f"eval/{k}": sums[k] / max(n, 1) for k in sums}
        means.update({f"eval_diag/{k}": float(v) for k, v in diag.items()})
        print(json.dumps(means, indent=2))
        if diag["hit_token_cap"]:
            print(
                f"WARNING: {diag['hit_token_cap']}/{n} generations hit the "
                f"--max_new_tokens={args.max_new_tokens} cap. Scores are bounded by the "
                "response budget, not extraction quality — raise the cap or prepare data "
                "with `--schema minimal` (see grpo/prepare_parquet.py)."
            )
        pred_path = out_dir / "predictions.jsonl"
        with open(pred_path, "w", encoding="utf-8") as f:
            for p in preds:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        summary_path = out_dir / "summary.json"
        summary_path.write_text(json.dumps({"n": n, **means}, indent=2), encoding="utf-8")

        if use_mlflow:
            log_metrics(means)
            log_metrics({"eval/n": float(n)})
            log_artifact(summary_path)
            log_artifact(pred_path)
            log_dict_as_json(means, "eval_means.json")
            print(f"MLflow eval logged → experiment {args.experiment}")


class _nullcontext:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


if __name__ == "__main__":
    main()
