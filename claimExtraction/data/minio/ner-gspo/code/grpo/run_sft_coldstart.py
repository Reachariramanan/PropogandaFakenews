#!/usr/bin/env python3
"""Light SFT cold-start on gold IE JSON before GSPO. Logs to MLflow."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grpo.mlflow_utils import log_artifact, log_metrics, mlflow_run, setup_mlflow


class IESFTDataset(Dataset):
    def __init__(self, parquet_path: str, tokenizer, max_length: int = 3072, max_samples: int | None = None):
        import pandas as pd

        df = pd.read_parquet(parquet_path)
        if max_samples:
            # Row order in the parquet is not randomized (e.g. grouped by source/
            # language) -- df.head() previously took a skewed prefix (99.7% English
            # in one measured case, vs ~92% for a proper sample) and silently
            # taught the model to only terminate cleanly on that slice.
            df = df.sample(n=min(max_samples, len(df)), random_state=42).reset_index(drop=True)
        self.rows = df.to_dict("records")
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        prompt = r["prompt"]
        if isinstance(prompt, str):
            prompt = json.loads(prompt)
        prompt = [dict(m) for m in list(prompt)]

        # `reward_model.ground_truth` is authoritative: it is the exact string the
        # RL reward scores against, so SFT must imitate it and nothing else.
        # `extra_info.answer` is only a fallback — prepare_parquet.py writes the
        # same string to both, but a converted split can leave a stale copy there
        # (see grpo/reschema_parquet.py), and silently training on the stale one
        # would teach a schema the reward doesn't score.
        rm = r.get("reward_model")
        if isinstance(rm, str):
            try:
                rm = json.loads(rm)
            except json.JSONDecodeError:
                rm = None
        completion = (rm or {}).get("ground_truth") if isinstance(rm, dict) else None
        if completion is None:
            ei = r.get("extra_info") or {}
            if isinstance(ei, str):
                try:
                    ei = json.loads(ei)
                except json.JSONDecodeError:
                    ei = {}
            completion = (ei or {}).get("answer")
        if completion is None:
            raise KeyError(
                f"row {idx}: no reward_model.ground_truth and no extra_info.answer — "
                "cannot build an SFT target"
            )

        messages = prompt + [{"role": "assistant", "content": completion}]
        # Prefer chat template
        if hasattr(self.tokenizer, "apply_chat_template"):
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            prompt_text = self.tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True
            )
        else:
            text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
            prompt_text = "\n".join(f"{m['role']}: {m['content']}" for m in prompt) + "\nassistant: "

        tok_kwargs = dict(truncation=True, max_length=self.max_length, padding=False, return_tensors=None)
        enc = self.tokenizer(text, **tok_kwargs)
        input_ids = enc["input_ids"]

        # Mask the prompt so loss is computed on the JSON target only. Without
        # this, ~70% of supervised tokens (up to 91% on long documents) are the
        # system prompt and source document, so the cold-start mostly learns to
        # reproduce its own input instead of learning to emit the target JSON.
        # The boundary is measured as the token-level common prefix rather than
        # assumed, so a chat template that doesn't render the assistant turn as a
        # clean suffix of the generation prompt can't silently misalign it.
        prompt_ids = self.tokenizer(prompt_text, **tok_kwargs)["input_ids"]
        n_prompt = 0
        for a, b in zip(input_ids, prompt_ids):
            if a != b:
                break
            n_prompt += 1

        labels = list(input_ids)
        # Guarantee at least one supervised token: an all -100 row yields a NaN
        # loss that propagates through the batch mean. This only triggers if
        # max_length truncated the entire target away.
        n_mask = min(n_prompt, max(len(labels) - 1, 0))
        for i in range(n_mask):
            labels[i] = -100

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def collate(features, pad_id: int):
    max_len = max(f["input_ids"].size(0) for f in features)
    bsz = len(features)
    input_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
    labels = torch.full((bsz, max_len), -100, dtype=torch.long)
    for i, f in enumerate(features):
        L = f["input_ids"].size(0)
        input_ids[i, :L] = f["input_ids"]
        attention_mask[i, :L] = f["attention_mask"]
        labels[i, :L] = f["labels"]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--train_file", default=str(ROOT / "data/grpo/train.parquet"))
    ap.add_argument("--output_dir", default=str(ROOT / "outputs/sft_qwen3_0.6b_ie"))
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_seq_length", type=int, default=3072)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--max_steps", type=int, default=None, help="Smoke: cap optimizer steps")
    ap.add_argument("--save_every", type=int, default=50, help="Save a resumable checkpoint every N optimizer steps")
    ap.add_argument("--resume_from", default=None, help="Path to a previous checkpoint dir to resume model+optimizer+step from")
    ap.add_argument("--attn_implementation", default="sdpa")
    ap.add_argument("--mlflow_uri", default=None)
    ap.add_argument("--experiment", default="gspo-ie-qwen3-0.6b")
    ap.add_argument("--no_mlflow", action="store_true")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from torch.utils.data import DataLoader

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    load_path = args.resume_from or args.model
    tok = AutoTokenizer.from_pretrained(load_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ds = IESFTDataset(args.train_file, tok, args.max_seq_length, args.max_samples)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda fs: collate(fs, tok.pad_token_id),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        load_path,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
    ).to(device)
    model.gradient_checkpointing_enable()
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    start_step = 0
    if args.resume_from:
        opt_path = Path(args.resume_from) / "optimizer.pt"
        state_path = Path(args.resume_from) / "train_state.json"
        if opt_path.exists():
            opt.load_state_dict(torch.load(opt_path, map_location=device))
            print(f"Resumed optimizer state from {opt_path}")
        if state_path.exists():
            start_step = json.loads(state_path.read_text()).get("global_step", 0)
            print(f"Resuming from global_step={start_step}")

    def save_checkpoint(step: int, final: bool = False):
        ckpt_dir = out / "final" if final else out / f"step_{step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(ckpt_dir)
        tok.save_pretrained(ckpt_dir)
        torch.save(opt.state_dict(), ckpt_dir / "optimizer.pt")
        (ckpt_dir / "train_state.json").write_text(json.dumps({"global_step": step}), encoding="utf-8")
        print(f"Saved checkpoint → {ckpt_dir} (step {step})")
        return ckpt_dir

    use_mlflow = not args.no_mlflow
    ctx = (
        mlflow_run(
            run_name=f"sft_coldstart_{Path(args.model).name}",
            tags={"stage": "sft", "model": args.model},
            params={
                "model": args.model,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "grad_accum": args.grad_accum,
                "lr": args.lr,
                "max_samples": args.max_samples or -1,
                "n_train": len(ds),
            },
            tracking_uri=args.mlflow_uri,
            experiment_name=args.experiment,
        )
        if use_mlflow
        else _nullcontext()
    )

    global_step = start_step
    last_ckpt_dir = None
    with ctx:
        for epoch in range(args.epochs):
            running = 0.0
            opt.zero_grad()
            pbar = tqdm(loader, desc=f"sft epoch {epoch}")
            for step, batch in enumerate(pbar):
                batch = {k: v.to(device) for k, v in batch.items()}
                out_m = model(**batch)
                loss = out_m.loss / args.grad_accum
                loss.backward()
                running += loss.item() * args.grad_accum
                if (step + 1) % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    opt.zero_grad()
                    global_step += 1
                    if use_mlflow and global_step % 10 == 0:
                        log_metrics({"sft/loss": running / max(step + 1, 1)}, step=global_step)
                    pbar.set_postfix(loss=running / max(step + 1, 1))
                    if args.save_every and global_step % args.save_every == 0:
                        new_ckpt = save_checkpoint(global_step)
                        # Only one resumable checkpoint kept at a time — disk-cheap, crash-safe.
                        if last_ckpt_dir and last_ckpt_dir != new_ckpt and last_ckpt_dir.exists():
                            import shutil

                            shutil.rmtree(last_ckpt_dir, ignore_errors=True)
                        last_ckpt_dir = new_ckpt
                    if args.max_steps and global_step >= args.max_steps:
                        break
            if args.max_steps and global_step >= args.max_steps:
                break

        final_dir = save_checkpoint(global_step, final=True)
        if last_ckpt_dir and last_ckpt_dir != final_dir and last_ckpt_dir.exists():
            import shutil

            shutil.rmtree(last_ckpt_dir, ignore_errors=True)
        meta = {"model": args.model, "steps": global_step, "output": str(final_dir)}
        (out / "sft_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        if use_mlflow:
            log_metrics({"sft/final_steps": float(global_step)})
            log_artifact(out / "sft_meta.json")
            log_metrics({"sft/checkpoint_saved": 1.0})
        print(f"Saved SFT checkpoint → {final_dir}")
        print(f"For GSPO: MODEL_PATH={final_dir} bash grpo/run_gspo_qwen3_5.sh")


class _nullcontext:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


if __name__ == "__main__":
    main()
