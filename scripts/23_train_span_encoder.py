"""Stage 23: fine-tune the Laya mmBERT-base encoder (322M, multilingual) as a propaganda span tagger.

Task: per-token multi-label (sigmoid) over the 14 PTC classes -> spans = runs of tokens above a
per-class threshold (overlapping spans of different techniques are allowed). Sliding windows
(max_len tokens, stride overlap), embeddings frozen, bf16 autocast + activation checkpointing,
micro-batch / length from fnp.laya.train.plan_resources with an OOM ladder.
Thresholds are tuned on the dev split only; test is scored once at the end.

  .venv/bin/python scripts/23_train_span_encoder.py --epochs 8
Outputs: data/models/prop_span_v1/ (weights, thresholds), reports/EVAL_prop_v2.{json,md}
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers import AutoTokenizer, ModernBertConfig, ModernBertModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fnp.datasets import load_ptc  # noqa: E402
from fnp.eval.ptc_scorer import CLASSES14, score  # noqa: E402
from fnp.laya.train import free_cuda, is_oom, oom_ladder, plan_resources  # noqa: E402

BASE = glob.glob(str(Path.home() / ".cache/huggingface/hub/models--convaiinnovations--laya/snapshots/*/multilingual"))[0]
OUT = ROOT / "data/models/prop_span_v1"
C = len(CLASSES14)
CIDX = {c: i for i, c in enumerate(CLASSES14)}


class SpanTagger(nn.Module):
    def __init__(self):
        super().__init__()
        cfg = ModernBertConfig.from_pretrained(f"{BASE}/encoder")
        self.encoder = ModernBertModel(cfg)
        sd = {k[len("encoder."):]: v for k, v in load_file(f"{BASE}/model.safetensors").items()
              if k.startswith("encoder.")}
        miss = self.encoder.load_state_dict(sd, strict=False)
        print("loaded encoder; missing", len(miss.missing_keys), "unexpected", len(miss.unexpected_keys))
        self.drop = nn.Dropout(0.1)
        self.head = nn.Linear(cfg.hidden_size, C)

    def forward(self, ids, mask):
        h = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        return self.head(self.drop(h))


def windows(n_tokens: int, size: int, stride: int):
    out, s = [], 0
    while True:
        out.append((s, min(n_tokens, s + size)))
        if s + size >= n_tokens:
            return out
        s += stride


def encode_doc(tok, text: str, size: int):
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    return enc["input_ids"], enc["offset_mapping"]


def token_labels(offs, spans) -> np.ndarray:
    y = np.zeros((len(offs), C), dtype=np.float32)
    for s, e, t in spans:
        k = CIDX.get(t)
        if k is None:
            continue
        for i, (a, b) in enumerate(offs):
            if b > s and a < e:
                y[i, k] = 1.0
    return y


def make_items(tok, texts: dict, gold: dict, docs, size: int, stride: int):
    items = []
    for d in docs:
        ids, offs = encode_doc(tok, texts[d], size)
        y = token_labels(offs, gold.get(d, []))
        for a, b in windows(len(ids), size - 2, stride):
            items.append((d, a, [tok.cls_token_id] + ids[a:b] + [tok.sep_token_id], y[a:b]))
    return items


def collate(batch, pad_id, dev):
    L = max(len(x[2]) for x in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch), L), dtype=torch.long)
    y = torch.zeros((len(batch), L, C))
    ym = torch.zeros((len(batch), L))
    for i, (_, _, t, lab) in enumerate(batch):
        ids[i, :len(t)] = torch.tensor(t)
        mask[i, :len(t)] = 1
        y[i, 1:1 + len(lab)] = torch.from_numpy(lab)
        ym[i, 1:1 + len(lab)] = 1
    return ids.to(dev), mask.to(dev), y.to(dev), ym.to(dev)


@torch.no_grad()
def predict_probs(model, tok, texts, docs, plan):
    model.eval()
    dev = plan["device"]
    out = {}
    for d in docs:
        ids, offs = encode_doc(tok, texts[d], plan["max_len"])
        acc = np.zeros((len(ids), C), dtype=np.float32)
        cnt = np.zeros(len(ids), dtype=np.float32)
        for a, b in windows(len(ids), plan["max_len"] - 2, (plan["max_len"] - 2) // 2):
            t = [tok.cls_token_id] + ids[a:b] + [tok.sep_token_id]
            x = torch.tensor([t], device=dev)
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=plan["amp"]):
                lg = model(x, torch.ones_like(x))
            p = torch.sigmoid(lg.float())[0, 1:1 + (b - a)].cpu().numpy()
            acc[a:b] += p
            cnt[a:b] += 1
        out[d] = (acc / np.maximum(cnt, 1)[:, None], offs)
    return out


def decode(probs, offs, thr, min_tokens=1):
    spans = []
    for k, c in enumerate(CLASSES14):
        on = probs[:, k] >= thr[k]
        i = 0
        while i < len(on):
            if on[i]:
                j = i
                while j + 1 < len(on) and on[j + 1]:
                    j += 1
                if j - i + 1 >= min_tokens:
                    spans.append((offs[i][0], offs[j][1], c))
                i = j + 1
            else:
                i += 1
    return spans


def tune_thresholds(P, gold, docs):
    grid = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6]
    thr = [0.4] * C
    for _ in range(2):
        for k, c in enumerate(CLASSES14):
            best = (-1, thr[k])
            for g in grid:
                t = list(thr)
                t[k] = g
                pred = {d: decode(*P[d], t) for d in docs}
                f = score(gold, pred, docs)["per_technique"][c]["f1"]
                if f > best[0] + 1e-9:
                    best = (f, g)
            thr[k] = best[1]
    return thr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=4e-5)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--limit", type=int, default=0, help="debug: limit train docs")
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)

    split = json.loads((ROOT / "data/splits/ptc_v1.json").read_text())["split"]
    arts, spans = load_ptc()
    texts = dict(zip(arts.article_id, arts.text))
    gold = defaultdict(list)
    for r in spans.itertuples():
        gold[r.article_id].append((r.start, r.end, r.technique))
    D = {s: sorted(d for d, x in split.items() if x == s) for s in ("train", "dev", "test")}
    if a.limit:
        D["train"] = D["train"][:a.limit]
    tok = AutoTokenizer.from_pretrained(f"{BASE}/tokenizer")

    plan = plan_resources()
    print("plan", plan, flush=True)
    pos_rate = np.zeros(C)
    stride = lambda p: (p["max_len"] - 2) // 2
    done = False
    for p in oom_ladder(plan):
        plan = p
        try:
            dev = plan["device"]
            model = SpanTagger().to(dev)
            for n, prm in model.encoder.named_parameters():
                if "tok_embeddings" in n:
                    prm.requires_grad_(False)
            model.encoder.gradient_checkpointing_enable() if hasattr(model.encoder, "gradient_checkpointing_enable") else None
            items = make_items(tok, texts, gold, D["train"], plan["max_len"], stride(plan))
            ys = np.concatenate([x[3] for x in items])
            pos_rate = ys.mean(0).clip(1e-4)
            pw = torch.tensor(np.clip((1 - pos_rate) / pos_rate, 1, 30) ** 0.5, dtype=torch.float32, device=dev)
            lossf = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pw)
            enc_p = [q for n, q in model.encoder.named_parameters() if q.requires_grad]
            opt = torch.optim.AdamW([{"params": enc_p, "lr": a.lr}, {"params": model.head.parameters(), "lr": 1e-3}],
                                    weight_decay=0.01)
            steps = a.epochs * (len(items) // (plan["micro_batch"] * a.accum) + 1)
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr, 1e-3], total_steps=steps, pct_start=0.1)
            print(f"train windows {len(items)} | mb {plan['micro_batch']} len {plan['max_len']} on {dev} | steps {steps}", flush=True)
            t0, step, mb = time.time(), 0, plan["micro_batch"]
            for ep in range(a.epochs):
                model.train()
                random.shuffle(items)
                tot, n = 0.0, 0
                for i in range(0, len(items), mb):
                    ids, mask, y, ym = collate(items[i:i + mb], tok.pad_token_id, dev)
                    with torch.autocast(dev, dtype=torch.bfloat16, enabled=plan["amp"]):
                        lg = model(ids, mask)
                    loss = (lossf(lg.float(), y).mean(-1) * ym).sum() / ym.sum().clamp(min=1)
                    (loss / a.accum).backward()
                    tot += float(loss); n += 1
                    if n % a.accum == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
                print(f"epoch {ep + 1}/{a.epochs} loss {tot / max(n, 1):.4f} {time.time() - t0:.0f}s", flush=True)
            done = True
            break
        except Exception as e:     # noqa: BLE001
            if not is_oom(e):
                raise
            print("OOM, retrying cheaper:", plan, flush=True)
            model = None; free_cuda()
    if not done:
        raise SystemExit("could not train at any setting")

    OUT.mkdir(parents=True, exist_ok=True)
    pdev = plan["device"]
    Pdev = predict_probs(model, tok, texts, D["dev"], plan)
    thr = tune_thresholds(Pdev, gold, D["dev"])
    res = {"plan": plan, "thresholds": dict(zip(CLASSES14, thr)), "epochs": a.epochs}
    for sp in ("dev", "test"):
        P = Pdev if sp == "dev" else predict_probs(model, tok, texts, D[sp], plan)
        pred = {d: decode(*P[d], thr) for d in D[sp]}
        res[sp] = score(gold, pred, D[sp])
        r = res[sp]
        print(f"{sp}: SI P/R/F1 {r['SI']['precision']:.3f}/{r['SI']['recall']:.3f}/{r['SI']['f1']:.3f} | "
              f"span+tech {r['TC_span']['precision']:.3f}/{r['TC_span']['recall']:.3f}/{r['TC_span']['f1']:.3f}", flush=True)
    torch.save({"head": model.head.state_dict(), "encoder": model.encoder.state_dict(), "thr": thr,
                "classes": CLASSES14}, OUT / "model.pt")
    (ROOT / "reports/EVAL_prop_v2.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
