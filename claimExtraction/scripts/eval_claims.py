#!/usr/bin/env python3
"""Baseline/fine-tuned claim-extraction eval on the claims-data parquet-style jsonl.
Metrics: JSON validity, verbatim rate, empty-doc accuracy, claim P/R/F1 at exact and
IoU>=0.5 (char-span) matching. Greedy decoding."""
import argparse, json, random, re, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_rows(path, n, seed=0):
    rows = [json.loads(l) for l in open(path)]
    random.Random(seed).shuffle(rows)
    return rows[:n]

def parse(out):
    out = out.strip()
    m = re.search(r"\{.*\}", out, re.S)
    try:
        d = json.loads(m.group(0) if m else out)
        cl = [c["claim"] for c in d["claims"] if isinstance(c, dict) and isinstance(c.get("claim"), str)]
        return cl, True
    except Exception:
        return [], False

def spans(doc, claims):
    s = []
    for c in claims:
        i = doc.find(c)
        if i >= 0: s.append((i, i + len(c)))
    return s

def iou(a, b):
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    return inter / (a[1] - a[0] + b[1] - b[0] - inter) if inter else 0.0

def match(pred, gold, thr):
    used, tp = set(), 0
    for p in pred:
        best, bj = 0, None
        for j, g in enumerate(gold):
            if j in used: continue
            v = iou(p, g)
            if v > best: best, bj = v, j
        if bj is not None and best >= thr: used.add(bj); tp += 1
    return tp

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=150); ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--max_new", type=int, default=768); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model); tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).cuda().eval()
    rows = load_rows(a.data, a.n)
    stats = dict(valid=0, verbatim=0, pred=0, gold=0, tp_exact=0, tp_iou=0, empty_ok=0, empty_n=0, cap=0)
    preds = []
    t0 = time.time()
    for i in range(0, len(rows), a.bs):
        b = rows[i:i + a.bs]
        texts = [tok.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True) for r in b]
        enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=a.max_new, do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
        for r, o in zip(b, g[:, enc.input_ids.shape[1]:]):
            raw = tok.decode(o, skip_special_tokens=True)
            doc = r["prompt"][1]["content"]
            gold = [c["claim"] for c in json.loads(r["reward_model"]["ground_truth"])["claims"]]
            pc, ok = parse(raw)
            stats["valid"] += ok; stats["cap"] += (o != tok.pad_token_id).sum().item() >= a.max_new
            stats["verbatim"] += sum(c in doc for c in pc)
            ps, gs = spans(doc, pc), spans(doc, gold)
            stats["pred"] += len(pc); stats["gold"] += len(gold)
            stats["tp_exact"] += len(set(pc) & set(gold))
            stats["tp_iou"] += match(ps, gs, 0.5)
            if not gold: stats["empty_n"] += 1; stats["empty_ok"] += (len(pc) == 0)
            preds.append(dict(doc_id=r["extra_info"]["doc_id"], raw=raw, pred=pc, gold=gold))
        print(f"{i+len(b)}/{len(rows)} {time.time()-t0:.0f}s", file=sys.stderr, flush=True)
    s = stats; n = len(rows)
    def prf(tp): 
        p = tp / max(1, s["pred"]); r = tp / max(1, s["gold"]); return round(p, 3), round(r, 3), round(2*p*r/max(1e-9, p+r), 3)
    res = dict(n=n, json_valid=round(s["valid"]/n, 3), verbatim=round(s["verbatim"]/max(1, s["pred"]), 3),
               hit_token_cap=s["cap"], empty_doc_acc=round(s["empty_ok"]/max(1, s["empty_n"]), 3), empty_n=s["empty_n"],
               pred_per_doc=round(s["pred"]/n, 2), gold_per_doc=round(s["gold"]/n, 2),
               exact_PRF=prf(s["tp_exact"]), iou50_PRF=prf(s["tp_iou"]))
    print(json.dumps(res)); json.dump(dict(summary=res, preds=preds), open(a.out, "w"), ensure_ascii=False, indent=1)
main()
