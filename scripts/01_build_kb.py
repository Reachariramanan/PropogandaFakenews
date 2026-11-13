#!/usr/bin/env python
"""Stage 01: build the debunked-claim KB (train splits only) -> data/models/kb_v1/. Dry run prints counts."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from fnp.retrieval import kb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--version", default="kb_v2", help="output dir under data/models (never overwrite a used KB)")
    a = ap.parse_args()
    df = kb.build_kb_frame()
    print(df.groupby(["source", "label5"]).size().unstack(fill_value=0).to_string())
    print("total", len(df))
    if not a.apply:
        print("(dry run)")
        return
    emb = np.concatenate([kb.embed_texts(df.claim[i:i + 512].tolist())
                          for i in range(0, len(df), 512)])
    out = kb.MODELS_DIR / a.version
    if (out / "emb.npy").exists():
        raise SystemExit(f"{out} exists; pick a new --version")
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "meta.parquet")
    np.save(out / "emb.npy", emb)       # written last: its presence marks the KB complete
    print("saved", emb.shape, "->", out)


if __name__ == "__main__":
    main()
