#!/usr/bin/env python3
"""Build a self-contained text-only checkpoint from the multimodal Qwen3.5 repo.

`Qwen/Qwen3.5-0.8B` ships as `Qwen3_5ForConditionalGeneration` (vision + text).
This project is pure-text IE, so we want only the text backbone. Two things have
to happen, and getting either wrong fails deep inside worker startup rather than
here:

1. **Config**: resolve the nested `text_config` and stamp
   `architectures: ["Qwen3_5ForCausalLM"]`. VERL picks its HF auto-class from
   `hf_config.architectures[0]` / config type (`verl/verl/utils/model.py::
   get_hf_auto_model_class`); the composite `Qwen3_5Config` is registered under
   `AutoModelForImageTextToText`, so leaving it alone loads the full VLM
   including a vision tower we never use.

2. **Weight keys**: transformers serializes this class with a `language_model.`
   segment (`model.language_model.layers.N...`), but vLLM's native
   `Qwen3_5ForCausalLM` expects `model.layers.N...` — only its
   `...ForConditionalGeneration` sibling has a `language_model` wrapper. Without
   the remap, the FSDP actor loads fine and the vLLM rollout server dies with
   `ValueError: There is no module or parameter named 'language_model' in
   Qwen3_5Model`.

Both the HF actor path and the vLLM rollout path read the same on-disk directory,
so the fix has to live in the checkpoint, not in a launch flag — vLLM is started
as `serve <path>` with no config-override hook for this.

Usage:
  python grpo/build_qwen3_5_text_only.py
  python grpo/build_qwen3_5_text_only.py --repo Qwen/Qwen3.5-0.8B --out_dir outputs/qwen3_5_0.8b_text_only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

LANG_PREFIX = "model.language_model."
LANG_PREFIX_FIXED = "model."


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--out_dir", default=str(ROOT / "outputs" / "qwen3_5_0.8b_text_only"))
    ap.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    ap.add_argument(
        "--verify_vllm",
        action="store_true",
        help="Additionally check the saved keys against vLLM's expected module names.",
    )
    args = ap.parse_args()

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    dtype = getattr(torch, args.dtype)

    # ---- 1. text-only config -------------------------------------------------
    cfg = AutoConfig.from_pretrained(args.repo)
    text_cfg = cfg.get_text_config()
    if type(text_cfg) is type(cfg):
        raise SystemExit(
            f"{args.repo}: config has no distinct nested text_config "
            f"(got {type(cfg).__name__}). This script is for the multimodal "
            "Qwen3.5 layout; a text-only repo needs no conversion."
        )
    text_cfg.architectures = ["Qwen3_5ForCausalLM"]
    print(f"text config: {type(text_cfg).__name__}  layers={text_cfg.num_hidden_layers}")

    # ---- 2. load only the text tower ----------------------------------------
    # Qwen3_5ForCausalLM declares _keys_to_ignore_on_load_unexpected for the
    # vision/MTP keys, so the extra tensors in the checkpoint are dropped here.
    print(f"loading {args.repo} text backbone ...")
    model = AutoModelForCausalLM.from_pretrained(args.repo, config=text_cfg, dtype=dtype)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"loaded {type(model).__name__}  params={n_params / 1e9:.3f}B")
    for attr in ("visual", "vision_tower", "mtp"):
        if hasattr(model, attr):
            raise SystemExit(f"unexpected {attr!r} submodule survived text-only load")

    model.save_pretrained(out_dir)
    AutoTokenizer.from_pretrained(args.repo).save_pretrained(out_dir)
    print(f"saved to {out_dir}")

    # ---- 3. strip the `language_model.` key segment --------------------------
    shards = sorted(out_dir.glob("*.safetensors"))
    if not shards:
        raise SystemExit(f"no .safetensors written under {out_dir}")

    total = renamed = 0
    for shard in shards:
        tensors = {}
        with safe_open(str(shard), framework="pt") as f:
            metadata = f.metadata()
            for k in f.keys():
                total += 1
                nk = k
                if k.startswith(LANG_PREFIX):
                    nk = LANG_PREFIX_FIXED + k[len(LANG_PREFIX) :]
                    renamed += 1
                tensors[nk] = f.get_tensor(k)
        save_file(tensors, str(shard), metadata=metadata)
    print(f"weight keys: {renamed}/{total} renamed ({LANG_PREFIX} -> {LANG_PREFIX_FIXED})")

    # Keep any weight map in sync with the renamed keys.
    index = out_dir / "model.safetensors.index.json"
    if index.is_file():
        idx = json.loads(index.read_text())
        idx["weight_map"] = {
            (k.replace(LANG_PREFIX, LANG_PREFIX_FIXED, 1) if k.startswith(LANG_PREFIX) else k): v
            for k, v in idx.get("weight_map", {}).items()
        }
        index.write_text(json.dumps(idx, indent=2))
        print("updated model.safetensors.index.json weight_map")

    # ---- 4. verify the result actually reloads ------------------------------
    reloaded = AutoModelForCausalLM.from_pretrained(out_dir, dtype=dtype)
    assert type(reloaded).__name__ == "Qwen3_5ForCausalLM", type(reloaded).__name__
    leaked = [k for k in reloaded.state_dict() if "language_model" in k]
    assert not leaked, f"language_model keys survived: {leaked[:5]}"
    print(f"reload OK: {type(reloaded).__name__}")

    if args.verify_vllm:
        from vllm.model_executor.models.registry import ModelRegistry

        archs = ModelRegistry.get_supported_archs()
        assert "Qwen3_5ForCausalLM" in archs, "vLLM lacks Qwen3_5ForCausalLM support"
        print("vLLM reports Qwen3_5ForCausalLM supported")

    print(f"\nDone. Use as MODEL_PATH:\n  {out_dir}")


if __name__ == "__main__":
    main()
