#!/usr/bin/env python3
"""
merge_lora_to_base.py — fold a PEFT LoRA adapter back into its base
model and save a single full-weights model directory. Used to turn
[base SFT] + [GRPO LoRA] into a flat starting point for further SFT.

Usage:
    python scripts/merge_lora_to_base.py \
        --base   results/backup/codev_gpu3_patience_20260423_sft/checkpoint_20260423_132954 \
        --lora   results/grpo_v6_C2_viable/checkpoint-150 \
        --output results/ckpt132954_v6lora150_merged
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--lora", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    args = ap.parse_args()

    base = Path(args.base)
    lora = Path(args.lora)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    dtype = {"bfloat16": torch.bfloat16,
             "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    print(f"[load] base = {base}")
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=dtype, low_cpu_mem_usage=True
    )
    print(f"[load] lora = {lora}")
    model = PeftModel.from_pretrained(model, lora)
    print(f"[merge] folding LoRA into base weights …")
    merged = model.merge_and_unload()

    print(f"[save] {out}")
    merged.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(base).save_pretrained(out)
    print(f"[done]")


if __name__ == "__main__":
    main()
