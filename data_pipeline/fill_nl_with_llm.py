#!/usr/bin/env python3
"""
fill_nl_with_llm.py — for every SFT sample whose `nl` field is empty or
placeholder, generate a natural-language spec from (rtl_context, reference_sva)
using a local code-aware LLM.

The generated NL is meant to match the FVEval-style fragment:
  "that the counter does not overflow. Use the signals 'count', 'count_d1', ..."

Output:
  data/train/sft/sft_train_nl_filled.jsonl  (full pool, EMPTY → filled)
  data/train/sft/sft_train_L<L>_nl_filled.jsonl  (per-TCL splits)

Usage:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python3 scripts/fill_nl_with_llm.py \\
      --model ${STUDENT_MODEL} \\
      --device cuda:0 \\
      --batch-size 1 \\
      --max-new-tokens 256
"""
import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent
SFT_DIR = EXPERIMENTS_DIR / "data" / "train" / "sft"
IN_PATH = SFT_DIR / "sft_train.jsonl"
OUT_PATH = SFT_DIR / "sft_train_nl_filled.jsonl"

PLACEHOLDER_RE = re.compile(r"^\s*\[(?:ASSERT|ASSUME|COVER)[^\]]*\]\s*$",
                            re.IGNORECASE)


def needs_nl(nl: str) -> bool:
    nl = (nl or "").strip()
    if len(nl) < 5:
        return True
    if PLACEHOLDER_RE.match(nl):
        return True
    return False


SYSTEM_PROMPT = (
    "You are an expert SystemVerilog hardware verification engineer. "
    "Given a SystemVerilog module and a SystemVerilog Assertion (SVA) over its "
    "signals, write a one-sentence natural-language description of what the "
    "assertion CHECKS, written as a fragment in the style of:\n"
    "    'that the counter does not overflow. Use the signals X, Y, Z.'\n"
    "    'when req is asserted, gnt must follow within 3 cycles. Use signals "
    "req, gnt.'\n"
    "Output ONLY the fragment — no preamble, no markdown, no explanation, no "
    "quoting. Start with 'that' or 'when' or 'whenever'. End with one sentence "
    "naming the relevant signals."
)


def build_user_prompt(rtl_context: str, reference_sva: str) -> str:
    rtl_context = (rtl_context or "").strip()
    if rtl_context:
        rtl_context = rtl_context[:4000]   # cap to keep token budget reasonable
        return (
            "Module:\n"
            f"```systemverilog\n{rtl_context}\n```\n\n"
            "Assertion:\n"
            f"```systemverilog\n{reference_sva.strip()}\n```\n\n"
            "Write the one-sentence NL description fragment now."
        )
    return (
        "Assertion (no module context available):\n"
        f"```systemverilog\n{reference_sva.strip()}\n```\n\n"
        "Write the one-sentence NL description fragment now."
    )


def generate_nl(tok, model, system: str, user: str, max_new: int,
                disable_thinking: bool = True) -> str:
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": user}]
    # Qwen3 supports `enable_thinking=False` in chat template — disable
    # the <think>...</think> reasoning block since we want a short NL fragment.
    try:
        prompt = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=not disable_thinking,
        )
    except TypeError:
        prompt = tok.apply_chat_template(msgs, tokenize=False,
                                         add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt", truncation=True,
                 max_length=8192).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new, do_sample=False,
            temperature=1.0, top_p=1.0, pad_token_id=tok.eos_token_id,
        )
    text = tok.decode(out[0, inputs["input_ids"].shape[1]:],
                      skip_special_tokens=True).strip()
    # Strip any residual <think>...</think> block
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Cleanup: drop leading "Description:" / "NL:" / quoted wrappers
    text = re.sub(r"^(?:NL|Description|Answer|Output)\s*[:：]\s*", "", text,
                  flags=re.IGNORECASE)
    text = text.strip().strip('"').strip("'").strip("`").strip()
    # Keep only the first sentence (fragment style)
    # Cut at first newline
    text = text.split("\n")[0].strip()
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="${STUDENT_MODEL}")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--limit", type=int, default=0,
                    help="debug: only fill first N missing-NL records")
    ap.add_argument("--in-path", default=str(IN_PATH),
                    help="input jsonl (default: sft_train.jsonl)")
    ap.add_argument("--out", default=str(OUT_PATH))
    ap.add_argument("--skip-per-tcl-splits", action="store_true",
                    help="skip writing sft_train_L{1..5}_nl_filled.jsonl")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    with open(args.in_path) as f:
        for line in f:
            records.append(json.loads(line))
    n_total = len(records)
    missing_idx = [i for i, r in enumerate(records) if needs_nl(r.get("nl", ""))]
    if args.limit:
        missing_idx = missing_idx[:args.limit]
    print(f"[fill-nl] total={n_total}  missing={len(missing_idx)}  "
          f"already_OK={n_total - len(missing_idx)}")

    print(f"[fill-nl] loading model {args.model}")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
             "fp32": torch.float32}[args.dtype]
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, low_cpu_mem_usage=True,
        trust_remote_code=True).to(args.device)
    model.eval()
    print(f"[fill-nl] model loaded")

    t0 = time.time()
    by_source = Counter()
    for ki, idx in enumerate(missing_idx):
        r = records[idx]
        rtl = r.get("rtl_context", "")
        ref = r.get("reference_sva", "")
        user = build_user_prompt(rtl, ref)
        try:
            nl = generate_nl(tok, model, SYSTEM_PROMPT, user,
                             args.max_new_tokens)
        except Exception as e:
            nl = ""
            print(f"  [WARN] generate failed for {r.get('id')}: {e}")
        records[idx]["nl"] = nl
        records[idx]["nl_filled_by"] = "qwen2.5-coder-7b"
        by_source[r.get("source", "?")] += 1
        if (ki + 1) % 25 == 0 or ki + 1 == len(missing_idx):
            elapsed = time.time() - t0
            eta = elapsed / (ki + 1) * (len(missing_idx) - ki - 1)
            print(f"  {ki+1}/{len(missing_idx)}  elapsed={elapsed:.0f}s  "
                  f"eta={eta:.0f}s  by_source={dict(by_source)}")

    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"\n[fill-nl] wrote {n_total} records → {out_path}")
    print(f"[fill-nl] total wallclock: {time.time() - t0:.0f}s")

    # Per-TCL splits
    if not args.skip_per_tcl_splits:
        by_tcl = defaultdict(list)
        for r in records:
            by_tcl[int(r.get("expected_tcl", 0))].append(r)
        for lvl, lst in by_tcl.items():
            lvl_path = SFT_DIR / f"sft_train_L{lvl}_nl_filled.jsonl"
            with open(lvl_path, "w") as f:
                for r in lst:
                    f.write(json.dumps(r) + "\n")
        print(f"[fill-nl] per-TCL splits written: "
              + "  ".join(f"L{lv}={len(by_tcl[lv])}" for lv in sorted(by_tcl)))


if __name__ == "__main__":
    main()
