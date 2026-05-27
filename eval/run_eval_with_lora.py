#!/usr/bin/env python3
"""Eval NL2SVA-Human on (base_model + LoRA adapter)."""
import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.tcl import classify_tcl
from src.mock_verifier import syntax_check

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent
TEST_JSONL = EXPERIMENTS_DIR / "data" / "test" / "nl2sva_human.jsonl"

SYSTEM_PROMPT = (
    "You are an expert in SystemVerilog Assertions (SVA). Given a natural-"
    "language description of a design property, output ONE syntactically "
    "correct SVA assertion. Emit ONLY the SVA — no explanation. Match "
    "temporal complexity: bare `##N` for fixed delays, `##[a:b]` for ranged, "
    "`|->`/`|=>` only when antecedent-consequent, `s_eventually`/`s_until` "
    "for liveness."
)


def extract_sva(text):
    text = (text or "").strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "<think>" in text:
        text = text.split("<think>")[0]
    m = re.search(r"```(?:systemverilog|sv|verilog)?\s*(.+?)```", text, re.DOTALL)
    if m: text = m.group(1).strip()
    m = re.search(r"(assert\s+property\s*\(.*?\)\s*;)", text, re.DOTALL | re.IGNORECASE)
    if m: return m.group(1).strip()
    return text.splitlines()[0].strip() if text else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="base model path (SFT ckpt)")
    ap.add_argument("--adapter", required=True, help="LoRA adapter path")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tag", default="grpo")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True).to(args.device)
    model = PeftModel.from_pretrained(base, args.adapter).to(args.device)
    model.eval()

    tasks = [json.loads(l) for l in open(TEST_JSONL)]
    print(f"Eval {len(tasks)} tasks with {args.tag}")

    per_tcl = defaultdict(lambda: {"total": 0, "syn": 0, "match": 0})
    rows = []
    for i, t in enumerate(tasks):
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Generate an SVA assertion for:\n{t['nl']}"}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tok(prompt, return_tensors="pt").to(args.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=256, do_sample=False,
                                  pad_token_id=tok.eos_token_id)
        raw = tok.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True)
        sva = extract_sva(raw)
        syn = syntax_check(sva)
        gen_tcl = None
        if syn["ok"]:
            cls = classify_tcl(sva)
            gen_tcl = cls[0] if isinstance(cls, tuple) else cls
        exp = t["expected_tcl"]
        per_tcl[exp]["total"] += 1
        if syn["ok"]: per_tcl[exp]["syn"] += 1
        if gen_tcl == exp: per_tcl[exp]["match"] += 1
        rows.append({"id": t["id"], "expected_tcl": exp, "generated_sva": sva,
                      "syntax_ok": syn["ok"], "gen_tcl": gen_tcl,
                      "match": gen_tcl == exp if gen_tcl is not None else False})
        if (i + 1) % 20 == 0:
            sn = sum(v["syn"] for v in per_tcl.values())
            mn = sum(v["match"] for v in per_tcl.values())
            print(f"  {i+1}/{len(tasks)}  syn={sn} match={mn}")
    syn = sum(v["syn"] for v in per_tcl.values())
    m = sum(v["match"] for v in per_tcl.values())
    t = len(tasks)
    print(f"\n[{args.tag}] syn {syn}/{t}={100*syn/t:.1f}%  "
          f"match {m}/{t}={100*m/t:.1f}%  "
          f"per-TCL: " + "  ".join(
              f"L{lv}={per_tcl[lv]['match']}/{per_tcl[lv]['total']}"
              for lv in sorted(per_tcl)))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = EXPERIMENTS_DIR / "results" / f"eval_grpo_{args.tag}_{ts}.json"
    json.dump({"tag": args.tag, "syntax": syn, "match": m, "total": t,
                "per_tcl": {str(k): v for k, v in per_tcl.items()},
                "rows": rows}, open(out_path, "w"), indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
