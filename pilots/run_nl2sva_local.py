#!/usr/bin/env python3
"""
run_nl2sva_local.py — NL→SVA generation with a LOCAL HuggingFace model.

Proxy for GPT-4o-mini / CodeV-SVA-14B baselines when no OPENAI_API_KEY is
available. Uses Qwen2.5-7B-Instruct (or the 0.5B fallback) loaded in bf16 on
a single GPU.

Metrics match run_nl2sva_pilot.py:
  - Syntax validity rate (per TCL level)
  - TCL level match rate (did model pick the right complexity?)
  - Classified-TCL distribution

Output: results/nl2sva_local_<timestamp>.json
"""

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tcl import classify_tcl
from src.mock_verifier import syntax_check

EXPERIMENTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(EXPERIMENTS_DIR, "results")

SYSTEM_PROMPT = (
    "You are an expert in SystemVerilog Assertions (SVA). "
    "Generate ONE syntactically correct SVA assertion for the given natural "
    "language specification. Output ONLY the SVA code — no explanation, no "
    "markdown fences. Use the `assert property (@(posedge clk) ...);` form. "
    "Match the temporal complexity implied by the specification: use bare "
    "`##N` for simple fixed delays, `##[a:b]` for ranged delays, `|->`/`|=>` "
    "only when the spec genuinely has an antecedent-consequent structure, and "
    "`s_eventually`/`s_until` for liveness."
)


def extract_sva(text: str) -> str:
    text = text.strip()
    # strip markdown fences if present
    m = re.search(r"```(?:systemverilog|sv|verilog)?\s*(.+?)```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    # keep only the first assert-property statement terminated by ';'
    m = re.search(r"(assert\s+property\s*\(.*?\)\s*;)", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # fallback: first line
    return text.splitlines()[0].strip() if text else ""


def load_model(model_path: str, device: str, dtype_str: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype_str]
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    return tok, model


def generate(tok, model, system_prompt: str, user_msg: str, max_new_tokens: int = 128):
    import torch

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
    ]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            pad_token_id=tok.eos_token_id,
        )
    gen_ids = out[0, inputs["input_ids"].shape[1]:]
    return tok.decode(gen_ids, skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/ssd2/xiacong/models/Qwen__Qwen2.5-7B-Instruct")
    ap.add_argument("--tasks", default=os.path.join(EXPERIMENTS_DIR, "data", "nl2sva_tasks_expanded.json"))
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--max-new-tokens", type=int, default=128)
    args = ap.parse_args()

    with open(args.tasks) as f:
        data = json.load(f)
    tasks = data["tasks"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = os.path.basename(args.model.rstrip("/"))

    print("=" * 60)
    print(f"NL2SVA Local Pilot — model={model_name}")
    print(f"Tasks: {len(tasks)}   Device: {args.device}   dtype: {args.dtype}")
    print("=" * 60)

    t0 = time.time()
    tok, model = load_model(args.model, args.device, args.dtype)
    print(f"Model loaded in {time.time()-t0:.1f}s")

    per_tcl = defaultdict(lambda: {"total": 0, "syntax_ok": 0, "tcl_match": 0})
    gen_tcl_distribution = defaultdict(int)
    results = []

    for t in tasks:
        user_msg = f"Generate an SVA assertion for: {t['nl']}"
        raw = generate(tok, model, SYSTEM_PROMPT, user_msg, args.max_new_tokens)
        sva = extract_sva(raw)
        syn = syntax_check(sva)
        if syn["ok"]:
            cls = classify_tcl(sva)
            gen_tcl = cls[0] if isinstance(cls, tuple) else cls
            gen_tcl_reason = cls[1] if isinstance(cls, tuple) else None
        else:
            gen_tcl = None
            gen_tcl_reason = None
        match = (gen_tcl == t["expected_tcl"]) if gen_tcl is not None else False

        per_tcl[t["expected_tcl"]]["total"] += 1
        if syn["ok"]:
            per_tcl[t["expected_tcl"]]["syntax_ok"] += 1
        if match:
            per_tcl[t["expected_tcl"]]["tcl_match"] += 1
        if gen_tcl is not None:
            gen_tcl_distribution[gen_tcl] += 1

        print(f"[{t['id']}] expected=L{t['expected_tcl']}  gen=L{gen_tcl}  "
              f"syn={syn['ok']}  match={match}")
        print(f"       SVA: {sva[:100]}")
        results.append({
            "id": t["id"],
            "nl": t["nl"],
            "expected_tcl": t["expected_tcl"],
            "raw_output": raw,
            "generated_sva": sva,
            "syntax_ok": syn["ok"],
            "syntax_reason": syn.get("reason"),
            "gen_tcl": gen_tcl,
            "gen_tcl_reason": gen_tcl_reason,
            "tcl_match": match,
        })

    total = len(tasks)
    syntax_ok = sum(1 for r in results if r["syntax_ok"])
    tcl_match = sum(1 for r in results if r["tcl_match"])
    summary = {
        "timestamp": timestamp,
        "model": model_name,
        "model_path": args.model,
        "tasks_file": args.tasks,
        "total_tasks": total,
        "syntax_valid_count": syntax_ok,
        "syntax_valid_rate": syntax_ok / total,
        "tcl_match_count": tcl_match,
        "tcl_match_rate": tcl_match / total,
        "per_tcl_stats": {str(k): v for k, v in per_tcl.items()},
        "gen_tcl_distribution": {str(k): v for k, v in gen_tcl_distribution.items()},
        "results": results,
    }

    out_path = os.path.join(RESULTS_DIR, f"nl2sva_local_{timestamp}.json")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Syntax valid: {syntax_ok}/{total} = {100*syntax_ok/total:.1f}%")
    print(f"TCL match:    {tcl_match}/{total} = {100*tcl_match/total:.1f}%")
    for lvl in sorted(per_tcl):
        s = per_tcl[lvl]
        print(f"  L{lvl}: syn {s['syntax_ok']}/{s['total']}  "
              f"match {s['tcl_match']}/{s['total']}")
    print(f"\nGenerated TCL distribution: "
          f"{dict(sorted(gen_tcl_distribution.items()))}")
    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()
