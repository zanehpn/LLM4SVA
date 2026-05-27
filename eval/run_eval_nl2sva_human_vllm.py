#!/usr/bin/env python3
"""
run_eval_nl2sva_human_vllm.py — vLLM-based eval on NL2SVA-Human or any JSONL
task file with schema:
  {id, nl, reference_sva, rtl_context, expected_tcl}

Supports base checkpoints and optional LoRA adapters.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
TEST_JSONL = ROOT / "data" / "test" / "nl2sva_human.jsonl"
RESULTS_DIR = ROOT / "results"

from src.tcl import classify_tcl
from src.mock_verifier import syntax_check

SIMPLE_SYSTEM_PROMPT = (
    "You are an expert in SystemVerilog Assertions (SVA). Given a "
    "natural-language description of a design property, output ONE "
    "syntactically correct SVA assertion. Emit ONLY the SVA — no explanation "
    "or markdown fences. Match temporal complexity to the spec: bare `##N` "
    "for fixed delays, `##[a:b]` for ranged, `|->`/`|=>` only when antecedent-"
    "consequent, `s_eventually`/`s_until` for liveness."
)

FVEVAL_SYSTEM_PROMPT = (
    "You are an AI assistant tasked with formal verification of register "
    "transfer level (RTL) designs. Your job is to translate a description "
    "of an assertion into a concrete SystemVerilog Assertion (SVA) "
    "implementation. Match temporal complexity: bare `##N` for fixed "
    "delays, `##[a:b]` for ranged delays, `|->` / `|=>` for "
    "antecedent-consequent implication, `s_eventually` / `s_until` for "
    "liveness."
)

FVEVAL_USER_POSTAMBLE = (
    "Do not add code to output an error message string.\n"
    "Enclose your SVA code with ```systemverilog and ```. "
    "Only output the code snippet and do NOT output anything else.\n\n"
    "For example,\n"
    "```systemverilog\n"
    "asrt: assert property (@(posedge clk) disable iff (tb_reset)\n"
    "    (a && b) != 1'b1\n"
    ");\n"
    "```\n"
    "Answer:"
)


def build_fveval_user_prompt(nl: str, rtl_context: str) -> str:
    parts = []
    if rtl_context.strip():
        parts.append("Here is the testbench to perform your translation:\n" + rtl_context.strip())
    nl = nl.strip()
    if not re.match(r"(?i)^\s*(create|generate|write)\s+(an?\s+|the\s+)?sva\b", nl):
        nl = f"Create a SVA assertion that checks: {nl}"
    parts.append(f"Question: {nl}")
    parts.append(FVEVAL_USER_POSTAMBLE)
    return "\n\n".join(parts)


def extract_sva(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "<think>" in text:
        text = text.split("<think>")[0]
    m = re.search(r"```(?:systemverilog|sv|verilog)?\s*(.+?)```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    m = re.search(r"(assert\s+property\s*\(.*?\)\s*;)", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text.splitlines()[0].strip() if text else ""


def load_tasks(path: Path, filter_tcls: set[int]) -> list[dict]:
    tasks = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            if filter_tcls and int(rec.get("expected_tcl", 0)) not in filter_tcls:
                continue
            tasks.append(rec)
    return tasks


def build_prompts(tasks: list[dict], prompt_format: str, tokenizer) -> list[str]:
    prompts = []
    for t in tasks:
        if prompt_format == "fveval":
            user = build_fveval_user_prompt(t.get("nl", ""), t.get("rtl_context", ""))
            msgs = [
                {"role": "system", "content": FVEVAL_SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ]
        else:
            msgs = [
                {"role": "system", "content": SIMPLE_SYSTEM_PROMPT},
                {"role": "user", "content": f"Generate an SVA assertion for:\n{t.get('nl', '').strip()}"},
            ]
        prompts.append(
            tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    return prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default="")
    ap.add_argument("--tasks", default=str(TEST_JSONL))
    ap.add_argument("--prompt-format", default="simple", choices=["simple", "fveval"])
    ap.add_argument("--filter-tcls", default="", help="comma-separated TCLs to keep, e.g. 1,4")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--tag", default="vllm_eval")
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    filter_tcls = {int(x) for x in args.filter_tcls.split(",") if x.strip()}
    tasks = load_tasks(Path(args.tasks), filter_tcls)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompts = build_prompts(tasks, args.prompt_format, tokenizer)

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enable_lora=bool(args.adapter),
    )
    sp = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_new_tokens,
        top_p=1.0,
    )
    lora_req = LoRARequest("eval_adapter", 1, args.adapter) if args.adapter else None
    outputs = llm.generate(prompts, sp, lora_request=lora_req)

    per_tcl = defaultdict(lambda: {"total": 0, "syntax_ok": 0, "tcl_match": 0})
    rows = []
    for t, out in zip(tasks, outputs):
        raw = out.outputs[0].text if out.outputs else ""
        sva = extract_sva(raw)
        syn = syntax_check(sva)
        gen_tcl = None
        if syn["ok"]:
            cls = classify_tcl(sva)
            gen_tcl = cls[0] if isinstance(cls, tuple) else cls
        exp_tcl = int(t.get("expected_tcl", 0))
        match = gen_tcl == exp_tcl if gen_tcl is not None else False
        per_tcl[exp_tcl]["total"] += 1
        if syn["ok"]:
            per_tcl[exp_tcl]["syntax_ok"] += 1
        if match:
            per_tcl[exp_tcl]["tcl_match"] += 1
        rows.append({
            "id": t.get("id"),
            "expected_tcl": exp_tcl,
            "generated_tcl": gen_tcl,
            "generated_sva": sva,
            "syntax_ok": syn["ok"],
            "tcl_match": match,
        })

    total = len(tasks)
    syntax = sum(v["syntax_ok"] for v in per_tcl.values())
    match = sum(v["tcl_match"] for v in per_tcl.values())
    result = {
        "tag": args.tag,
        "model": args.model,
        "adapter": args.adapter,
        "tasks": total,
        "syntax": syntax,
        "match": match,
        "total": total,
        "per_tcl": {str(k): v for k, v in per_tcl.items()},
        "rows": rows,
    }
    out_path = Path(args.output) if args.output else RESULTS_DIR / f"eval_vllm_{args.tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({
        "output": str(out_path),
        "syntax": syntax,
        "match": match,
        "total": total,
        "per_tcl": result["per_tcl"],
    }, indent=2))


if __name__ == "__main__":
    main()
