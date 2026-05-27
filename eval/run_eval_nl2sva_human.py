#!/usr/bin/env python3
"""
run_eval_nl2sva_human.py — zero-shot eval on the held-out NL2SVA-Human test
set for any local HuggingFace causal-LM checkpoint (Qwen, CodeV-SVA, our own
fine-tuned checkpoints, etc.).

Metrics:
  - Syntax validity rate (overall + per TCL level)
  - TCL-level match rate    (correct complexity class?)
  - Functional match proxy   (tokenized body equals reference_sva after norm)

Results written to:
  results/eval_nl2sva_human_<model>_<ts>.json

Usage:
  python scripts/run_eval_nl2sva_human.py \\
      --model ${TEACHER_MODEL} \\
      --device cuda:1 \\
      --dtype bf16
"""
import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tcl import classify_tcl
from src.mock_verifier import syntax_check

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent
TEST_JSONL = EXPERIMENTS_DIR / "data" / "test" / "nl2sva_human.jsonl"
RESULTS_DIR = EXPERIMENTS_DIR / "results"

SYSTEM_PROMPT = (
    "You are an AI assistant tasked with formal verification of register "
    "transfer level (RTL) designs. Your job is to translate a description "
    "of an assertion into a concrete SystemVerilog Assertion (SVA) "
    "implementation. Match temporal complexity: bare `##N` for fixed "
    "delays, `##[a:b]` for ranged delays, `|->` / `|=>` for "
    "antecedent-consequent implication, `s_eventually` / `s_until` for "
    "liveness."
)

USER_POSTAMBLE = (
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


def build_user_prompt(nl: str, rtl_context: str) -> str:
    parts = []
    if rtl_context.strip():
        parts.append(
            "Here is the testbench to perform your translation:\n"
            f"{rtl_context.strip()}"
        )
    # `nl` is already wrapped by fetch_benchmarks.py with the FVEval
    # "Create a SVA assertion that checks: ..." preamble. Older JSONLs may
    # still hold the bare fragment, so add the preamble defensively.
    nl = nl.strip()
    if not re.match(
        r"(?i)^\s*(create|generate|write)\s+(an?\s+|the\s+)?sva\b", nl
    ):
        nl = f"Create a SVA assertion that checks: {nl}"
    parts.append(f"Question: {nl}")
    parts.append(USER_POSTAMBLE)
    return "\n\n".join(parts)


def extract_sva(text: str) -> str:
    text = text.strip()
    # Strip reasoning-model thinking blocks (CodeV-SVA / DeepSeek-R1 style).
    # Handle both closed and open-ended <think>...</think>.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # If the <think> block is unterminated (generation truncated), drop
    # everything before the last </think> OR from first <think> onward.
    if "<think>" in text:
        text = text.split("<think>")[0] + "\n" + text.split("</think>")[-1] \
            if "</think>" in text else text.split("<think>")[0]
    text = text.strip()
    m = re.search(r"```(?:systemverilog|sv|verilog)?\s*(.+?)```",
                  text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    m = re.search(r"(assert\s+property\s*\(.*?\)\s*;)", text,
                  re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text.splitlines()[0].strip() if text else ""


def norm_body(sva: str) -> str:
    return re.sub(r"\s+", " ", sva).strip()


def load_model(path: str, device: str, dtype_str: str, adapter: str = ""):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
             "fp32": torch.float32}[dtype_str]
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=dtype, low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).to(device)
    model.eval()
    return tok, model


def generate(tok, model, system: str, user: str, max_new: int):
    import torch
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": user}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new, do_sample=False,
            temperature=1.0, top_p=1.0, pad_token_id=tok.eos_token_id,
        )
    gen = out[0, inputs["input_ids"].shape[1]:]
    return tok.decode(gen, skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to HF checkpoint")
    ap.add_argument("--adapter", default="",
                    help="optional LoRA adapter path (PEFT)")
    ap.add_argument("--tag", default="",
                    help="override the model basename in output filename")
    ap.add_argument("--tasks", default=str(TEST_JSONL),
                    help="JSONL of test samples (default: NL2SVA-Human 79)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bf16",
                    choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--max-new-tokens", type=int, default=4096,
                    help="default 4096 — reasoning models (CodeV-SVA, R1) "
                         "easily exceed 1k tokens inside <think>...</think>")
    ap.add_argument("--limit", type=int, default=0,
                    help="debug: only first N tasks")
    ap.add_argument("--rerun-empty", default="",
                    help="path to existing eval JSON; only re-run tasks "
                         "whose generated_sva is empty (truncation rescue)")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model_name = args.tag or os.path.basename(args.model.rstrip("/"))
    if args.adapter and not args.tag:
        model_name = f"{model_name}+{os.path.basename(args.adapter.rstrip('/'))}"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"eval_nl2sva_human_{model_name}_{ts}.json"

    # Load tasks
    tasks = []
    with open(args.tasks) as f:
        for line in f:
            tasks.append(json.loads(line))
    if args.limit:
        tasks = tasks[:args.limit]

    # --rerun-empty: load prior eval, restrict tasks to ones with empty SVA
    prior_eval = None
    if args.rerun_empty:
        prior_eval = json.load(open(args.rerun_empty))
        empty_ids = {s["id"] for s in prior_eval["per_sample"]
                     if not s.get("generated_sva", "").strip()}
        tasks = [t for t in tasks if t["id"] in empty_ids]
        print(f"[rerun-empty] {len(tasks)} EMPTY tasks identified from "
              f"{args.rerun_empty}")

    print("=" * 60)
    print(f"NL2SVA-Human zero-shot eval — model={model_name}")
    print(f"Tasks: {len(tasks)}   Device: {args.device}   dtype: {args.dtype}")
    print("=" * 60)

    t0 = time.time()
    tok, model = load_model(args.model, args.device, args.dtype, args.adapter)
    print(f"Model loaded in {time.time()-t0:.1f}s"
          + (f" (+ adapter {args.adapter})" if args.adapter else ""))

    per_tcl = defaultdict(lambda: {
        "total": 0, "syntax_ok": 0, "tcl_match": 0, "body_match": 0,
    })
    gen_tcl_dist = defaultdict(int)
    records = []
    t_gen = time.time()

    for i, t in enumerate(tasks):
        nl = t.get("nl", "").strip()
        ref = t.get("reference_sva", "").strip()
        exp_tcl = t.get("expected_tcl", 0)
        user = build_user_prompt(nl, t.get("rtl_context", ""))
        raw = generate(tok, model, SYSTEM_PROMPT, user, args.max_new_tokens)
        sva = extract_sva(raw)
        syn = syntax_check(sva)
        if syn["ok"]:
            cls = classify_tcl(sva)
            gen_tcl = cls[0] if isinstance(cls, tuple) else cls
        else:
            gen_tcl = None

        body_match = (norm_body(sva) == norm_body(ref)) if sva and ref else False
        tcl_match = (gen_tcl == exp_tcl) if gen_tcl is not None else False

        per_tcl[exp_tcl]["total"] += 1
        if syn["ok"]:
            per_tcl[exp_tcl]["syntax_ok"] += 1
        if tcl_match:
            per_tcl[exp_tcl]["tcl_match"] += 1
        if body_match:
            per_tcl[exp_tcl]["body_match"] += 1
        if gen_tcl:
            gen_tcl_dist[gen_tcl] += 1

        records.append({
            "id": t.get("id"), "nl": nl, "reference_sva": ref,
            "generated_raw": raw, "generated_sva": sva,
            "syntax_ok": syn["ok"], "syntax_issues": syn.get("issues", []),
            "expected_tcl": exp_tcl, "generated_tcl": gen_tcl,
            "tcl_match": tcl_match, "body_match": body_match,
        })
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(tasks)}] "
                  f"elapsed={time.time()-t_gen:.0f}s  "
                  f"syntax={sum(v['syntax_ok'] for v in per_tcl.values())}, "
                  f"tcl_match={sum(v['tcl_match'] for v in per_tcl.values())}")

    # Aggregate
    total = len(tasks)
    syn_total = sum(v["syntax_ok"] for v in per_tcl.values())
    match_total = sum(v["tcl_match"] for v in per_tcl.values())
    body_total = sum(v["body_match"] for v in per_tcl.values())

    # In rerun-empty mode, MERGE new records into the prior eval and recompute
    # aggregate stats over the full task set, rather than only the EMPTY subset.
    if prior_eval is not None:
        new_by_id = {r["id"]: r for r in records}
        merged = []
        for s in prior_eval["per_sample"]:
            merged.append(new_by_id.get(s["id"], s))
        records = merged
        # Recompute aggregates on full merged set
        per_tcl = defaultdict(lambda: {
            "total": 0, "syntax_ok": 0, "tcl_match": 0, "body_match": 0,
        })
        gen_tcl_dist = defaultdict(int)
        for r in records:
            t_lvl = r.get("expected_tcl", 0)
            per_tcl[t_lvl]["total"] += 1
            if r.get("syntax_ok"):
                per_tcl[t_lvl]["syntax_ok"] += 1
            if r.get("tcl_match"):
                per_tcl[t_lvl]["tcl_match"] += 1
            if r.get("body_match"):
                per_tcl[t_lvl]["body_match"] += 1
            if r.get("generated_tcl"):
                gen_tcl_dist[r["generated_tcl"]] += 1
        total = len(records)
        syn_total = sum(v["syntax_ok"] for v in per_tcl.values())
        match_total = sum(v["tcl_match"] for v in per_tcl.values())
        body_total = sum(v["body_match"] for v in per_tcl.values())

    report = {
        "model": model_name,
        "model_path": args.model,
        "test_set": str(args.tasks),
        "num_tasks": total,
        "overall": {
            "syntax_ok_pct": round(100 * syn_total / max(total, 1), 2),
            "tcl_match_pct": round(100 * match_total / max(total, 1), 2),
            "body_match_pct": round(100 * body_total / max(total, 1), 2),
        },
        "per_tcl": {str(lv): {
            **v,
            "syntax_pct":  round(100 * v["syntax_ok"] / max(v["total"], 1), 1),
            "match_pct":   round(100 * v["tcl_match"] / max(v["total"], 1), 1),
            "body_match_pct": round(100 * v["body_match"] / max(v["total"], 1), 1),
        } for lv, v in sorted(per_tcl.items())},
        "generated_tcl_distribution": dict(sorted(gen_tcl_dist.items())),
        "wallclock_seconds": round(time.time() - t_gen, 1),
        "per_sample": records,
    }
    if prior_eval is not None:
        report["rerun_empty_source"] = args.rerun_empty
        report["max_new_tokens_used"] = args.max_new_tokens
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print()
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Overall syntax OK:  {report['overall']['syntax_ok_pct']:.1f}%")
    print(f"Overall TCL match:  {report['overall']['tcl_match_pct']:.1f}%")
    print(f"Overall body match: {report['overall']['body_match_pct']:.1f}%")
    print()
    print("Per-TCL:")
    for lv, v in report["per_tcl"].items():
        print(f"  L{lv}: total={v['total']:>3}  "
              f"syn={v['syntax_ok']}/{v['total']} ({v['syntax_pct']:.0f}%)  "
              f"tcl={v['tcl_match']}/{v['total']} ({v['match_pct']:.0f}%)  "
              f"body={v['body_match']}/{v['total']} ({v['body_match_pct']:.0f}%)")
    print()
    print(f"Generated TCL dist: {report['generated_tcl_distribution']}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
