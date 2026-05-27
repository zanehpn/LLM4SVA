#!/usr/bin/env python3
"""
run_funcatk_eval.py — FVEval-style Func@k evaluation with vLLM generation
and PEC scoring.

Main metric:
  - Func@k (strict): pass@k over PEC EQUIVALENT candidates
  - Func@k (relaxed): pass@k over PEC EQUIVALENT or one-sided implication

Diagnostic metric:
  - Greedy syntax validity
  - Greedy TCL match

This keeps TCL match as a shape/dispatch diagnostic while making Func@k the
primary headline metric, aligned with FVEval-style sampling.
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DEFAULT_TASKS = ROOT / "data" / "test" / "nl2sva_human.jsonl"
DEFAULT_COVERAGE = ROOT / "results" / "pec_coverage_check.json"
RESULTS_DIR = ROOT / "results"

from src.mock_verifier import syntax_check
from src.pec_yosys import prop_equivalence, infer_reset_expr
from src.tcl import classify_tcl

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
        parts.append(
            "Here is the testbench to perform your translation:\n"
            + rtl_context.strip()
        )
    nl = nl.strip()
    if not re.match(r"(?i)^\s*(create|generate|write)\s+(an?\s+|the\s+)?sva\b", nl):
        nl = f"Create a SVA assertion that checks: {nl}"
    parts.append(f"Question: {nl}")
    parts.append(FVEVAL_USER_POSTAMBLE)
    return "\n\n".join(parts)


def build_prompt(task: dict, prompt_format: str) -> list[dict]:
    if prompt_format == "fveval":
        return [
            {"role": "system", "content": FVEVAL_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_fveval_user_prompt(
                    task.get("nl", ""), task.get("rtl_context", "")
                ),
            },
        ]
    return [
        {"role": "system", "content": SIMPLE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Generate an SVA assertion for:\n{task.get('nl', '').strip()}",
        },
    ]


def apply_chat_template(prompts: list[list[dict]], tokenizer) -> list[str]:
    rendered = []
    for msgs in prompts:
        rendered.append(
            tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    return rendered


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


def load_coverage_map(tasks_path: Path, coverage_path: str) -> dict[str, bool]:
    if coverage_path:
        cov_path = Path(coverage_path)
    elif tasks_path.resolve() == DEFAULT_TASKS.resolve() and DEFAULT_COVERAGE.exists():
        cov_path = DEFAULT_COVERAGE
    else:
        return {}
    if not cov_path.exists():
        return {}
    cov = json.load(open(cov_path))
    return {r["id"]: r.get("verdict") == "EQUIVALENT" for r in cov.get("rows", [])}


def pass_at_k(n: int, c: int, k: int) -> float:
    if c <= 0:
        return 0.0
    if k >= n:
        return 1.0 if c > 0 else 0.0
    if n - c < k:
        return 1.0
    return 1.0 - (math.comb(n - c, k) / math.comb(n, k))


def _pec_work(args):
    (task_idx, cand_idx, gen_sva, ref_sva, rtl, depth, timeout,
     reset_expr, liveness_bound) = args
    if not gen_sva.strip():
        return task_idx, cand_idx, "EMPTY", "", "", 0.0
    r = prop_equivalence(gen_sva, ref_sva, rtl,
                         depth=depth, timeout=timeout,
                         reset_expr=reset_expr,
                         liveness_bound=liveness_bound)
    return (
        task_idx,
        cand_idx,
        r.verdict,
        r.fwd_status,
        r.bwd_status,
        round(r.wallclock_s, 2),
    )


def run_greedy_diagnostic(
    llm: LLM,
    prompts: list[list[dict]],
    tasks: list[dict],
    adapter: str,
    max_new_tokens: int,
) -> dict:
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_new_tokens)
    lora_req = LoRARequest("eval_adapter", 1, adapter) if adapter else None
    outputs = llm.generate(prompts, sp, lora_request=lora_req)

    per_tcl = defaultdict(lambda: {"total": 0, "syntax_ok": 0, "tcl_match": 0})
    rows = []
    for task, out in zip(tasks, outputs):
        raw = out.outputs[0].text if out.outputs else ""
        sva = extract_sva(raw)
        syn = syntax_check(sva)
        gen_tcl = None
        if syn["ok"]:
            cls = classify_tcl(sva)
            gen_tcl = cls[0] if isinstance(cls, tuple) else cls
        exp_tcl = int(task.get("expected_tcl", 0))
        match = gen_tcl == exp_tcl if gen_tcl is not None else False
        per_tcl[exp_tcl]["total"] += 1
        if syn["ok"]:
            per_tcl[exp_tcl]["syntax_ok"] += 1
        if match:
            per_tcl[exp_tcl]["tcl_match"] += 1
        rows.append(
            {
                "id": task.get("id"),
                "generated_sva": sva,
                "syntax_ok": syn["ok"],
                "expected_tcl": exp_tcl,
                "generated_tcl": gen_tcl,
                "tcl_match": match,
            }
        )

    total = len(tasks)
    syntax = sum(v["syntax_ok"] for v in per_tcl.values())
    match = sum(v["tcl_match"] for v in per_tcl.values())
    return {
        "total": total,
        "syntax_ok": syntax,
        "syntax_ok_pct": round(100 * syntax / max(total, 1), 2),
        "tcl_match": match,
        "tcl_match_pct": round(100 * match / max(total, 1), 2),
        "per_tcl": {
            str(k): {
                **v,
                "syntax_pct": round(100 * v["syntax_ok"] / max(v["total"], 1), 2),
                "tcl_match_pct": round(100 * v["tcl_match"] / max(v["total"], 1), 2),
            }
            for k, v in sorted(per_tcl.items())
        },
        "rows": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default="")
    ap.add_argument("--tasks", default=str(DEFAULT_TASKS))
    ap.add_argument("--prompt-format", default="fveval", choices=["simple", "fveval"])
    ap.add_argument("--filter-tcls", default="", help="comma-separated TCL levels")
    ap.add_argument("--num-samples", type=int, default=32)
    ap.add_argument("--ks", default="1,16,32")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--depth", type=int, default=15)
    ap.add_argument("--timeout", type=int, default=30)
    ap.add_argument(
        "--liveness-bound",
        type=int,
        default=0,
        help="Bounded-liveness rewrite bound for `s_eventually` / `nexttime` "
        "/ `s_always`. Paper §4.3 / App. E specify that liveness rollouts "
        "should be returned as UNSUPPORTED so the open PEC is sound. 0 "
        "(default) keeps UNSUPPORTED on liveness; -1 falls back to --depth; "
        "any positive value sets an explicit cycle bound and breaks the "
        "soundness claim — use only for ablation.",
    )
    ap.add_argument(
        "--pec-reset-mode",
        default="auto",
        help="Cadence-aligned reset canonicalization for PEC. "
        "'off' keeps strict LRM semantics; 'auto' infers the reset signal "
        "per task (disable-iff → assign-alias → common names); "
        "'force:<expr>' pins a single expression for every task.",
    )
    ap.add_argument("--coverage-json", default="")
    ap.add_argument("--skip-coverage-check", action="store_true")
    ap.add_argument("--skip-greedy-diagnostic", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    ks = sorted({int(x) for x in args.ks.split(",") if x.strip()})
    filter_tcls = {int(x) for x in args.filter_tcls.split(",") if x.strip()}
    tasks_path = Path(args.tasks)
    tasks = load_tasks(tasks_path, filter_tcls)
    prompt_msgs = [build_prompt(t, args.prompt_format) for t in tasks]
    coverage_map = {} if args.skip_coverage_check else load_coverage_map(tasks_path, args.coverage_json)

    model_name = args.tag or os.path.basename(args.model.rstrip("/"))
    if args.adapter and not args.tag:
        model_name = f"{model_name}+{os.path.basename(args.adapter.rstrip('/'))}"

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompts = apply_chat_template(prompt_msgs, tokenizer)

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enable_lora=bool(args.adapter),
    )
    lora_req = LoRARequest("eval_adapter", 1, args.adapter) if args.adapter else None

    print("=" * 60)
    print(f"Func@k eval — model={model_name}")
    print(f"tasks={len(tasks)} prompt={args.prompt_format} n={args.num_samples} ks={ks}")
    print("=" * 60)

    diagnostic = None
    if not args.skip_greedy_diagnostic:
        print("[diag] running greedy syntax/TCL diagnostic...")
        diagnostic = run_greedy_diagnostic(
            llm=llm,
            prompts=prompts,
            tasks=tasks,
            adapter=args.adapter,
            max_new_tokens=args.max_new_tokens,
        )
        print(
            "[diag] syntax="
            f"{diagnostic['syntax_ok']}/{diagnostic['total']} "
            f"({diagnostic['syntax_ok_pct']:.1f}%)  "
            f"tcl_match={diagnostic['tcl_match']}/{diagnostic['total']} "
            f"({diagnostic['tcl_match_pct']:.1f}%)"
        )

    print("[sample] generating stochastic candidates...")
    sp = SamplingParams(
        n=args.num_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
    )
    outputs = llm.generate(prompts, sp, lora_request=lora_req)

    # Resolve the PEC reset expression once per task, up front, so the
    # worker pool can run without re-parsing rtl/ref on each candidate and
    # so the eventual result JSON can pin down exactly what each task was
    # canonicalized with.
    reset_mode = args.pec_reset_mode.strip()
    forced_reset_expr = None
    if reset_mode.startswith("force:"):
        forced_reset_expr = reset_mode.split(":", 1)[1].strip() or None
        reset_mode_label = f"force:{forced_reset_expr}"
    elif reset_mode == "off":
        reset_mode_label = "off"
    elif reset_mode == "auto":
        reset_mode_label = "auto"
    else:
        raise SystemExit(
            f"--pec-reset-mode must be 'off', 'auto', or 'force:<expr>'; "
            f"got {reset_mode!r}"
        )

    task_reset_expr: list[str | None] = []
    for task in tasks:
        if reset_mode == "off":
            task_reset_expr.append(None)
        elif forced_reset_expr is not None:
            task_reset_expr.append(forced_reset_expr)
        else:
            task_reset_expr.append(
                infer_reset_expr(
                    task.get("rtl_context", ""),
                    task.get("reference_sva", ""),
                    "",  # lm SVA not yet generated; disable-iff hint comes from ref
                )
            )
    resolved_any = sum(1 for r in task_reset_expr if r)
    print(
        f"[pec] reset_mode={reset_mode_label} "
        f"({resolved_any}/{len(tasks)} tasks received a reset expression)"
    )

    # Resolve liveness_bound: -1 means "use depth". 0 disables rewrite
    # (keep UNSUPPORTED). Positive = explicit.
    if args.liveness_bound < 0:
        liveness_bound = args.depth
    elif args.liveness_bound == 0:
        liveness_bound = None
    else:
        liveness_bound = args.liveness_bound
    print(
        f"[pec] liveness_bound="
        f"{liveness_bound if liveness_bound is not None else 'off (legacy UNSUPPORTED)'}"
    )

    candidate_rows = []
    pec_work = []
    syntax_counts = Counter()
    evaluable_total = 0
    evaluable_by_tcl = Counter()
    total_by_tcl = Counter()

    for task_idx, (task, out_group) in enumerate(zip(tasks, outputs)):
        exp_tcl = int(task.get("expected_tcl", 0))
        total_by_tcl[exp_tcl] += 1
        evaluable = coverage_map.get(task.get("id"), True) if coverage_map else True
        if evaluable:
            evaluable_total += 1
            evaluable_by_tcl[exp_tcl] += 1
        cands = []
        for cand_idx, cand in enumerate(out_group.outputs):
            raw = cand.text
            sva = extract_sva(raw)
            syn = syntax_check(sva)
            gen_tcl = None
            if syn["ok"]:
                cls = classify_tcl(sva)
                gen_tcl = cls[0] if isinstance(cls, tuple) else cls
            cands.append(
                {
                    "sample_idx": cand_idx,
                    "generated_raw": raw,
                    "generated_sva": sva,
                    "syntax_ok": syn["ok"],
                    "expected_tcl": exp_tcl,
                    "generated_tcl": gen_tcl,
                    "tcl_match": gen_tcl == exp_tcl if gen_tcl is not None else False,
                }
            )
            syntax_counts["total"] += 1
            syntax_counts["ok"] += int(syn["ok"])
            if evaluable:
                pec_work.append(
                    (
                        task_idx,
                        cand_idx,
                        sva,
                        task.get("reference_sva", ""),
                        task.get("rtl_context", ""),
                        args.depth,
                        args.timeout,
                        task_reset_expr[task_idx],
                        liveness_bound,
                    )
                )
        candidate_rows.append(
            {
                "id": task.get("id"),
                "nl": task.get("nl", ""),
                "expected_tcl": exp_tcl,
                "evaluable": evaluable,
                "pec_reset_expr": task_reset_expr[task_idx],
                "candidates": cands,
            }
        )

    print(
        "[sample] syntax-ok candidates="
        f"{syntax_counts['ok']}/{syntax_counts['total']} "
        f"({100*syntax_counts['ok']/max(syntax_counts['total'],1):.1f}%)"
    )
    print(
        "[pec] evaluable tasks="
        f"{evaluable_total}/{len(tasks)}"
        + (" (coverage filtered)" if coverage_map else " (all tasks)")
    )

    verdict_counts = Counter()
    if pec_work:
        with mp.Pool(args.workers) as pool:
            for done, (task_idx, cand_idx, verdict, fwd, bwd, dt) in enumerate(
                pool.imap_unordered(_pec_work, pec_work, chunksize=2), 1
            ):
                candidate_rows[task_idx]["candidates"][cand_idx]["pec_verdict"] = verdict
                candidate_rows[task_idx]["candidates"][cand_idx]["pec_fwd"] = fwd
                candidate_rows[task_idx]["candidates"][cand_idx]["pec_bwd"] = bwd
                candidate_rows[task_idx]["candidates"][cand_idx]["pec_seconds"] = dt
                verdict_counts[verdict] += 1
                if done % 50 == 0 or done == len(pec_work):
                    print(f"[pec] {done}/{len(pec_work)} verdicts: {dict(verdict_counts)}")

    strict_sum = Counter()
    relaxed_sum = Counter()
    strict_by_tcl = defaultdict(Counter)
    relaxed_by_tcl = defaultdict(Counter)

    for row in candidate_rows:
        if not row["evaluable"]:
            continue
        exp_tcl = row["expected_tcl"]
        strict_c = sum(1 for c in row["candidates"] if c.get("pec_verdict") == "EQUIVALENT")
        relaxed_c = sum(
            1
            for c in row["candidates"]
            if c.get("pec_verdict") in ("EQUIVALENT", "IMPLIES_REF_TO_LM", "IMPLIES_LM_TO_REF")
        )
        row["strict_correct"] = strict_c
        row["relaxed_correct"] = relaxed_c
        for k in ks:
            strict_sum[k] += pass_at_k(args.num_samples, strict_c, k)
            relaxed_sum[k] += pass_at_k(args.num_samples, relaxed_c, k)
            strict_by_tcl[exp_tcl][k] += pass_at_k(args.num_samples, strict_c, k)
            relaxed_by_tcl[exp_tcl][k] += pass_at_k(args.num_samples, relaxed_c, k)

    overall_func = {
        f"func@{k}": round(100 * strict_sum[k] / max(evaluable_total, 1), 2) for k in ks
    }
    overall_relaxed = {
        f"func_relaxed@{k}": round(100 * relaxed_sum[k] / max(evaluable_total, 1), 2)
        for k in ks
    }
    per_tcl = {}
    for tcl in sorted(total_by_tcl):
        denom = evaluable_by_tcl.get(tcl, 0)
        per_tcl[str(tcl)] = {
            "total": total_by_tcl[tcl],
            "evaluable": denom,
            **{
                f"func@{k}": round(100 * strict_by_tcl[tcl][k] / max(denom, 1), 2)
                for k in ks
            },
            **{
                f"func_relaxed@{k}": round(100 * relaxed_by_tcl[tcl][k] / max(denom, 1), 2)
                for k in ks
            },
        }

    result = {
        "model": model_name,
        "model_path": args.model,
        "adapter": args.adapter,
        "tasks_path": str(tasks_path),
        "prompt_format": args.prompt_format,
        "num_tasks": len(tasks),
        "num_samples": args.num_samples,
        "ks": ks,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "coverage_filtered": bool(coverage_map) and not args.skip_coverage_check,
        "pec_reset_mode": reset_mode_label,
        "pec_reset_resolved": resolved_any,
        "pec_liveness_bound": liveness_bound,
        "evaluable_tasks": evaluable_total,
        "overall": {
            **overall_func,
            **overall_relaxed,
            "candidate_syntax_ok_pct": round(
                100 * syntax_counts["ok"] / max(syntax_counts["total"], 1), 2
            ),
        },
        "per_tcl": per_tcl,
        "diagnostic_greedy": diagnostic,
        "verdict_counts": dict(verdict_counts),
        "per_sample": candidate_rows,
    }

    out_path = (
        Path(args.output)
        if args.output
        else RESULTS_DIR
        / f"funcatk_eval_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print()
    print("=" * 60)
    print(f"Func@k — model={model_name}")
    print("=" * 60)
    print(f"Evaluable tasks: {evaluable_total}/{len(tasks)}")
    for k in ks:
        print(
            f"  Func@{k}: {overall_func[f'func@{k}']:.2f}%   "
            f"Relaxed@{k}: {overall_relaxed[f'func_relaxed@{k}']:.2f}%"
        )
    if diagnostic:
        print(
            f"Diagnostic greedy TCL match: {diagnostic['tcl_match']}/{diagnostic['total']} "
            f"({diagnostic['tcl_match_pct']:.1f}%)"
        )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
