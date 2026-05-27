#!/usr/bin/env python3
"""
build_viable_grpo_pool.py — filter the disjoint GRPO pool to prompts where
the starting SFT-replay50 policy can already produce at least one
EQUIVALENT or IMPLIES_* SVA out of N samples.

Why: GRPO with sparse rewards collapses when most groups have zero variance.
By restricting training prompts to ones where the *base policy itself* can
sometimes hit a positive reward, every group gets non-zero advantage and
the gradient signal stays alive.

Pipeline:
  1. Subsample N_PROMPTS prompts from the disjoint GRPO pool.
  2. For each, sample N_SAMPLES SVA candidates with vLLM at temperature=0.9.
  3. Run PEC equivalence on each (candidate, reference) pair.
  4. Keep prompts where at least one candidate is EQUIVALENT (strict) or
     IMPLIES_* (relaxed).
  5. Write data/train/grpo/grpo_pool_viable.jsonl + manifest.

Usage:
  source ${OSS_CAD_SUITE}/environment
  CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python3 scripts/build_viable_grpo_pool.py
"""
import argparse
import json
import multiprocessing as mp
import random
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXPERIMENTS_DIR))

POOL_IN = EXPERIMENTS_DIR / "data" / "train" / "grpo" / "grpo_pool_disjoint.jsonl"
POOL_OUT_STRICT = EXPERIMENTS_DIR / "data" / "train" / "grpo" / "grpo_pool_viable.jsonl"
POOL_OUT_RELAXED = EXPERIMENTS_DIR / "data" / "train" / "grpo" / "grpo_pool_viable_relaxed.jsonl"


SYSTEM_PROMPT = (
    "You are an expert in SystemVerilog Assertions (SVA). Given a "
    "natural-language description of a design property, output ONE "
    "syntactically correct SVA assertion. Emit ONLY the SVA — no "
    "explanation. Match temporal complexity: bare `##N` for fixed delays, "
    "`##[a:b]` for ranged, `|->`/`|=>` only when antecedent-consequent, "
    "`s_eventually`/`s_until` for liveness."
)


def extract_sva(text):
    text = (text or "").strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "<think>" in text:
        text = text.split("<think>")[0]
    m = re.search(r"```(?:systemverilog|sv|verilog)?\s*(.+?)```",
                  text, re.DOTALL)
    if m: text = m.group(1).strip()
    m = re.search(r"(assert\s+property\s*\(.*?\)\s*;)", text,
                  re.DOTALL | re.IGNORECASE)
    if m: return m.group(1).strip()
    return text.splitlines()[0].strip() if text else ""


def free_input_rtl(reference_sva):
    """Build a synthetic free-input module covering all identifiers in ref."""
    KEYWORDS = {
        "assert", "property", "posedge", "negedge", "disable", "iff", "if",
        "else", "begin", "end", "always", "always_ff", "always_comb", "logic",
        "wire", "reg", "module", "endmodule", "input", "output", "and", "or",
        "not", "throughout", "within", "intersect", "first_match", "until",
        "until_with", "s_until", "s_eventually", "s_always", "nexttime",
        "strong", "weak", "1", "0", "1'b0", "1'b1",
        "rose", "fell", "stable", "past", "changed", "sampled",
    }
    ids = set()
    for tok in re.findall(r"[A-Za-z_]\w*", reference_sva):
        if tok.lower() in KEYWORDS or tok in ("clk", "tb_reset"):
            continue
        ids.add(tok)
    decls = ["    input logic clk", "    input logic tb_reset"]
    for ident in sorted(ids):
        decls.append(f"    input logic [31:0] {ident}")
    return "module pec_top (\n" + ",\n".join(decls) + "\n);\nendmodule\n"


def _pec_check(args):
    idx, gen_sva, ref_sva, rtl = args
    if not gen_sva.strip():
        return idx, "EMPTY"
    from src.mock_verifier import syntax_check
    if not syntax_check(gen_sva)["ok"]:
        return idx, "PARSE_ERROR"
    try:
        from src.pec_yosys import prop_equivalence
        # Cadence-aligned: symmetric reset canonicalization + bounded
        # liveness rewrite so s_eventually/etc. don't auto-fail BMC.
        r = prop_equivalence(gen_sva, ref_sva, rtl, depth=10, timeout=15,
                             reset_expr="tb_reset", liveness_bound=15)
        return idx, r.verdict
    except Exception:
        return idx, "ERROR"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="results/sft_qwen_coder_7b_replay50/checkpoint_20260420_210826")
    ap.add_argument("--input", default=str(POOL_IN),
                    help="prompt pool jsonl to screen (rows need nl + reference_sva)")
    ap.add_argument("--output-strict", default=str(POOL_OUT_STRICT),
                    help="strict-EQUIV subset output jsonl")
    ap.add_argument("--output-relaxed", default=str(POOL_OUT_RELAXED),
                    help="relaxed (EQUIV or IMPLIES_*) subset output jsonl")
    ap.add_argument("--manifest", default="",
                    help="manifest json path (default: data/train/grpo/viable_manifest.json)")
    ap.add_argument("--n-prompts", type=int, default=2000,
                    help="random subsample size; 0 = no subsample (use all)")
    ap.add_argument("--n-samples", type=int, default=4,
                    help="rollouts per prompt to test")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--max-new-tokens", type=int, default=192)
    ap.add_argument("--workers", type=int, default=8,
                    help="PEC parallel workers")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.40)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)

    # Load pool
    pool_in = Path(args.input)
    pool = [json.loads(l) for l in open(pool_in)]
    print(f"[load] {len(pool)} prompts from {pool_in.name}")
    if args.n_prompts and args.n_prompts < len(pool):
        pool = random.sample(pool, args.n_prompts)
        print(f"[sample] {len(pool)} prompts subsampled (seed={args.seed})")
    else:
        print(f"[sample] using all {len(pool)} prompts (no subsample)")

    # Build prompts for vLLM
    print(f"[load] policy: {args.policy}")
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.policy, trust_remote_code=True)
    llm = LLM(
        model=args.policy, trust_remote_code=True, dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    sp = SamplingParams(
        temperature=args.temperature, top_p=0.95,
        max_tokens=args.max_new_tokens, n=args.n_samples,
    )

    prompts_text = []
    for r in pool:
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",
             "content": f"Generate an SVA assertion for:\n{r['nl']}"},
        ]
        prompts_text.append(tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True))

    print(f"[generate] generating {args.n_samples} samples for "
          f"{len(prompts_text)} prompts via vLLM...")
    t0 = time.time()
    outputs = llm.generate(prompts_text, sp)
    print(f"[generate] done in {time.time()-t0:.0f}s "
          f"({(time.time()-t0)/len(pool):.2f}s/prompt)")

    # Build PEC tasks
    tasks = []
    cand_for_prompt = [[] for _ in pool]
    for pi, out in enumerate(outputs):
        for si, sample in enumerate(out.outputs):
            sva = extract_sva(sample.text)
            cand_for_prompt[pi].append(sva)
            ref = pool[pi]["reference_sva"]
            rtl = free_input_rtl(ref)
            tasks.append((len(tasks), sva, ref, rtl))

    print(f"[pec] running PEC on {len(tasks)} (gen, ref) pairs "
          f"with {args.workers} workers...")
    t0 = time.time()
    verdict_per_task = [None] * len(tasks)
    done = 0
    with mp.Pool(args.workers) as pool_p:
        for idx, verdict in pool_p.imap_unordered(_pec_check, tasks, chunksize=4):
            verdict_per_task[idx] = verdict
            done += 1
            if done % 200 == 0:
                eta = (time.time()-t0) / done * (len(tasks)-done)
                print(f"  pec {done}/{len(tasks)}  elapsed={time.time()-t0:.0f}s  eta={eta:.0f}s")
    print(f"[pec] done in {time.time()-t0:.0f}s")

    # Aggregate per prompt
    strict_keep = []
    relaxed_keep = []
    per_prompt_summary = []
    cnt_strict = cnt_relaxed = 0
    for pi, p in enumerate(pool):
        verdicts = []
        for si in range(args.n_samples):
            ti = pi * args.n_samples + si
            verdicts.append(verdict_per_task[ti])
        any_equiv = any(v == "EQUIVALENT" for v in verdicts)
        any_implies = any(v in ("EQUIVALENT", "IMPLIES_REF_TO_LM",
                                 "IMPLIES_LM_TO_REF") for v in verdicts)
        per_prompt_summary.append({
            "id": p.get("id", ""), "verdicts": verdicts,
            "any_equiv": any_equiv, "any_implies": any_implies,
        })
        if any_equiv:
            strict_keep.append(p); cnt_strict += 1
        if any_implies:
            relaxed_keep.append(p); cnt_relaxed += 1

    print(f"\n[filter] strict EQUIV-able: {cnt_strict}/{len(pool)} "
          f"({100*cnt_strict/len(pool):.1f}%)")
    print(f"[filter] relaxed (EQUIV or IMPLIES): {cnt_relaxed}/{len(pool)} "
          f"({100*cnt_relaxed/len(pool):.1f}%)")

    # Per-TCL breakdown
    by_tcl_strict = Counter(p["expected_tcl"] for p in strict_keep)
    by_tcl_relaxed = Counter(p["expected_tcl"] for p in relaxed_keep)
    print(f"[filter] strict per-TCL:  " + "  ".join(
        f"L{l}={by_tcl_strict.get(l, 0)}" for l in (1, 2, 3, 4, 5)))
    print(f"[filter] relaxed per-TCL: " + "  ".join(
        f"L{l}={by_tcl_relaxed.get(l, 0)}" for l in (1, 2, 3, 4, 5)))

    # Write pools
    out_strict = Path(args.output_strict)
    out_relaxed = Path(args.output_relaxed)
    out_strict.parent.mkdir(parents=True, exist_ok=True)
    out_relaxed.parent.mkdir(parents=True, exist_ok=True)
    with open(out_strict, "w") as f:
        for p in strict_keep:
            f.write(json.dumps(p) + "\n")
    with open(out_relaxed, "w") as f:
        for p in relaxed_keep:
            f.write(json.dumps(p) + "\n")
    print(f"\n[write] strict: {out_strict}")
    print(f"[write] relaxed: {out_relaxed}")

    manifest = {
        "policy": args.policy,
        "n_prompts_sampled": len(pool),
        "n_samples_per_prompt": args.n_samples,
        "temperature": args.temperature,
        "n_strict_equiv": cnt_strict,
        "n_relaxed": cnt_relaxed,
        "strict_per_tcl": dict(by_tcl_strict),
        "relaxed_per_tcl": dict(by_tcl_relaxed),
        "verdict_distribution": dict(Counter(verdict_per_task)),
        "ts": datetime.now().isoformat(),
    }
    out_manifest = (Path(args.manifest) if args.manifest
                    else EXPERIMENTS_DIR / "data" / "train" / "grpo" / "viable_manifest.json")
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(out_manifest, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[write] manifest: {out_manifest}")


if __name__ == "__main__":
    main()
