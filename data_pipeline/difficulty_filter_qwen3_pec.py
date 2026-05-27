"""difficulty_filter_qwen3_pec.py
Step (2) of the SVA training-data filtering pipeline:

> Difficulty Filtering via a Weaker LLM. We use Qwen3-8B, a general-purpose
> LLM with weak SVA generation ability, to generate 5 SVAs {y_ijk}_{k=1}^5
> for each NL x*_ij and remove the instance where all SVAs are equivalent
> to y'_ij, thereby filtering trivial data points.

Equivalence is judged by PEC (`src.pec_yosys.prop_equivalence`) — same
backend used by `run_funcatk_eval.py`.

Usage:
    python scripts/difficulty_filter_qwen3_pec.py \\
        --input  data/master/master_train_aligned.jsonl \\
        --output data/master/master_train_difficulty_filtered.jsonl \\
        --model  /ssd2/xiacong/models/Qwen_Qwen3-8B \\
        --num-samples 5 \\
        --workers 16
"""
from __future__ import annotations
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.run_funcatk_eval import build_prompt, extract_sva
from src.pec_yosys import prop_equivalence


def read_jsonl(p: Path):
    with open(p) as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def write_jsonl(p: Path, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _pec_one(args):
    (row_idx, cand_idx, gen_sva, ref_sva, rtl, depth, timeout,
     reset_expr, liveness_bound) = args
    try:
        r = prop_equivalence(
            gen_sva, ref_sva, rtl,
            depth=depth, timeout=timeout,
            reset_expr=reset_expr,
            liveness_bound=liveness_bound,
        )
        verdict = r.verdict
    except Exception as e:
        verdict = f"EXCEPTION:{type(e).__name__}"
    return row_idx, cand_idx, verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="/ssd2/xiacong/models/Qwen_Qwen3-8B")
    ap.add_argument("--prompt-format", default="fveval",
                    choices=["fveval", "simple"])
    ap.add_argument("--num-samples", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--prompt-rtl-cap", type=int, default=8000,
                    help="cap rtl_context length used for the prompt (PEC "
                         "still uses the full untruncated rtl)")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--workers", type=int, default=16,
                    help="parallel PEC workers")
    ap.add_argument("--depth", type=int, default=15)
    ap.add_argument("--timeout", type=int, default=30)
    ap.add_argument("--liveness-bound", type=int, default=15)
    ap.add_argument("--limit", type=int, default=0,
                    help="cap rows for smoke testing (0=all)")
    ap.add_argument("--gen-cache", default="",
                    help="optional .jsonl path to dump raw generations "
                         "before PEC")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    src = Path(args.input)
    dst = Path(args.output)
    rows = read_jsonl(src)
    if args.limit > 0:
        rows = rows[: args.limit]
    print(f"[input] {src}  rows={len(rows)}")

    # vLLM init
    print(f"[gen] loading {args.model} ...")
    from vllm import LLM, SamplingParams
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        disable_log_stats=True,
        enable_prefix_caching=True,
    )
    tok = llm.get_tokenizer()
    sp = SamplingParams(
        n=args.num_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
    )

    # Build chat-templated prompts (cap rtl_context for prompt only)
    # enable_thinking=False keeps Qwen3 from emitting <think>...</think>
    # reasoning before the actual answer; we want raw SVA, not chain-of-
    # thought.
    prompts = []
    template_kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        # Probe whether the chat template supports enable_thinking
        tok.apply_chat_template(
            [{"role": "user", "content": "x"}],
            **template_kwargs, enable_thinking=False,
        )
        template_kwargs["enable_thinking"] = False
        print("[gen] enable_thinking=False (Qwen3 reasoning mode disabled)")
    except TypeError:
        print("[gen] tokenizer chat template does not accept enable_thinking; "
              "proceeding with default")
    for r in rows:
        rtl = r.get("rtl_context") or ""
        truncated_rtl = rtl[: args.prompt_rtl_cap] if len(rtl) > args.prompt_rtl_cap else rtl
        prompt_row = {**r, "rtl_context": truncated_rtl}
        msgs = build_prompt(prompt_row, args.prompt_format)
        text = tok.apply_chat_template(msgs, **template_kwargs)
        prompts.append(text)

    print(f"[gen] generating {args.num_samples} samples for {len(prompts)} prompts ...")
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    print(f"[gen] done in {time.time()-t0:.1f}s")

    # Collect candidates per row
    candidates_per_row: list[list[str]] = []
    for o in outs:
        cands = []
        for c in o.outputs:
            sva = extract_sva(c.text or "")
            cands.append(sva or (c.text or "").strip())
        candidates_per_row.append(cands)

    # Optional dump
    if args.gen_cache:
        gen_path = Path(args.gen_cache)
        gen_path.parent.mkdir(parents=True, exist_ok=True)
        with open(gen_path, "w") as f:
            for r, cands in zip(rows, candidates_per_row):
                f.write(json.dumps({
                    "id": r.get("id"),
                    "candidates": cands,
                }, ensure_ascii=False) + "\n")
        print(f"[gen-cache] dumped to {gen_path}")

    # Build PEC work items
    work = []
    for ri, (r, cands) in enumerate(zip(rows, candidates_per_row)):
        rtl = r.get("rtl_context") or ""
        ref = r.get("reference_sva") or ""
        reset_expr = "tb_reset"  # rows are aligned subset; canonical
        for ci, gen in enumerate(cands):
            if not gen.strip():
                continue
            work.append((ri, ci, gen, ref, rtl,
                         args.depth, args.timeout,
                         reset_expr, args.liveness_bound))

    print(f"[pec] {len(work)} candidate verifications across {len(rows)} rows")
    verdicts: dict[int, dict[int, str]] = {}
    counts = Counter()
    t0 = time.time()
    with mp.Pool(args.workers) as pool:
        for done, (ri, ci, v) in enumerate(
                pool.imap_unordered(_pec_one, work, chunksize=4), 1):
            verdicts.setdefault(ri, {})[ci] = v
            counts[v] += 1
            if done % 200 == 0 or done == len(work):
                pct = 100 * done / max(len(work), 1)
                print(f"[pec] {done}/{len(work)}  ({pct:.1f}%)  "
                      f"verdicts={dict(counts)}  "
                      f"elapsed={time.time()-t0:.0f}s")

    # Apply filter: drop rows where ALL candidates are EQUIVALENT
    kept = []
    drop_counts = Counter()
    for ri, r in enumerate(rows):
        per_cand = verdicts.get(ri, {})
        n_eq = sum(1 for v in per_cand.values() if v == "EQUIVALENT")
        n_total = len(per_cand)
        all_equiv = (n_total > 0 and n_eq == n_total)
        if all_equiv:
            drop_counts["all_equivalent_trivial"] += 1
            continue
        # Otherwise keep, annotated
        new_r = dict(r)
        new_r["_difficulty_filter"] = {
            "n_candidates": n_total,
            "n_equivalent": n_eq,
            "verdicts": list(per_cand.values()),
        }
        kept.append(new_r)
        drop_counts["kept"] += 1

    write_jsonl(dst, kept)
    print(f"\n[difficulty_filter]")
    print(f"  input rows : {len(rows)}")
    print(f"  output rows: {len(kept)}")
    for k, c in drop_counts.most_common():
        print(f"  {k:<25} {c}")
    print(f"  verdict counts overall: {dict(counts)}")
    print(f"  output: {dst}")


if __name__ == "__main__":
    main()
