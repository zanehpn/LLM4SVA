#!/usr/bin/env python3
"""
build_phase2_pool.py — distribution-matched, multi-reference GRPO pool.

Phase 2 of the GRPO improvement plan:
  #4 distribution-matched pool:
    Source: NL2SVA-Machine (300 records). The NL phrasing style and signal
    naming match what the test set NL2SVA-Human uses, so training on this
    distribution should transfer better than the in-the-wild scraped pool.
  #2A multi-reference per prompt:
    For each (paraphrased NL, original ref_SVA) pair, ask the LLM for K
    alternative-equivalent SVAs, then PEC-verify which ones are actually
    EQUIVALENT to the original. Keep all verified equivalents alongside
    the original — GRPO's reward becomes max(PEC(gen, ref_i)) over all
    refs, so the policy isn't penalized for finding a different but
    equivalent assertion.

Pipeline stages (each writes a checkpoint file; rerun resumes from last):
  A. paraphrase: 300 NL → 300×K_NL paraphrased prompts
     out: phase2_paraphrased.jsonl
  B. alt-gen:    each prompt → K_ALT candidate SVAs
     out: phase2_alts.jsonl
  C. pec-verify: PEC each alt vs original ref → keep EQUIV / IMPLIES_LM_TO_REF
     (alt is at least as strong as ref → safe to use as additional reward
      target)
     out: phase2_pool.jsonl

Usage:
  source ${OSS_CAD_SUITE}/environment
  CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python3 scripts/build_phase2_pool.py \\
      --stage all --k-nl 10 --k-alt 4 --workers 8
"""
import argparse
import json
import multiprocessing as mp
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXPERIMENTS_DIR))

DS_CODER_V2 = (
    "/ssd2/junyi/hf_cache/hub/models--deepseek-ai--DeepSeek-Coder-V2-Lite-Instruct/"
    "snapshots/e434a23f91ba5b4923cf6c9d9a238eb4a08e3a11"
)

NL2SVA_MACHINE = EXPERIMENTS_DIR / "data" / "train" / "nl2sva_machine.jsonl"
PHASE2_DIR = EXPERIMENTS_DIR / "data" / "train" / "grpo"
F_PARAPHRASED = PHASE2_DIR / "phase2_paraphrased.jsonl"
F_ALTS = PHASE2_DIR / "phase2_alts.jsonl"
F_POOL = PHASE2_DIR / "grpo_pool_phase2.jsonl"
F_MANIFEST = PHASE2_DIR / "phase2_manifest.json"


# --- Stage A: NL paraphrase --------------------------------------------------

PARAPHRASE_SYSTEM = (
    "You are a hardware verification engineer. Given an SVA assertion and its "
    "natural-language description, write K diverse paraphrases of the "
    "description that all describe THE SAME assertion. Vary surface form "
    "(word choice, sentence structure, voice) but PRESERVE meaning exactly. "
    "Output format: K lines, each one paraphrase, no numbering, no preamble, "
    "no markdown. Use the same signal names as in the original."
)


def build_paraphrase_prompt(nl: str, sva: str, k: int) -> str:
    return (
        f"Original NL: {nl}\n\n"
        f"Reference SVA:\n```systemverilog\n{sva}\n```\n\n"
        f"Write {k} diverse paraphrases of the NL (one per line, no numbers)."
    )


def cleanup_paraphrase_block(text: str, k: int) -> list[str]:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        line = re.sub(r"^\s*[-*•]\s*", "", line)
        line = re.sub(r"^\s*\d+[.):]\s*", "", line)
        line = line.strip().strip('"').strip("'").strip("`").strip()
        if len(line) < 10:
            continue
        if line.lower().startswith(("here are", "paraphrase", "version")):
            continue
        lines.append(line)
        if len(lines) >= k:
            break
    return lines


def stage_a_paraphrase(args, llm, tok):
    print(f"[stage A] paraphrasing {NL2SVA_MACHINE.name}")
    src = [json.loads(l) for l in open(NL2SVA_MACHINE)]
    print(f"[stage A] {len(src)} source records")

    from vllm import SamplingParams
    sp = SamplingParams(
        temperature=0.85, top_p=0.95, max_tokens=args.max_paraphrase_tokens, n=1,
    )

    prompts = []
    for r in src:
        msgs = [
            {"role": "system", "content": PARAPHRASE_SYSTEM},
            {"role": "user",
             "content": build_paraphrase_prompt(r["nl"], r["reference_sva"],
                                                args.k_nl)},
        ]
        prompts.append(tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True))
    print(f"[stage A] {len(prompts)} prompts built")

    t0 = time.time()
    outputs = llm.generate(prompts, sp)
    print(f"[stage A] inference {time.time()-t0:.0f}s "
          f"({(time.time()-t0)/len(prompts):.2f}s/prompt)")

    out_records = []
    n_short = 0
    for r, out in zip(src, outputs):
        paraphrases = cleanup_paraphrase_block(
            out.outputs[0].text, args.k_nl)
        if len(paraphrases) < args.k_nl:
            n_short += 1
        # Always keep the original NL too
        all_nls = [r["nl"]] + paraphrases
        for i, nl in enumerate(all_nls):
            out_records.append({
                "id": f"{r['id']}__pp{i}",
                "src_id": r["id"],
                "nl": nl,
                "is_original_nl": (i == 0),
                "reference_sva": r["reference_sva"],
                "rtl_context": r["rtl_context"],
                "expected_tcl": r["expected_tcl"],
                "source": "phase2_paraphrased",
            })
    print(f"[stage A] under-K paraphrases: {n_short}/{len(src)}")
    print(f"[stage A] total records: {len(out_records)}")

    with open(F_PARAPHRASED, "w") as f:
        for r in out_records:
            f.write(json.dumps(r) + "\n")
    print(f"[stage A] wrote {F_PARAPHRASED}")
    return out_records


# --- Stage B: alt-SVA generation ---------------------------------------------

ALTGEN_SYSTEM = (
    "You are an expert in SystemVerilog Assertions (SVA). Given a "
    "natural-language description of a design property, output ONE "
    "syntactically correct SVA assertion. Emit ONLY the SVA — no "
    "explanation. Match temporal complexity: bare `##N` for fixed delays, "
    "`##[a:b]` for ranged, `|->`/`|=>` only when antecedent-consequent, "
    "`s_eventually`/`s_until` for liveness."
)


def extract_sva(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "<think>" in text:
        text = text.split("<think>")[0]
    m = re.search(r"```(?:systemverilog|sv|verilog)?\s*(.+?)```",
                  text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    m = re.search(r"(assert\s+property\s*\(.*?\)\s*;)", text,
                  re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text.splitlines()[0].strip() if text else ""


def stage_b_altgen(args, llm, tok, paraphrased):
    """Generate K_ALT candidate SVAs per prompt at high temperature."""
    print(f"[stage B] generating {args.k_alt} alt SVAs per prompt")

    from vllm import SamplingParams
    sp = SamplingParams(
        temperature=0.95, top_p=0.95, max_tokens=args.max_alt_tokens,
        n=args.k_alt,
    )

    prompts = []
    for r in paraphrased:
        msgs = [
            {"role": "system", "content": ALTGEN_SYSTEM},
            {"role": "user",
             "content": f"Generate an SVA assertion for:\n{r['nl']}"},
        ]
        prompts.append(tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True))
    print(f"[stage B] {len(prompts)} prompts built")

    t0 = time.time()
    outputs = llm.generate(prompts, sp)
    print(f"[stage B] inference {time.time()-t0:.0f}s "
          f"({(time.time()-t0)/len(prompts):.2f}s/prompt)")

    out_records = []
    n_empty = 0
    for r, out in zip(paraphrased, outputs):
        alts = []
        for s in out.outputs:
            sva = extract_sva(s.text)
            if sva:
                alts.append(sva)
        if not alts:
            n_empty += 1
        rec = dict(r)
        rec["alt_svas"] = alts
        out_records.append(rec)
    print(f"[stage B] empty-alt prompts: {n_empty}/{len(paraphrased)}")

    with open(F_ALTS, "w") as f:
        for r in out_records:
            f.write(json.dumps(r) + "\n")
    print(f"[stage B] wrote {F_ALTS}")
    return out_records


# --- Stage C: PEC-verify alts ------------------------------------------------

def free_input_rtl(reference_sva: str, base_rtl: str = "") -> str:
    """Synthetic free-input module covering all identifiers in ref."""
    KEYWORDS = {
        "assert", "property", "posedge", "negedge", "disable", "iff", "if",
        "else", "begin", "end", "always", "always_ff", "always_comb", "logic",
        "wire", "reg", "module", "endmodule", "input", "output", "and", "or",
        "not", "throughout", "within", "intersect", "first_match", "until",
        "until_with", "s_until", "s_eventually", "s_always", "nexttime",
        "strong", "weak", "1", "0", "1'b0", "1'b1",
        "rose", "fell", "stable", "past", "changed", "sampled",
        "onehot", "onehot0", "countones", "isunknown",
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


def _pec_check(args_tuple):
    idx, alt_sva, ref_sva, rtl = args_tuple
    if not alt_sva.strip():
        return idx, "EMPTY"
    from src.mock_verifier import syntax_check
    if not syntax_check(alt_sva)["ok"]:
        return idx, "PARSE_ERROR"
    try:
        from src.pec_yosys import prop_equivalence
        r = prop_equivalence(alt_sva, ref_sva, rtl, depth=10, timeout=15)
        return idx, r.verdict
    except Exception:
        return idx, "ERROR"


def stage_c_pec(args, with_alts):
    """PEC-verify each alt vs original ref. Keep alts where the alt is
    EQUIVALENT or IMPLIES_LM_TO_REF (alt-implies-ref → alt is at least as
    strong as ref; safe to add as additional reward target since rewarding
    a stronger property than the reference is monotonic in correctness)."""
    print(f"[stage C] verifying alts via PEC ({args.workers} workers)")

    tasks = []
    task_meta = []
    for ri, r in enumerate(with_alts):
        ref = r["reference_sva"]
        rtl = free_input_rtl(ref, r.get("rtl_context", ""))
        for ai, alt in enumerate(r.get("alt_svas", [])):
            tasks.append((len(tasks), alt, ref, rtl))
            task_meta.append((ri, ai))
    print(f"[stage C] {len(tasks)} alt × ref pairs to verify")

    verdicts = [None] * len(tasks)
    t0 = time.time()
    done = 0
    with mp.Pool(args.workers) as pool_p:
        for idx, verdict in pool_p.imap_unordered(_pec_check, tasks,
                                                   chunksize=4):
            verdicts[idx] = verdict
            done += 1
            if done % 500 == 0:
                eta = (time.time() - t0) / done * (len(tasks) - done)
                print(f"  pec {done}/{len(tasks)}  "
                      f"elapsed={time.time()-t0:.0f}s  eta={eta:.0f}s")
    print(f"[stage C] PEC done in {time.time()-t0:.0f}s")
    print(f"[stage C] verdict dist: {Counter(verdicts)}")

    # Aggregate equivalent alts per prompt
    equiv_per_prompt = defaultdict(list)
    implies_per_prompt = defaultdict(list)
    for ti, (ri, ai) in enumerate(task_meta):
        v = verdicts[ti]
        alt = with_alts[ri]["alt_svas"][ai]
        if v == "EQUIVALENT":
            equiv_per_prompt[ri].append(alt)
        elif v == "IMPLIES_LM_TO_REF":
            # alt-implies-ref: alt is *stronger* than ref. Acceptable target
            # because rewarding a stronger property still improves correctness.
            implies_per_prompt[ri].append(alt)

    # Build pool with multi-ref schema
    pool = []
    n_only_orig = 0
    n_with_extra = 0
    extra_counts = []
    for ri, r in enumerate(with_alts):
        ref_svas = [r["reference_sva"]]
        # Dedup by stripped string
        seen = {re.sub(r"\s+", " ", r["reference_sva"]).strip()}
        for alt in equiv_per_prompt.get(ri, []) + implies_per_prompt.get(ri, []):
            key = re.sub(r"\s+", " ", alt).strip()
            if key in seen:
                continue
            seen.add(key)
            ref_svas.append(alt)
        rec = {
            "id": r["id"],
            "src_id": r["src_id"],
            "nl": r["nl"],
            "is_original_nl": r["is_original_nl"],
            "reference_sva": r["reference_sva"],   # canonical ref
            "ref_svas": ref_svas,                  # multi-ref list (≥1)
            "n_extra_refs": len(ref_svas) - 1,
            "rtl_context": r["rtl_context"],
            "expected_tcl": r["expected_tcl"],
            "source": "phase2",
        }
        pool.append(rec)
        if len(ref_svas) == 1:
            n_only_orig += 1
        else:
            n_with_extra += 1
            extra_counts.append(len(ref_svas) - 1)

    avg_extra = sum(extra_counts) / max(1, len(extra_counts))
    print(f"[stage C] prompts only-orig: {n_only_orig}/{len(pool)}")
    print(f"[stage C] prompts with extra: {n_with_extra}/{len(pool)}")
    print(f"[stage C] avg extra refs (where >0): {avg_extra:.2f}")

    with open(F_POOL, "w") as f:
        for r in pool:
            f.write(json.dumps(r) + "\n")
    print(f"[stage C] wrote {F_POOL}")

    by_tcl = Counter(r["expected_tcl"] for r in pool)
    by_tcl_extra = Counter(r["expected_tcl"] for r in pool
                            if r["n_extra_refs"] > 0)
    manifest = {
        "ts": datetime.now().isoformat(),
        "n_source_records": 300,
        "k_nl": args.k_nl,
        "k_alt": args.k_alt,
        "n_pool": len(pool),
        "n_pool_with_extra_ref": n_with_extra,
        "verdict_dist": dict(Counter(verdicts)),
        "tcl_dist": dict(by_tcl),
        "tcl_dist_with_extra": dict(by_tcl_extra),
        "avg_extra_refs": round(avg_extra, 3),
    }
    with open(F_MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[stage C] wrote manifest: {F_MANIFEST}")
    return pool


# --- main --------------------------------------------------------------------

def load_vllm(args):
    print(f"[vllm] loading {DS_CODER_V2}")
    from vllm import LLM
    llm = LLM(
        model=DS_CODER_V2,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    tok = llm.get_tokenizer()
    return llm, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["A", "B", "C", "AB", "all"],
                    help="A=paraphrase, B=alt-gen, C=pec-verify")
    ap.add_argument("--k-nl", type=int, default=10,
                    help="paraphrases per source NL")
    ap.add_argument("--k-alt", type=int, default=4,
                    help="alt SVAs generated per (paraphrased NL) prompt")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-paraphrase-tokens", type=int, default=512)
    ap.add_argument("--max-alt-tokens", type=int, default=192)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    args = ap.parse_args()

    PHASE2_DIR.mkdir(parents=True, exist_ok=True)

    needs_vllm = args.stage in ("A", "B", "AB", "all")
    llm = tok = None
    if needs_vllm:
        llm, tok = load_vllm(args)

    if args.stage in ("A", "AB", "all"):
        paraphrased = stage_a_paraphrase(args, llm, tok)
    elif args.stage in ("B", "C"):
        if not F_PARAPHRASED.exists():
            print(f"[error] need {F_PARAPHRASED} for stage {args.stage}",
                  file=sys.stderr)
            sys.exit(2)
        paraphrased = [json.loads(l) for l in open(F_PARAPHRASED)]
        print(f"[stage] resumed paraphrased: {len(paraphrased)} records")

    if args.stage in ("B", "AB", "all"):
        with_alts = stage_b_altgen(args, llm, tok, paraphrased)
    elif args.stage == "C":
        if not F_ALTS.exists():
            print(f"[error] need {F_ALTS} for stage C", file=sys.stderr)
            sys.exit(2)
        with_alts = [json.loads(l) for l in open(F_ALTS)]
        print(f"[stage] resumed alts: {len(with_alts)} records")

    if args.stage in ("C", "all"):
        if needs_vllm and llm is not None:
            # free GPU before CPU-heavy PEC stage (PEC uses subprocesses)
            del llm
            import gc, torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        stage_c_pec(args, with_alts)

    print("\n[done] Phase 2 pool build complete")


if __name__ == "__main__":
    main()
