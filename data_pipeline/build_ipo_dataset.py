#!/usr/bin/env python3
"""
build_ipo_dataset.py — construct (prompt, chosen, rejected) triples for IPO
training from the multi-reference GRPO pool.

Pipeline:
  1. Load grpo_pool_phase2_v3.jsonl (disable-iff-wrapped refs).
  2. For each prompt, sample K=8 rollouts from the SFT-replay50 base via
     vLLM at T=1.0 with the FVEval prompt format.
  3. PEC each rollout against the prompt's ref_svas (multi-ref). Split into
     positives (PEC-EQUIVALENT to any ref) and negatives (NOT_EQUIVALENT,
     PARSE_ERROR, ERROR, UNSUPPORTED).
  4. Emit up to MAX_PAIRS_PER_PROMPT (chosen=positive, rejected=negative)
     pairs per prompt — skip prompts with no positives or no negatives.
  5. Write data/train/ipo/ipo_pairs.jsonl + manifest.

Why K=8: gives enough rollout diversity to find at least one EQUIV on
easy-to-medium prompts while keeping GPU cost bounded (3.3K × 8 = 26.4K
generations, ~3 min on one H200).

Why T=1.0 (not 0.7 or 1.1): we want a realistic sample of what the base
policy produces, neither overly diverse (many broken outputs) nor overly
peaked (no diversity). The SFT model's natural decoding temperature is
close to 1.0 on SVA generation.

Usage:
  source ${OSS_CAD_SUITE}/environment   # for PEC stage
  CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python3 \\
      scripts/build_ipo_dataset.py --stage all --k-rollouts 8 --workers 32
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

POOL_IN = EXPERIMENTS_DIR / "data" / "train" / "grpo" / "grpo_pool_phase2_v3.jsonl"
IPO_DIR = EXPERIMENTS_DIR / "data" / "train" / "ipo"
F_ROLLOUTS = IPO_DIR / "ipo_rollouts.jsonl"
F_PAIRS = IPO_DIR / "ipo_pairs.jsonl"
F_MANIFEST = IPO_DIR / "ipo_manifest.json"

# Reuse the FVEval prompt helpers and extract_sva from run_grpo_pilot
from scripts.run_grpo_pilot import (
    FVEVAL_SYSTEM_PROMPT,
    build_fveval_user_prompt,
    extract_sva,
    _free_input_rtl,
)

SFT_POLICY = EXPERIMENTS_DIR / "results" / "sft_qwen_coder_7b_replay50" / "checkpoint_20260420_210826"


# --- Stage R: rollout sampling ----------------------------------------------

def stage_rollouts(args):
    """Sample K rollouts per prompt from SFT-replay50 via vLLM."""
    from vllm import LLM, SamplingParams
    print(f"[vllm] loading {SFT_POLICY}")
    llm = LLM(
        model=str(SFT_POLICY),
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    tok = llm.get_tokenizer()

    pool = [json.loads(l) for l in open(POOL_IN)]
    print(f"[load] {len(pool)} prompts from {POOL_IN.name}")

    prompts = []
    for r in pool:
        msgs = [
            {"role": "system", "content": FVEVAL_SYSTEM_PROMPT},
            {"role": "user",
             "content": build_fveval_user_prompt(r["nl"], r["rtl_context"])},
        ]
        prompts.append(tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True))

    sp = SamplingParams(
        temperature=args.temperature, top_p=0.95,
        max_tokens=args.max_tokens, n=args.k_rollouts,
    )
    print(f"[generate] K={args.k_rollouts} rollouts × {len(prompts)} prompts "
          f"at T={args.temperature}")
    t0 = time.time()
    outputs = llm.generate(prompts, sp)
    print(f"[generate] done in {time.time()-t0:.0f}s")

    # Extract raw completions (keep full text, not just SVA — we need to
    # feed chat-format 'chosen'/'rejected' strings back to DPOTrainer)
    out_records = []
    for r, out in zip(pool, outputs):
        rollouts = []
        for s in out.outputs:
            raw = s.text  # the full assistant turn content
            sva = extract_sva(raw)
            rollouts.append({"raw": raw, "sva": sva})
        rec = {
            "id": r["id"],
            "src_id": r.get("src_id", r["id"]),
            "nl": r["nl"],
            "rtl_context": r["rtl_context"],
            "reference_sva": r["reference_sva"],
            "ref_svas": r["ref_svas"],
            "expected_tcl": r["expected_tcl"],
            "rollouts": rollouts,
        }
        out_records.append(rec)

    IPO_DIR.mkdir(parents=True, exist_ok=True)
    with open(F_ROLLOUTS, "w") as f:
        for r in out_records:
            f.write(json.dumps(r) + "\n")
    print(f"[write] {F_ROLLOUTS}")
    return out_records


# --- Stage P: PEC verification of rollouts ---------------------------------

def _pec_check(args_tuple):
    idx, sva, ref_list, rtl = args_tuple
    if not sva.strip():
        return idx, "EMPTY"
    from src.mock_verifier import syntax_check
    if not syntax_check(sva)["ok"]:
        return idx, "PARSE_ERROR"
    try:
        from src.pec_yosys import prop_equivalence
        best = None
        for ref in ref_list:
            try:
                r = prop_equivalence(sva, ref, rtl, depth=10, timeout=15)
                if r.verdict == "EQUIVALENT":
                    return idx, "EQUIVALENT"
                if best is None or _verdict_rank(r.verdict) > _verdict_rank(best):
                    best = r.verdict
            except Exception:
                continue
        return idx, best or "ERROR"
    except Exception:
        return idx, "ERROR"


def _verdict_rank(v):
    """Higher rank = more useful verdict for "does the model know this"."""
    return {"EQUIVALENT": 4, "IMPLIES_LM_TO_REF": 3, "IMPLIES_REF_TO_LM": 2,
            "NOT_EQUIVALENT": 1, "UNSUPPORTED": 0, "PARSE_ERROR": -1,
            "EMPTY": -2, "ERROR": -1}.get(v, 0)


def stage_pec(args, rollout_records):
    """PEC each rollout against the prompt's multi-ref list."""
    print(f"[pec] verifying rollouts with {args.workers} workers")

    tasks = []
    task_meta = []
    for ri, r in enumerate(rollout_records):
        rtl = _free_input_rtl(r["ref_svas"][0])
        for ki, roll in enumerate(r["rollouts"]):
            tasks.append((len(tasks), roll["sva"], r["ref_svas"], rtl))
            task_meta.append((ri, ki))
    print(f"[pec] {len(tasks)} rollouts × ref pairs")

    verdicts = [None] * len(tasks)
    t0 = time.time()
    done = 0
    with mp.Pool(args.workers) as pool_p:
        for idx, verdict in pool_p.imap_unordered(_pec_check, tasks,
                                                   chunksize=4):
            verdicts[idx] = verdict
            done += 1
            if done % 1000 == 0:
                eta = (time.time() - t0) / done * (len(tasks) - done)
                print(f"  pec {done}/{len(tasks)}  "
                      f"elapsed={time.time()-t0:.0f}s  eta={eta:.0f}s")
    print(f"[pec] done in {time.time()-t0:.0f}s")
    print(f"[pec] verdict dist: {Counter(verdicts)}")

    # Attach verdicts back to rollouts
    for ti, (ri, ki) in enumerate(task_meta):
        rollout_records[ri]["rollouts"][ki]["verdict"] = verdicts[ti]

    with open(F_ROLLOUTS, "w") as f:
        for r in rollout_records:
            f.write(json.dumps(r) + "\n")
    print(f"[write] {F_ROLLOUTS} (rollouts+verdicts)")
    return rollout_records


# --- Stage X: pair construction ---------------------------------------------

POS_SET = {"EQUIVALENT"}
NEG_SET = {"NOT_EQUIVALENT", "PARSE_ERROR", "ERROR", "EMPTY"}


def stage_pairs(args, rollout_records):
    pairs = []
    n_no_pos = 0
    n_no_neg = 0
    n_both = 0
    for r in rollout_records:
        rolls = r["rollouts"]
        pos = [ro for ro in rolls if ro.get("verdict") in POS_SET]
        neg = [ro for ro in rolls if ro.get("verdict") in NEG_SET]
        if not pos:
            n_no_pos += 1
            continue
        if not neg:
            n_no_neg += 1
            continue
        n_both += 1
        # Pair each pos with each neg (capped)
        for p in pos[:args.max_pos_per_prompt]:
            for n in neg[:args.max_neg_per_prompt]:
                # Build the chat-format prompt (string, with chat template
                # applied downstream by TRL if needed — but TRL DPOTrainer
                # accepts either chat-list or raw strings)
                user_content = build_fveval_user_prompt(r["nl"], r["rtl_context"])
                pairs.append({
                    "prompt": [
                        {"role": "system", "content": FVEVAL_SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    "chosen": [{"role": "assistant", "content": p["raw"]}],
                    "rejected": [{"role": "assistant", "content": n["raw"]}],
                    # Diagnostic fields (ignored by DPOTrainer but useful)
                    "prompt_id": r["id"],
                    "expected_tcl": r["expected_tcl"],
                    "chosen_verdict": p["verdict"],
                    "rejected_verdict": n["verdict"],
                })
                if len(pairs) >= args.max_pairs_total:
                    break
            if len(pairs) >= args.max_pairs_total:
                break
        if len(pairs) >= args.max_pairs_total:
            break

    print(f"\n[pairs] prompts with both pos+neg: {n_both}/"
          f"{len(rollout_records)}")
    print(f"[pairs] skipped no-positive: {n_no_pos}")
    print(f"[pairs] skipped no-negative: {n_no_neg}")
    print(f"[pairs] total pairs: {len(pairs)}")

    with open(F_PAIRS, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    print(f"[write] {F_PAIRS}")

    manifest = {
        "ts": datetime.now().isoformat(),
        "source_pool": str(POOL_IN.name),
        "k_rollouts": args.k_rollouts,
        "temperature": args.temperature,
        "max_pos_per_prompt": args.max_pos_per_prompt,
        "max_neg_per_prompt": args.max_neg_per_prompt,
        "n_prompts_with_both": n_both,
        "n_pairs_total": len(pairs),
        "tcl_dist": dict(Counter(p["expected_tcl"] for p in pairs)),
        "chosen_verdict_dist": dict(Counter(p["chosen_verdict"] for p in pairs)),
        "rejected_verdict_dist": dict(Counter(p["rejected_verdict"] for p in pairs)),
    }
    with open(F_MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[write] {F_MANIFEST}")


# --- main --------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["R", "P", "X", "RP", "all"])
    ap.add_argument("--k-rollouts", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--max-pos-per-prompt", type=int, default=2)
    ap.add_argument("--max-neg-per-prompt", type=int, default=2)
    ap.add_argument("--max-pairs-total", type=int, default=20000)
    args = ap.parse_args()

    IPO_DIR.mkdir(parents=True, exist_ok=True)

    if args.stage in ("R", "RP", "all"):
        records = stage_rollouts(args)
    elif args.stage in ("P", "X"):
        if not F_ROLLOUTS.exists():
            print(f"[error] need {F_ROLLOUTS}", file=sys.stderr)
            sys.exit(2)
        records = [json.loads(l) for l in open(F_ROLLOUTS)]
        print(f"[resume] {len(records)} records from {F_ROLLOUTS.name}")

    if args.stage in ("P", "RP", "all"):
        if args.stage in ("RP", "all"):
            # free GPU before CPU-heavy PEC stage
            import gc, torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        records = stage_pec(args, records)

    if args.stage in ("X", "all"):
        stage_pairs(args, records)

    print("\n[done] IPO dataset build complete")


if __name__ == "__main__":
    main()
