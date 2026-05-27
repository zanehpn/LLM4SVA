#!/usr/bin/env python3
"""
build_industrial_grpo_pool.py — experiment β step 3.

Takes the NL-refilled industrial pool (industrial_nl_filled.jsonl, 5K
records) and produces a GRPO-ready multi-reference pool with disable-iff-
wrapped canonical references matching the NL2SVA-Human eval convention.

Pipeline:
  1. Load industrial_nl_filled.jsonl.
  2. Drop records where NL refill didn't produce a real description.
  3. Add `disable iff (tb_reset)` to canonical refs that lack it (keep
     raw form as an additional multi-ref alternative).
  4. Generate K alt-SVA candidates per prompt at temp 0.7 + 0.95 via
     DS-Coder-V2-Lite (vLLM).
  5. PEC-verify each alt against the canonical ref; keep alts that are
     EQUIVALENT or IMPLIES_LM_TO_REF.
  6. Emit grpo_pool_industrial.jsonl with `ref_svas` list.

Reuses helpers from build_phase2_pool and extend_phase2_alts so the
schema and reward logic match Pilot 5's recipe.

Usage:
  source ${OSS_CAD_SUITE}/environment
  CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python3 \\
      scripts/build_industrial_grpo_pool.py --stage all --workers 32
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

IND_DIR = EXPERIMENTS_DIR / "data" / "train" / "grpo"
F_IN = IND_DIR / "industrial_nl_filled.jsonl"
F_ALTS = IND_DIR / "industrial_alts.jsonl"
F_POOL = IND_DIR / "grpo_pool_industrial.jsonl"
F_MANIFEST = IND_DIR / "industrial_manifest.json"

# Reuse extract_sva, free_input_rtl, _pec_check, ALTGEN_SYSTEM
from scripts.build_phase2_pool import (
    ALTGEN_SYSTEM, extract_sva, free_input_rtl, _pec_check,
)
from scripts.add_disable_iff_refs import add_disable_iff


PLACEHOLDER_RE = re.compile(r"^\s*\[(ASSERT|ASSUME|COVER)", re.I)


def nl_is_real(nl: str) -> bool:
    """NL passes if it's non-trivial, not a placeholder, and doesn't look
    like a C-style code comment."""
    nl = (nl or "").strip()
    if len(nl) < 15 or PLACEHOLDER_RE.match(nl):
        return False
    # Drop code-comment-looking NLs
    if nl.lower().startswith(("//", "/*", "--", "#", "todo", "fixme")):
        return False
    return True


def stage_prep(args):
    """Load industrial_nl_filled.jsonl, apply disable-iff wrap, filter NL."""
    records = [json.loads(l) for l in open(F_IN)]
    print(f"[load] {len(records)} records from {F_IN.name}")

    n_drop_nl = 0
    n_disable_added = 0
    n_disable_present = 0
    prepped = []
    for r in records:
        if not nl_is_real(r.get("nl", "")):
            n_drop_nl += 1
            continue
        raw_ref = r["reference_sva"]
        di_ref = add_disable_iff(raw_ref)
        if di_ref == raw_ref:
            if "disable iff" in raw_ref.lower():
                n_disable_present += 1
            # else the regex didn't match — keep raw anyway
        else:
            n_disable_added += 1
        ref_svas = [di_ref]
        if di_ref != raw_ref:
            ref_svas.append(raw_ref)
        rec = {
            "id": r["id"],
            "source": r["source"],
            "nl": r["nl"].strip(),
            "reference_sva": di_ref,
            "ref_svas": ref_svas,
            "rtl_context": r.get("rtl_context", ""),
            "expected_tcl": r["expected_tcl"],
            "hash": r["hash"],
        }
        prepped.append(rec)

    print(f"[filter] dropped for NL quality: {n_drop_nl}")
    print(f"[disable-iff] wrapped now: {n_disable_added}")
    print(f"[disable-iff] already had:  {n_disable_present}")
    print(f"[prepped] {len(prepped)} records ready for alt-gen")
    return prepped


def stage_altgen(args, prepped):
    """Generate K alts per prompt at two temps."""
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print(f"[vllm] loading {DS_CODER_V2}")
    llm = LLM(
        model=DS_CODER_V2,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    tok = llm.get_tokenizer()

    # Build prompts once
    prompts = []
    for r in prepped:
        msgs = [
            {"role": "system", "content": ALTGEN_SYSTEM},
            {"role": "user", "content": f"Generate an SVA assertion for:\n{r['nl']}"},
        ]
        prompts.append(tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True))

    for round_cfg in args.rounds:
        sp = SamplingParams(
            temperature=round_cfg["temp"], top_p=0.95,
            max_tokens=round_cfg["max"], n=round_cfg["k"],
        )
        print(f"[round t={round_cfg['temp']} k={round_cfg['k']} "
              f"max={round_cfg['max']}] {len(prompts)} prompts")
        t0 = time.time()
        outputs = llm.generate(prompts, sp)
        print(f"  inference {time.time()-t0:.0f}s")

        for r, out in zip(prepped, outputs):
            existing = set(re.sub(r"\s+", " ", s).strip()
                           for s in r.get("alt_svas", []))
            r.setdefault("alt_svas", [])
            new = 0
            for s in out.outputs:
                sva = extract_sva(s.text)
                if not sva:
                    continue
                key = re.sub(r"\s+", " ", sva).strip()
                if key in existing:
                    continue
                existing.add(key)
                r["alt_svas"].append(sva)
                new += 1
        post_counts = [len(r.get("alt_svas", [])) for r in prepped]
        print(f"  avg alts/prompt after round: "
              f"{sum(post_counts)/len(post_counts):.1f}")

    with open(F_ALTS, "w") as f:
        for r in prepped:
            f.write(json.dumps(r) + "\n")
    print(f"[write] {F_ALTS}")
    # free GPU
    del llm
    import gc, torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return prepped


def stage_pec(args, with_alts):
    """PEC-verify all alts; build final multi-ref pool."""
    tasks = []
    task_meta = []
    for ri, r in enumerate(with_alts):
        ref = r["reference_sva"]
        rtl = free_input_rtl(ref)
        for ai, alt in enumerate(r.get("alt_svas", [])):
            tasks.append((len(tasks), alt, ref, rtl))
            task_meta.append((ri, ai))
    print(f"[pec] {len(tasks)} alts to verify ({args.workers} workers)")

    verdicts = [None] * len(tasks)
    t0 = time.time()
    done = 0
    with mp.Pool(args.workers) as pool_p:
        for idx, verdict in pool_p.imap_unordered(_pec_check, tasks, chunksize=4):
            verdicts[idx] = verdict
            done += 1
            if done % 2000 == 0:
                eta = (time.time()-t0)/done * (len(tasks)-done)
                print(f"  pec {done}/{len(tasks)}  "
                      f"elapsed={time.time()-t0:.0f}s  eta={eta:.0f}s")
    print(f"[pec] done in {time.time()-t0:.0f}s")
    print(f"[pec] verdict dist: {Counter(verdicts)}")

    equiv = defaultdict(list)
    implies = defaultdict(list)
    for ti, (ri, ai) in enumerate(task_meta):
        alt = with_alts[ri]["alt_svas"][ai]
        v = verdicts[ti]
        if v == "EQUIVALENT":
            equiv[ri].append(alt)
        elif v == "IMPLIES_LM_TO_REF":
            implies[ri].append(alt)

    # Finalize pool
    pool = []
    n_with_extra = 0
    extra_counts = []
    for ri, r in enumerate(with_alts):
        ref_svas = list(r["ref_svas"])
        seen = {re.sub(r"\s+", " ", s).strip() for s in ref_svas}
        for alt in equiv.get(ri, []) + implies.get(ri, []):
            key = re.sub(r"\s+", " ", alt).strip()
            if key in seen:
                continue
            seen.add(key)
            ref_svas.append(alt)
        rec = {
            "id": r["id"],
            "source": r["source"],
            "nl": r["nl"],
            "reference_sva": r["reference_sva"],
            "ref_svas": ref_svas,
            "n_extra_refs": len(ref_svas) - 1,
            "rtl_context": r.get("rtl_context", ""),
            "expected_tcl": r["expected_tcl"],
        }
        pool.append(rec)
        if rec["n_extra_refs"] > 0:
            n_with_extra += 1
            extra_counts.append(rec["n_extra_refs"])

    avg_extra = sum(extra_counts) / max(1, len(extra_counts))
    print(f"[pool] prompts with extra ref: {n_with_extra}/{len(pool)} "
          f"({100*n_with_extra/len(pool):.1f}%)")
    print(f"[pool] avg extra refs (where >0): {avg_extra:.2f}")

    with open(F_POOL, "w") as f:
        for r in pool:
            f.write(json.dumps(r) + "\n")
    print(f"[write] {F_POOL}")

    manifest = {
        "ts": datetime.now().isoformat(),
        "input_records": str(F_IN.name),
        "n_pool": len(pool),
        "n_with_extra_ref": n_with_extra,
        "verdict_dist": dict(Counter(verdicts)),
        "tcl_dist": dict(Counter(r["expected_tcl"] for r in pool)),
        "tcl_dist_with_extra": dict(Counter(
            r["expected_tcl"] for r in pool if r["n_extra_refs"] > 0)),
        "source_dist": dict(Counter(r["source"] for r in pool)),
        "avg_extra_refs": round(avg_extra, 3),
    }
    with open(F_MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[write] {F_MANIFEST}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["prep", "altgen", "pec", "all"])
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    args = ap.parse_args()

    # Two alt-gen rounds: clean + diverse
    args.rounds = [
        {"temp": 0.7, "k": 4, "max": 320},
        {"temp": 0.95, "k": 4, "max": 320},
    ]

    if args.stage in ("prep", "all"):
        prepped = stage_prep(args)
    else:
        if not F_ALTS.exists():
            print(f"[error] need {F_ALTS} for stage", file=sys.stderr)
            sys.exit(2)
        prepped = [json.loads(l) for l in open(F_ALTS)]

    if args.stage in ("altgen", "all"):
        prepped = stage_altgen(args, prepped)

    if args.stage in ("pec", "all"):
        stage_pec(args, prepped)

    print("\n[done] industrial GRPO pool build complete")


if __name__ == "__main__":
    main()
