#!/usr/bin/env python3
"""
extend_phase2_alts.py — boost the multi-reference yield in grpo_pool_phase2.

Phase 2 v1 produced only 6.8% prompts with extra refs because the alt-gen
stage hit 78% PARSE_ERROR, mostly from truncation (max_tokens=192) and
high temp (0.95). This script:

  1. Loads the existing phase2_alts.jsonl.
  2. Runs N additional alt-gen rounds with caller-specified (temp, k,
     max_tokens) tuples — defaults to two rounds: a clean low-temp pass
     and a diverse high-temp pass with longer budget.
  3. Appends the new candidates to each prompt's alt_svas list.
  4. Re-runs PEC on the new candidates only (existing verdicts are
     preserved on disk via phase2_alts.jsonl).
  5. Merges all verified-equivalent alts into a bigger
     grpo_pool_phase2_v2.jsonl.

Usage:
  source ${OSS_CAD_SUITE}/environment   # for stage C
  CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. python3 scripts/extend_phase2_alts.py \
      --rounds "temp=0.7,k=4,max=320" "temp=0.95,k=4,max=320" \
      --workers 32
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

PHASE2_DIR = EXPERIMENTS_DIR / "data" / "train" / "grpo"
F_ALTS_IN = PHASE2_DIR / "phase2_alts.jsonl"
F_ALTS_OUT = PHASE2_DIR / "phase2_alts_v2.jsonl"
F_POOL_OUT = PHASE2_DIR / "grpo_pool_phase2_v2.jsonl"
F_MANIFEST = PHASE2_DIR / "phase2_v2_manifest.json"

# import shared helpers from the original builder
from scripts.build_phase2_pool import (
    ALTGEN_SYSTEM, extract_sva, free_input_rtl, _pec_check,
)


def parse_round(spec: str) -> dict:
    """Parse 'temp=0.7,k=4,max=320' into {'temp':0.7, 'k':4, 'max':320}."""
    out = {}
    for part in spec.split(","):
        k, v = part.split("=")
        out[k.strip()] = float(v) if "." in v else int(v)
    assert {"temp", "k", "max"} <= out.keys(), f"bad round spec: {spec}"
    return out


def run_altgen_round(llm, tok, with_alts, round_cfg):
    """Generate `k` additional alts per prompt at `temp`, append to alt_svas."""
    from vllm import SamplingParams
    sp = SamplingParams(
        temperature=round_cfg["temp"], top_p=0.95,
        max_tokens=round_cfg["max"], n=round_cfg["k"],
    )
    prompts = []
    for r in with_alts:
        msgs = [
            {"role": "system", "content": ALTGEN_SYSTEM},
            {"role": "user",
             "content": f"Generate an SVA assertion for:\n{r['nl']}"},
        ]
        prompts.append(tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True))
    print(f"[round t={round_cfg['temp']} k={round_cfg['k']} "
          f"max={round_cfg['max']}] {len(prompts)} prompts")

    t0 = time.time()
    outputs = llm.generate(prompts, sp)
    print(f"  inference {time.time()-t0:.0f}s "
          f"({(time.time()-t0)/len(prompts):.2f}s/prompt)")

    new_alts_total = 0
    for r, out in zip(with_alts, outputs):
        existing = set(re.sub(r"\s+", " ", s).strip()
                       for s in r.get("alt_svas", []))
        for s in out.outputs:
            sva = extract_sva(s.text)
            if not sva:
                continue
            key = re.sub(r"\s+", " ", sva).strip()
            if key in existing:
                continue
            existing.add(key)
            r["alt_svas"].append(sva)
            new_alts_total += 1
    print(f"  added {new_alts_total} new dedup alts "
          f"(avg {new_alts_total/len(with_alts):.2f}/prompt)")
    return with_alts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", nargs="+",
                    default=["temp=0.7,k=4,max=320",
                             "temp=0.95,k=4,max=320"],
                    help="alt-gen rounds to add — each 'temp=,k=,max='")
    ap.add_argument("--stage", default="all", choices=["B", "C", "all"])
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    args = ap.parse_args()

    rounds = [parse_round(s) for s in args.rounds]
    print(f"[extend] {len(rounds)} additional alt-gen rounds: {rounds}")

    # Load existing alts
    if not F_ALTS_IN.exists():
        print(f"[error] need {F_ALTS_IN}", file=sys.stderr)
        sys.exit(2)
    with_alts = [json.loads(l) for l in open(F_ALTS_IN)]
    print(f"[load] {len(with_alts)} prompts from {F_ALTS_IN.name}")
    pre_counts = [len(r.get("alt_svas", [])) for r in with_alts]
    print(f"[load] existing alts: total={sum(pre_counts)}  "
          f"avg={sum(pre_counts)/len(pre_counts):.1f}/prompt")

    if args.stage in ("B", "all"):
        from vllm import LLM
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
        for ri, cfg in enumerate(rounds):
            with_alts = run_altgen_round(llm, tok, with_alts, cfg)
            # Save after each round so a mid-run crash doesn't lose work.
            with open(F_ALTS_OUT, "w") as f:
                for r in with_alts:
                    f.write(json.dumps(r) + "\n")
            post_counts = [len(r.get("alt_svas", [])) for r in with_alts]
            print(f"[round {ri+1}/{len(rounds)} saved] total alts="
                  f"{sum(post_counts)}  avg={sum(post_counts)/len(post_counts):.1f}/prompt"
                  f" → {F_ALTS_OUT.name}")
        # release GPU before stage C subprocesses
        del llm
        import gc, torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    elif args.stage == "C":
        if not F_ALTS_OUT.exists():
            print(f"[error] need {F_ALTS_OUT} for stage C", file=sys.stderr)
            sys.exit(2)
        with_alts = [json.loads(l) for l in open(F_ALTS_OUT)]

    if args.stage in ("C", "all"):
        # PEC over all alts (re-verify everything for simplicity & correctness)
        tasks = []
        task_meta = []
        for ri, r in enumerate(with_alts):
            ref = r["reference_sva"]
            rtl = free_input_rtl(ref, r.get("rtl_context", ""))
            for ai, alt in enumerate(r.get("alt_svas", [])):
                tasks.append((len(tasks), alt, ref, rtl))
                task_meta.append((ri, ai))
        print(f"[pec] {len(tasks)} alt × ref pairs")
        verdicts = [None] * len(tasks)
        t0 = time.time()
        done = 0
        with mp.Pool(args.workers) as pool_p:
            for idx, verdict in pool_p.imap_unordered(_pec_check, tasks,
                                                       chunksize=4):
                verdicts[idx] = verdict
                done += 1
                if done % 1000 == 0:
                    eta = (time.time()-t0)/done * (len(tasks)-done)
                    print(f"  pec {done}/{len(tasks)}  "
                          f"elapsed={time.time()-t0:.0f}s  eta={eta:.0f}s")
        print(f"[pec] done in {time.time()-t0:.0f}s")
        print(f"[pec] verdict dist: {Counter(verdicts)}")

        equiv = defaultdict(list)
        implies = defaultdict(list)
        for ti, (ri, ai) in enumerate(task_meta):
            v = verdicts[ti]
            alt = with_alts[ri]["alt_svas"][ai]
            if v == "EQUIVALENT":
                equiv[ri].append(alt)
            elif v == "IMPLIES_LM_TO_REF":
                implies[ri].append(alt)

        pool = []
        n_only_orig = 0
        n_with_extra = 0
        extra_counts = []
        for ri, r in enumerate(with_alts):
            ref_svas = [r["reference_sva"]]
            seen = {re.sub(r"\s+", " ", r["reference_sva"]).strip()}
            for alt in equiv.get(ri, []) + implies.get(ri, []):
                key = re.sub(r"\s+", " ", alt).strip()
                if key in seen:
                    continue
                seen.add(key)
                ref_svas.append(alt)
            rec = {
                "id": r["id"],
                "src_id": r.get("src_id", r["id"]),
                "nl": r["nl"],
                "is_original_nl": r.get("is_original_nl", True),
                "reference_sva": r["reference_sva"],
                "ref_svas": ref_svas,
                "n_extra_refs": len(ref_svas) - 1,
                "rtl_context": r["rtl_context"],
                "expected_tcl": r["expected_tcl"],
                "source": "phase2_v2",
            }
            pool.append(rec)
            if len(ref_svas) == 1:
                n_only_orig += 1
            else:
                n_with_extra += 1
                extra_counts.append(len(ref_svas) - 1)
        avg_extra = sum(extra_counts) / max(1, len(extra_counts))
        print(f"[pool] only-orig: {n_only_orig}/{len(pool)}")
        print(f"[pool] with-extra: {n_with_extra}/{len(pool)} "
              f"({100*n_with_extra/len(pool):.1f}%)")
        print(f"[pool] avg extra refs (where >0): {avg_extra:.2f}")

        with open(F_POOL_OUT, "w") as f:
            for r in pool:
                f.write(json.dumps(r) + "\n")
        print(f"[write] {F_POOL_OUT}")

        by_tcl = Counter(r["expected_tcl"] for r in pool)
        by_tcl_extra = Counter(r["expected_tcl"] for r in pool
                                if r["n_extra_refs"] > 0)
        manifest = {
            "ts": datetime.now().isoformat(),
            "input_alts": str(F_ALTS_IN.name),
            "output_alts": str(F_ALTS_OUT.name),
            "rounds_added": rounds,
            "n_pool": len(pool),
            "n_pool_with_extra_ref": n_with_extra,
            "verdict_dist": dict(Counter(verdicts)),
            "tcl_dist": dict(by_tcl),
            "tcl_dist_with_extra": dict(by_tcl_extra),
            "avg_extra_refs": round(avg_extra, 3),
        }
        with open(F_MANIFEST, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"[write] manifest: {F_MANIFEST}")

    print("\n[done] phase2 v2 build complete")


if __name__ == "__main__":
    main()
