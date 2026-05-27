#!/usr/bin/env python3
"""
build_rtl_aware_multiref.py — alt-gen + PEC verify for the RTL-aware pool.

Input:  data/train/grpo/industrial_with_rtl_disable_iff.jsonl
Output: data/train/grpo/grpo_pool_pilot8_rtl.jsonl (multi-ref schema)

Reuses the Phase 2 pool builder's alt-gen + PEC verify stages. Unlike
previous pool builds, this one carries real (2KB+) rtl_context through to
the final training records.
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

F_IN = EXPERIMENTS_DIR / "data" / "train" / "grpo" / "industrial_with_rtl_disable_iff.jsonl"
F_ALTS = EXPERIMENTS_DIR / "data" / "train" / "grpo" / "pilot8_alts.jsonl"
F_POOL = EXPERIMENTS_DIR / "data" / "train" / "grpo" / "grpo_pool_pilot8_rtl.jsonl"
F_MANIFEST = EXPERIMENTS_DIR / "data" / "train" / "grpo" / "pilot8_manifest.json"

from scripts.build_phase2_pool import ALTGEN_SYSTEM, extract_sva, free_input_rtl, _pec_check


def stage_altgen(recs):
    from vllm import LLM, SamplingParams
    print(f"[vllm] loading {DS_CODER_V2}")
    llm = LLM(model=DS_CODER_V2, trust_remote_code=True, dtype="bfloat16",
              gpu_memory_utilization=0.5, max_model_len=4096)
    tok = llm.get_tokenizer()
    # Build prompts: NL only (same as Phase 2 flow — alt-gen doesn't need RTL)
    prompts = []
    for r in recs:
        msgs = [
            {"role": "system", "content": ALTGEN_SYSTEM},
            {"role": "user", "content": f"Generate an SVA assertion for:\n{r['nl']}"},
        ]
        prompts.append(tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True))
    rounds = [{"temp": 0.7, "k": 4, "max": 320}, {"temp": 0.95, "k": 4, "max": 320}]
    for cfg in rounds:
        sp = SamplingParams(temperature=cfg["temp"], top_p=0.95,
                            max_tokens=cfg["max"], n=cfg["k"])
        print(f"[round t={cfg['temp']} k={cfg['k']}] {len(prompts)} prompts")
        t0 = time.time()
        outs = llm.generate(prompts, sp)
        print(f"  inference {time.time()-t0:.0f}s")
        for r, out in zip(recs, outs):
            existing = set(re.sub(r"\s+", " ", s).strip() for s in r.get("alt_svas", []))
            r.setdefault("alt_svas", [])
            for s in out.outputs:
                sva = extract_sva(s.text)
                if not sva: continue
                k = re.sub(r"\s+", " ", sva).strip()
                if k in existing: continue
                existing.add(k)
                r["alt_svas"].append(sva)
    with open(F_ALTS, "w") as f:
        for r in recs: f.write(json.dumps(r) + "\n")
    print(f"[write] {F_ALTS}")
    del llm
    import gc, torch; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return recs


def stage_pec(recs, workers=32):
    tasks, meta = [], []
    for ri, r in enumerate(recs):
        ref = r["reference_sva"]
        rtl = free_input_rtl(ref)
        for ai, alt in enumerate(r.get("alt_svas", [])):
            tasks.append((len(tasks), alt, ref, rtl))
            meta.append((ri, ai))
    print(f"[pec] {len(tasks)} alts ({workers} workers)")
    verdicts = [None]*len(tasks)
    t0 = time.time(); done = 0
    with mp.Pool(workers) as pool:
        for idx, v in pool.imap_unordered(_pec_check, tasks, chunksize=4):
            verdicts[idx] = v; done += 1
            if done % 2000 == 0:
                eta = (time.time()-t0)/done * (len(tasks)-done)
                print(f"  pec {done}/{len(tasks)} elapsed={time.time()-t0:.0f}s eta={eta:.0f}s")
    print(f"[pec] done in {time.time()-t0:.0f}s")
    print(f"[pec] verdict dist: {Counter(verdicts)}")

    equiv = defaultdict(list); implies = defaultdict(list)
    for ti, (ri, ai) in enumerate(meta):
        alt = recs[ri]["alt_svas"][ai]
        v = verdicts[ti]
        if v == "EQUIVALENT": equiv[ri].append(alt)
        elif v == "IMPLIES_LM_TO_REF": implies[ri].append(alt)

    pool = []
    n_extra = 0; extras = []
    for ri, r in enumerate(recs):
        ref_svas = list(r["ref_svas"])
        seen = {re.sub(r"\s+", " ", s).strip() for s in ref_svas}
        for a in equiv.get(ri, []) + implies.get(ri, []):
            k = re.sub(r"\s+", " ", a).strip()
            if k in seen: continue
            seen.add(k); ref_svas.append(a)
        pool.append({
            "id": r["id"], "source": r["source"], "nl": r["nl"],
            "reference_sva": r["reference_sva"], "ref_svas": ref_svas,
            "n_extra_refs": len(ref_svas) - 1,
            "rtl_context": r["rtl_context"],
            "expected_tcl": r["expected_tcl"],
        })
        if len(ref_svas) > 1: n_extra += 1; extras.append(len(ref_svas)-1)
    avg = sum(extras)/max(1,len(extras))
    print(f"[pool] extra-ref prompts: {n_extra}/{len(pool)} ({100*n_extra/len(pool):.1f}%) avg={avg:.2f}")

    with open(F_POOL, "w") as f:
        for r in pool: f.write(json.dumps(r) + "\n")
    print(f"[write] {F_POOL}")
    manifest = {
        "ts": datetime.now().isoformat(),
        "n_pool": len(pool),
        "n_with_extra_ref": n_extra,
        "verdict_dist": dict(Counter(verdicts)),
        "tcl_dist": dict(Counter(r["expected_tcl"] for r in pool)),
        "avg_extra_refs": round(avg, 3),
    }
    with open(F_MANIFEST, "w") as f: json.dump(manifest, f, indent=2)
    print(f"[write] {F_MANIFEST}")


def main():
    recs = [json.loads(l) for l in open(F_IN)]
    print(f"[load] {len(recs)} from {F_IN.name}")
    recs = stage_altgen(recs)
    stage_pec(recs)
    print("[done] pilot 8 pool build complete")


if __name__ == "__main__":
    main()
