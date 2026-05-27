#!/usr/bin/env python3
"""
build_rtl_aware_pool.py — build Pilot 8 training pool from RTL-aware scrape.

Input:  data/train/grpo/industrial_with_rtl.jsonl  (4091 raw records)
Output: data/train/grpo/industrial_with_rtl_filtered.jsonl  (~3K filtered records
        with meaningful rtl_context — for subsequent NL fill + disable-iff +
        multi-ref pipeline)

Filters:
  - drop records with rtl_context < 500 chars (non-informative context)
  - drop records where SVA body < 30 or > 2000 chars
  - drop duplicate SVA bodies
  - contamination-filter against NL2SVA-Human test set
  - classify expected_tcl (copied from src/tcl.py logic)
  - assign stable id

TCL classification uses the same rules as src/tcl_classifier.py:
  L5: s_eventually / s_until / s_always / nexttime
  L4: |-> or |=>
  L3: ##[ (range delay)
  L2: ##\\d (fixed delay)
  L1: otherwise
"""
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent
F_IN = EXPERIMENTS_DIR / "data" / "train" / "grpo" / "industrial_with_rtl.jsonl"
F_OUT = EXPERIMENTS_DIR / "data" / "train" / "grpo" / "industrial_with_rtl_filtered.jsonl"
F_TEST = EXPERIMENTS_DIR / "data" / "test" / "nl2sva_human.jsonl"


def body_hash(sva: str) -> str:
    return hashlib.sha256(re.sub(r"\s+", " ", sva).strip().encode()).hexdigest()[:16]


def classify_tcl(sva: str) -> int:
    s = sva.lower()
    if any(k in s for k in ("s_eventually", "s_until", "s_always", "nexttime")):
        return 5
    if "|->" in sva or "|=>" in sva:
        return 4
    if "##[" in sva:
        return 3
    if re.search(r"##\s*\d+", sva):
        return 2
    return 1


def main():
    # Contamination: body hashes from test set
    contam = set()
    for line in open(F_TEST):
        r = json.loads(line)
        contam.add(body_hash(r["reference_sva"]))
    print(f"[contam] {len(contam)} test-set body hashes to exclude")

    recs = [json.loads(l) for l in open(F_IN)]
    print(f"[load] {len(recs)} records from {F_IN.name}")

    seen = set()
    kept = []
    drop = Counter()
    for r in recs:
        sva = r.get("sva", "")
        rtl = r.get("rtl_context", "")
        if len(sva) < 30 or len(sva) > 2000:
            drop["bad_sva_len"] += 1
            continue
        if len(rtl) < 500:
            drop["short_rtl"] += 1
            continue
        h = body_hash(sva)
        if h in seen:
            drop["dup"] += 1
            continue
        if h in contam:
            drop["contam"] += 1
            continue
        seen.add(h)
        kept.append({
            "id": f"rtl_aware_{len(kept)}",
            "source": "rtl_aware_snapshot",
            "source_repo": r.get("source_repo", ""),
            "nl": "",                          # to be filled by vLLM
            "reference_sva": sva,
            "rtl_context": rtl,
            "expected_tcl": classify_tcl(sva),
            "hash": h,
            "file": r.get("file", ""),
            "line": r.get("line", 0),
        })

    with open(F_OUT, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")

    tcl = Counter(r["expected_tcl"] for r in kept)
    src = Counter(r["source_repo"] for r in kept)
    print(f"\n[drop]     {dict(drop)}")
    print(f"[write]    {len(kept)} records → {F_OUT}")
    print(f"[tcl]      {dict(tcl)}")
    print(f"[top 10 repos]")
    for name, n in src.most_common(10):
        print(f"  {n:4d}  {name}")


if __name__ == "__main__":
    main()
