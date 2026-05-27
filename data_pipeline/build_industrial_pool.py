#!/usr/bin/env python3
"""
build_industrial_pool.py — experiment β pool builder.

Merge industrial SVA sources (OpenTitan macros + 3 GitHub scrape batches +
named-properties) into a single deduplicated 5K-record pool with real
industrial signal names. This is the input for the distribution-match
hypothesis test: can GRPO surpass the synthetic-pool 28.4% ceiling when
trained on SVAs that share the same signal-naming convention as the
NL2SVA-Human eval set?

Output schema matches grpo_pool_phase2.jsonl:
  {id, source, nl, reference_sva, rtl_context, expected_tcl, hash}

Filters:
  - syntactically valid SVA (basic regex match for assert property)
  - deduplicated by SVA-body hash (across all sources)
  - NL field retained as-is (downstream fill step will re-generate the
    NLs that are placeholder, empty, or look like code comments)
  - contamination check against NL2SVA-Human (79 tasks) — drops any SVA
    whose body hash matches a test sample

Usage:
  PYTHONPATH=. python3 scripts/build_industrial_pool.py \\
      --max-records 5000
"""
import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXPERIMENTS_DIR))

TRAIN_DIR = EXPERIMENTS_DIR / "data" / "train"
TEST_DIR = EXPERIMENTS_DIR / "data" / "test"
OUT_DIR = TRAIN_DIR / "grpo"
OUT_FILE = OUT_DIR / "industrial_raw.jsonl"

SOURCES = [
    TRAIN_DIR / "named_properties.jsonl",        # 950
    TRAIN_DIR / "snapshots_scraped.jsonl",       # 5982
    TRAIN_DIR / "github_search_scraped.jsonl",   # 4704
    TRAIN_DIR / "github_scraped.jsonl",          # 2752
    TRAIN_DIR / "opentitan_macros.jsonl",        # 1639
]


def body_hash(sva: str) -> str:
    """Normalize whitespace and lowercase for dedup-by-body."""
    s = re.sub(r"\s+", " ", sva or "").strip().lower()
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def is_valid_sva(sva: str) -> bool:
    s = (sva or "").strip().lower()
    if "assert" not in s or "property" not in s:
        return False
    if len(s) < 30 or len(s) > 2000:
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-records", type=int, default=5000)
    ap.add_argument("--out", default=str(OUT_FILE))
    args = ap.parse_args()

    # Contamination set: SVA body-hashes from the test set
    contam = set()
    test_file = TEST_DIR / "nl2sva_human.jsonl"
    if test_file.exists():
        for line in open(test_file):
            r = json.loads(line)
            contam.add(body_hash(r.get("reference_sva") or ""))
        print(f"[contam] {len(contam)} test-set body hashes to exclude")

    seen = set()
    kept = []
    source_counts = Counter()
    drop_reasons = Counter()

    for src in SOURCES:
        if not src.exists():
            print(f"[skip] missing {src}")
            continue
        for line in open(src):
            r = json.loads(line)
            sva = r.get("reference_sva") or r.get("sva") or ""
            if not is_valid_sva(sva):
                drop_reasons["invalid_sva"] += 1
                continue
            h = body_hash(sva)
            if h in seen:
                drop_reasons["duplicate"] += 1
                continue
            if h in contam:
                drop_reasons["test_contamination"] += 1
                continue
            seen.add(h)
            rec = {
                "id": r.get("id") or f"{src.stem}_{len(kept)}",
                "source": r.get("source") or src.stem,
                "nl": r.get("nl") or "",
                "reference_sva": sva,
                "rtl_context": r.get("rtl_context") or "",
                "expected_tcl": r.get("expected_tcl", 0),
                "hash": h,
            }
            kept.append(rec)
            source_counts[src.stem] += 1

    # Shuffle deterministically then subsample
    import random
    random.seed(0)
    random.shuffle(kept)
    if args.max_records and len(kept) > args.max_records:
        kept = kept[: args.max_records]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")

    # Tally TCL and source after subsampling
    tcl = Counter(r["expected_tcl"] for r in kept)
    src_after = Counter(r["source"] for r in kept)

    print(f"\n[write] {len(kept)} records → {args.out}")
    print(f"[drop]  {dict(drop_reasons)}")
    print(f"[source ingest]  {dict(source_counts)}")
    print(f"[source kept]    {dict(src_after)}")
    print(f"[tcl]            {dict(tcl)}")


if __name__ == "__main__":
    main()
