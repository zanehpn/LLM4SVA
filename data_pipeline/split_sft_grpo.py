#!/usr/bin/env python3
"""
split_sft_grpo.py — produce strictly disjoint SFT and GRPO pools.

Rationale: if a sample is used to SFT the model, then reused for GRPO, the
improvement attribution becomes ambiguous ("did GRPO help, or was the model
already memorizing this sample?"). The two stages must see disjoint data.

Split policy:
  GRPO pool = data/train/grpo/verifiable_parseable.jsonl  (909 samples)
    — these have yosys-slang-parseable full RTL modules, the only ones on
      which the GRPO formal-verifier reward can actually fire.

  SFT pool  = data/train/unified/train_unified.jsonl
              MINUS any body present in the GRPO pool.

Outputs under data/train/sft/:
  sft_train.jsonl             (all SFT samples, concatenated)
  sft_train_L{1..5}.jsonl     (per-TCL shards for curriculum SFT)
  manifest.json               (counts, contamination verification)

Also emits a contamination report: intersection of SFT and GRPO body
hashes must be 0.
"""
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent
UNIFIED = EXPERIMENTS_DIR / "data" / "train" / "unified" / "train_unified.jsonl"
GRPO_PARSEABLE = EXPERIMENTS_DIR / "data" / "train" / "grpo" / "verifiable_parseable.jsonl"
SFT_DIR = EXPERIMENTS_DIR / "data" / "train" / "sft"


def body_hash(sva: str) -> str:
    body = re.sub(r"\s+", " ", sva).strip()
    return hashlib.sha256(body.encode()).hexdigest()[:16]


def main():
    SFT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Collect GRPO body hashes
    grpo_hashes = set()
    grpo_per_tcl = Counter()
    with open(GRPO_PARSEABLE) as f:
        for line in f:
            r = json.loads(line)
            grpo_hashes.add(body_hash(r["sva"]))
            grpo_per_tcl[int(r.get("expected_tcl", 0))] += 1
    print(f"[split] GRPO pool: {len(grpo_hashes)} unique body hashes")

    # 2. Stream unified, drop anything in GRPO pool
    kept_per_tcl = defaultdict(list)
    dropped_contamination = 0
    with open(UNIFIED) as f:
        for line in f:
            r = json.loads(line)
            if body_hash(r["reference_sva"]) in grpo_hashes:
                dropped_contamination += 1
                continue
            kept_per_tcl[int(r.get("expected_tcl", 0))].append(r)
    total_sft = sum(len(v) for v in kept_per_tcl.values())
    print(f"[split] Unified raw: "
          f"{sum(1 for _ in open(UNIFIED))}")
    print(f"[split] Removed from SFT (overlap with GRPO): "
          f"{dropped_contamination}")
    print(f"[split] SFT pool: {total_sft}")

    # 3. Write SFT files
    all_out = SFT_DIR / "sft_train.jsonl"
    with open(all_out, "w") as f:
        for lvl in (1, 2, 3, 4, 5):
            for r in kept_per_tcl.get(lvl, []):
                f.write(json.dumps(r) + "\n")

    for lvl in (1, 2, 3, 4, 5):
        with open(SFT_DIR / f"sft_train_L{lvl}.jsonl", "w") as f:
            for r in kept_per_tcl.get(lvl, []):
                f.write(json.dumps(r) + "\n")

    # 4. Contamination verification
    sft_hashes = set()
    for lvl, recs in kept_per_tcl.items():
        for r in recs:
            sft_hashes.add(body_hash(r["reference_sva"]))
    overlap = sft_hashes & grpo_hashes
    assert len(overlap) == 0, (
        f"contamination: {len(overlap)} bodies appear in both SFT and GRPO")

    # 5. Manifest
    manifest = {
        "policy": "SFT and GRPO pools are strictly disjoint; intersection "
                  "on body hash enforced to be 0.",
        "grpo_pool_size": len(grpo_hashes),
        "grpo_source":    "data/train/grpo/verifiable_parseable.jsonl",
        "grpo_per_tcl":   {f"L{lvl}": grpo_per_tcl[lvl] for lvl in range(1, 6)},
        "sft_pool_size":  total_sft,
        "sft_source":     "data/train/unified/train_unified.jsonl minus GRPO",
        "sft_per_tcl":    {f"L{lvl}": len(kept_per_tcl.get(lvl, []))
                           for lvl in range(1, 6)},
        "removed_from_sft_due_to_grpo_overlap": dropped_contamination,
        "intersection_check": "PASS (0 bodies)",
    }
    with open(SFT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # 6. Console summary
    print()
    print("=" * 60)
    print("DISJOINT SFT / GRPO SPLIT")
    print("=" * 60)
    print(f"GRPO pool (parseable full-RTL):  {len(grpo_hashes):>6}")
    print(f"  per TCL: L1={grpo_per_tcl[1]} L2={grpo_per_tcl[2]} "
          f"L3={grpo_per_tcl[3]} L4={grpo_per_tcl[4]} L5={grpo_per_tcl[5]}")
    print()
    print(f"SFT pool  (unified minus GRPO):  {total_sft:>6}")
    for lvl in (1, 2, 3, 4, 5):
        n = len(kept_per_tcl.get(lvl, []))
        print(f"  L{lvl}: {n:>5}  → {SFT_DIR / f'sft_train_L{lvl}.jsonl'}")
    print()
    print(f"Intersection (SFT ∩ GRPO): 0 bodies  ✓")
    print(f"Dropped from SFT due to overlap:  {dropped_contamination}")
    print()
    print(f"Manifest: {SFT_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()
