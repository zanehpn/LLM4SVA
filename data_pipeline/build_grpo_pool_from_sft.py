#!/usr/bin/env python3
"""
build_grpo_pool_from_sft.py — re-purpose the NL-filled SFT pool as a GRPO
training pool, using **PEC equivalence to the reference SVA** as the reward
signal (instead of formal-verify on RTL).

Why this is better than the original tiered pool:
  - Pool size jumps from 1454 → ~10K (any sample with usable NL works)
  - Reward signal Δ jumps from 0.06 → ~0.7 (PEC EQUIV vs NOT_EQUIV is binary)
  - Tier 2 syntax-only fallback no longer needed
  - Zero dependency on RTL-parseable modules (PEC works on free-input SVA)

Output:
  data/train/grpo/grpo_pool_from_sft.jsonl
  data/train/grpo/from_sft_manifest.json

Schema per record:
  { "id": <orig id>, "nl": <usable NL>, "reference_sva": <ref>,
    "expected_tcl": int, "rtl_context": <optional, may be empty>,
    "source": <orig source>, "hash": <body hash>,
    "lowering_ok": bool, "pec_supported": bool  ← optional precomputed flags }

Optional pre-flight: --check-lowering runs lower_sva on each ref to filter
out samples our PEC oracle can't even encode (saves wasted GRPO compute).
"""
import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXPERIMENTS_DIR))

SFT_PATH = EXPERIMENTS_DIR / "data" / "train" / "sft" / "sft_train_nl_filled.jsonl"
TEST_DIR = EXPERIMENTS_DIR / "data" / "test"
OUT_DIR = EXPERIMENTS_DIR / "data" / "train" / "grpo"
OUT_POOL = OUT_DIR / "grpo_pool_from_sft.jsonl"
OUT_MANIFEST = OUT_DIR / "from_sft_manifest.json"

PLACEHOLDER_RE = re.compile(r"^\s*\[(?:ASSERT|ASSUME|COVER)[^\]]*\]\s*$",
                            re.IGNORECASE)


def is_real_nl(nl: str) -> bool:
    nl = (nl or "").strip()
    if len(nl) < 8:
        return False
    if PLACEHOLDER_RE.match(nl):
        return False
    return True


def body_hash(sva: str) -> str:
    return hashlib.sha256(re.sub(r"\s+", " ", sva).strip().encode()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-path", default=str(SFT_PATH))
    ap.add_argument("--out-pool", default=str(OUT_POOL))
    ap.add_argument("--out-manifest", default=str(OUT_MANIFEST))
    ap.add_argument("--check-lowering", action="store_true",
                    help="filter out SVAs whose ref can't be lowered (= PEC "
                         "can't evaluate them anyway). Adds ~2-5 min on full pool.")
    ap.add_argument("--max-len-nl", type=int, default=400,
                    help="drop NL longer than this (avoid truncated dumps)")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    if not in_path.exists():
        raise SystemExit(
            f"missing {in_path}. Run scripts/fill_nl_with_llm.py first.")

    # Load test body hashes — STRICTLY enforce no overlap
    test_bodies = set()
    for f in [TEST_DIR / "nl2sva_human.jsonl",
              TEST_DIR / "assertionbench.jsonl"]:
        if not f.exists(): continue
        for line in open(f):
            r = json.loads(line)
            test_bodies.add(body_hash(r["reference_sva"]))
    print(f"[guard] test body hashes loaded: {len(test_bodies)}")

    records = []
    with open(in_path) as f:
        for line in f:
            records.append(json.loads(line))
    n_total = len(records)
    print(f"[load] {n_total} samples from {in_path.name}")

    # Filter
    kept = []
    drop_no_nl = drop_test = drop_no_sva = drop_long = drop_lowering = 0
    seen_bodies = set()
    for r in records:
        ref = (r.get("reference_sva") or "").strip()
        nl = (r.get("nl") or "").strip()
        if not ref:
            drop_no_sva += 1; continue
        if not is_real_nl(nl):
            drop_no_nl += 1; continue
        if len(nl) > args.max_len_nl:
            drop_long += 1; continue
        bh = body_hash(ref)
        if bh in test_bodies:
            drop_test += 1; continue
        if bh in seen_bodies:
            continue   # de-dup
        seen_bodies.add(bh)
        kept.append({
            "id": r.get("id", ""),
            "nl": nl,
            "reference_sva": ref,
            "expected_tcl": int(r.get("expected_tcl", 0)),
            "rtl_context": r.get("rtl_context", ""),
            "source": r.get("source", ""),
            "hash": bh,
            "nl_filled_by": r.get("nl_filled_by", ""),
        })

    print(f"[filter] dropped: no_NL={drop_no_nl}  no_SVA={drop_no_sva}  "
          f"long_NL={drop_long}  test_overlap={drop_test}")
    print(f"[filter] kept (post-dedup): {len(kept)}")

    # Optional: pre-flight lowering check
    if args.check_lowering:
        from src.sva_lowering import lower_sva
        before = len(kept)
        kept2 = []
        for r in kept:
            lo = lower_sva(r["reference_sva"])
            r["lowering_ok"] = bool(lo["ok"])
            r["lowering_pattern"] = lo.get("pattern")
            if lo["ok"]:
                kept2.append(r)
            else:
                drop_lowering += 1
        kept = kept2
        print(f"[lowering] dropped {drop_lowering} unlowerable SVAs "
              f"({before} → {len(kept)})")

    # Per-TCL breakdown
    per_tcl = Counter(r["expected_tcl"] for r in kept)
    per_src = Counter(r["source"] for r in kept)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(args.out_pool, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    print(f"\n[grpo-from-sft] wrote {len(kept)} records → {args.out_pool}")

    print(f"\nTCL distribution:")
    for lvl in (1, 2, 3, 4, 5):
        n = per_tcl.get(lvl, 0)
        print(f"  L{lvl}: {n:>5d}")
    print(f"\nSource distribution:")
    for s, n in sorted(per_src.items(), key=lambda x: -x[1]):
        print(f"  {s:30s} {n:>5d}")

    manifest = {
        "policy": "PEC-equivalence-reward GRPO pool. Reward = "
                  "1.0 if PEC(gen, ref) == EQUIVALENT, "
                  "0.5 if IMPLIES_*, 0.15 if syntax_ok else 0. "
                  "Pool drawn from NL-filled SFT corpus.",
        "input": str(in_path),
        "n_records": len(kept),
        "per_tcl": dict(per_tcl),
        "per_source": dict(per_src),
        "drops": {
            "no_NL": drop_no_nl, "no_SVA": drop_no_sva,
            "long_NL": drop_long, "test_overlap": drop_test,
            "unlowerable": drop_lowering,
        },
        "lowering_check_run": args.check_lowering,
    }
    with open(args.out_manifest, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest: {args.out_manifest}")


if __name__ == "__main__":
    main()
