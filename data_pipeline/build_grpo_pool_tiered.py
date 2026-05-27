#!/usr/bin/env python3
"""
build_grpo_pool_tiered.py — build a tiered GRPO pool.

Tiers:
  Tier 1  (full_reward=True)  = yosys-slang parseable full-RTL samples.
                                Verifier can fire formal + vacuity reward
                                (the 0.60 part of the reward).
  Tier 2  (full_reward=False) = full-RTL samples that are NOT standalone
                                parseable. Verifier degrades gracefully:
                                the formal/vacuity rewards return 0 and
                                GRPO advantage comes from syntax+AST_sim
                                (the 0.40 part). Still useful RL signal.

Balancing: Tier 2 draws from the thin TCL levels (L2/L3/L5) first so the
GRPO pool isn't L1/L4-dominated.

Output:  data/train/grpo/grpo_pool_tiered.jsonl
         data/train/grpo/tiered_manifest.json

Also re-runs SFT/GRPO disjoint split — SFT excludes every body now in the
expanded GRPO pool.
"""
import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent
GRPO_DIR = EXPERIMENTS_DIR / "data" / "train" / "grpo"
SFT_DIR = EXPERIMENTS_DIR / "data" / "train" / "sft"
UNIFIED = EXPERIMENTS_DIR / "data" / "train" / "unified" / "train_unified.jsonl"
IN_EXPANDED = GRPO_DIR / "verifiable_expanded.jsonl"
OUT_POOL = GRPO_DIR / "grpo_pool_tiered.jsonl"
OUT_MANIFEST = GRPO_DIR / "tiered_manifest.json"

# Target Tier 2 quotas per TCL level (to re-balance)
TIER2_QUOTA = {
    1: 300,   # L1 already well-represented in Tier 1 — cap Tier 2
    2: 400,   # thin in Tier 1 (only 9) — grab many
    3: 200,   # thin in Tier 1 (only 3) — grab what's available
    4: 400,   # L4 is well-represented — cap
    5: 200,   # thin in Tier 1 (only 8) — grab many
}


def body_hash(sva: str) -> str:
    return hashlib.sha256(re.sub(r"\s+", " ", sva).strip().encode()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--require-golden-pass", action="store_true",
                    help="C1 mode: keep only Tier-1 samples whose golden SVA "
                         "verifies as PASS (signal-rich pool). Requires the "
                         "labeled file produced by "
                         "scripts/label_tier1_golden_status.py.")
    ap.add_argument("--labeled-tier1",
                    default=str(GRPO_DIR / "verifiable_parseable_labeled.jsonl"),
                    help="path to the labeled tier1 jsonl (used with "
                         "--require-golden-pass)")
    ap.add_argument("--out-pool", default=str(OUT_POOL))
    ap.add_argument("--out-manifest", default=str(OUT_MANIFEST))
    args = ap.parse_args()
    random.seed(args.seed)
    out_pool = Path(args.out_pool)
    out_manifest = Path(args.out_manifest)
    # When using C1 PASS-filter, default to a separate output so we don't
    # clobber the production pool unless caller passed an explicit --out-pool.
    if args.require_golden_pass and out_pool == OUT_POOL:
        out_pool = GRPO_DIR / "grpo_pool_tiered_passonly.jsonl"
        out_manifest = GRPO_DIR / "tiered_manifest_passonly.json"
        print(f"[C1] redirecting outputs to {out_pool.name} / "
              f"{out_manifest.name} (pass --out-pool to override)")

    # 0. Load test body hashes — GRPO MUST NOT contain any of these
    test_bodies = set()
    for f in [EXPERIMENTS_DIR / "data" / "test" / "nl2sva_human.jsonl",
              EXPERIMENTS_DIR / "data" / "test" / "assertionbench.jsonl"]:
        if not f.exists():
            continue
        for line in open(f):
            r = json.loads(line)
            test_bodies.add(body_hash(r["reference_sva"]))
    print(f"[guard] test body hashes loaded: {len(test_bodies)}")

    # 1. Load all expanded records, drop any that overlap with test
    all_recs = []
    dropped_test = 0
    for line in open(IN_EXPANDED):
        r = json.loads(line)
        if body_hash(r["sva"]) in test_bodies:
            dropped_test += 1
            continue
        # Extra paranoia: drop anything whose source says assertionbench
        if r.get("source") == "assertionbench":
            dropped_test += 1
            continue
        all_recs.append(r)
    if dropped_test:
        print(f"[guard] dropped {dropped_test} records that overlap with test "
              f"set (or source==assertionbench)")

    tier1 = [r for r in all_recs if r.get("parseable")]
    tier2_candidates = [r for r in all_recs if not r.get("parseable")]
    print(f"[load] expanded: {len(all_recs)}  tier1={len(tier1)}  "
          f"tier2-candidates={len(tier2_candidates)}")

    # C1: filter Tier 1 to golden-PASS samples for stronger RLVF signal
    if args.require_golden_pass:
        labeled_path = Path(args.labeled_tier1)
        if not labeled_path.exists():
            raise SystemExit(
                f"--require-golden-pass set but labeled file missing: "
                f"{labeled_path}\nRun: PYTHONPATH=. python3 "
                f"scripts/label_tier1_golden_status.py")
        # Build sva-body lookup from the labeled file
        labeled_status = {}
        for line in open(labeled_path):
            r = json.loads(line)
            labeled_status[body_hash(r["sva"])] = r.get("golden_status", "")
        before = len(tier1)
        kept = []
        unlabeled = 0
        status_breakdown = Counter()
        for r in tier1:
            st = labeled_status.get(body_hash(r["sva"]))
            if st is None:
                unlabeled += 1
                continue
            status_breakdown[st] += 1
            if st == "PASS":
                r["golden_status"] = "PASS"
                kept.append(r)
        print(f"[C1] tier1 PASS-filter: {before} → {len(kept)} "
              f"(status breakdown of labeled subset: {dict(status_breakdown)}"
              + (f", {unlabeled} unlabeled-skipped" if unlabeled else "") + ")")
        tier1 = kept

    # 2. Bucket Tier 2 candidates by TCL
    by_tcl = defaultdict(list)
    for r in tier2_candidates:
        by_tcl[int(r.get("expected_tcl", 0))].append(r)

    tier2_picked = []
    for lvl, quota in TIER2_QUOTA.items():
        cands = by_tcl.get(lvl, [])
        random.shuffle(cands)
        tier2_picked.extend(cands[:quota])
        print(f"  L{lvl} tier2: {min(len(cands), quota)} / "
              f"{len(cands)} (quota {quota})")

    # 3. Write pool with full_reward flag + tier
    kept_body = set()
    with open(out_pool, "w") as fo:
        for r in tier1:
            r["tier"] = 1
            r["full_reward"] = True
            fo.write(json.dumps(r) + "\n")
            kept_body.add(body_hash(r["sva"]))
        for r in tier2_picked:
            r["tier"] = 2
            r["full_reward"] = False
            fo.write(json.dumps(r) + "\n")
            kept_body.add(body_hash(r["sva"]))

    total = len(tier1) + len(tier2_picked)
    print(f"\n[grpo] pool size: {total}   tier1={len(tier1)}   "
          f"tier2={len(tier2_picked)}")

    # TCL breakdown
    per_tcl = Counter()
    per_tier_per_tcl = defaultdict(lambda: Counter())
    with open(out_pool) as f:
        for line in f:
            r = json.loads(line)
            lvl = int(r.get("expected_tcl", 0))
            per_tcl[lvl] += 1
            per_tier_per_tcl[r["tier"]][lvl] += 1
    print(f"\n[grpo] TCL distribution in pool:")
    for lvl in (1, 2, 3, 4, 5):
        t1 = per_tier_per_tcl[1].get(lvl, 0)
        t2 = per_tier_per_tcl[2].get(lvl, 0)
        print(f"  L{lvl}: {per_tcl[lvl]:>5}  (tier1={t1} + tier2={t2})")

    # 4. Re-split SFT — exclude every body now in GRPO pool.
    # Skipped under --require-golden-pass (C1) since we're producing a
    # variant pool for ablation, not replacing the production split.
    sft_kept = defaultdict(list)
    dropped = 0
    sft_total = 0
    if not args.require_golden_pass:
        with open(UNIFIED) as f:
            for line in f:
                r = json.loads(line)
                if body_hash(r["reference_sva"]) in kept_body:
                    dropped += 1
                    continue
                sft_kept[int(r.get("expected_tcl", 0))].append(r)
        sft_total = sum(len(v) for v in sft_kept.values())

        SFT_DIR.mkdir(parents=True, exist_ok=True)
        with open(SFT_DIR / "sft_train.jsonl", "w") as fo:
            for lvl in (1, 2, 3, 4, 5):
                for r in sft_kept.get(lvl, []):
                    fo.write(json.dumps(r) + "\n")
        for lvl in (1, 2, 3, 4, 5):
            with open(SFT_DIR / f"sft_train_L{lvl}.jsonl", "w") as f:
                for r in sft_kept.get(lvl, []):
                    f.write(json.dumps(r) + "\n")
        print(f"\n[sft] excluded {dropped} bodies now in GRPO pool")
        print(f"[sft] pool size: {sft_total}")
        print(f"[sft] per-TCL:")
        for lvl in (1, 2, 3, 4, 5):
            print(f"  L{lvl}: {len(sft_kept.get(lvl, [])):>5}")
    else:
        print(f"\n[sft] skipped re-split (--require-golden-pass mode produces "
              f"a variant pool for ablation only)")

    # 5. Manifest
    manifest = {
        "policy": "GRPO has a 2-tier pool. Tier 1 = verifier-parseable "
                  "(full formal+vacuity reward). Tier 2 = non-parseable "
                  "but has full-RTL module (degraded reward: syntax+AST_sim only).",
        "tier1_filter": ("require-golden-pass (C1)"
                         if args.require_golden_pass else "all parseable"),
        "grpo_pool_size": total,
        "grpo_tier1": len(tier1),
        "grpo_tier2": len(tier2_picked),
        "grpo_per_tcl": dict(per_tcl),
        "grpo_tier1_per_tcl": dict(per_tier_per_tcl[1]),
        "grpo_tier2_per_tcl": dict(per_tier_per_tcl[2]),
        "sft_pool_size": sft_total,
        "sft_per_tcl": {f"L{lvl}": len(sft_kept.get(lvl, []))
                        for lvl in range(1, 6)},
        "sft_excluded_for_disjointness": dropped,
        "intersection_check": "PASS (body hashes disjoint by construction)",
    }
    with open(out_manifest, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest: {out_manifest}")


if __name__ == "__main__":
    main()
