#!/usr/bin/env python3
"""
test_formal_oracle.py — measure how reliable our (SVA lowering + sby)
pipeline is on known-good samples.

Per proposal Appendix (reliability validation):
  - PASS rate on golden Tier-1 samples should be ≥ 60% for the reward to be
    a usable training signal.
  - Anything less = lowering or verifier is too noisy; training degrades.

We run the pipeline on N random Tier-1 samples and report per-pattern PASS /
FAIL / PARSE_ERROR / UNSUPPORTED rates.
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXPERIMENTS_DIR))

from src.formal_verify import formal_verify

TIER1_POOL = EXPERIMENTS_DIR / "data" / "train" / "grpo" / "verifiable_parseable.jsonl"


def _run_one(args):
    idx, sva, rtl = args
    r = formal_verify(sva, rtl, timeout=12, depth=8)
    return idx, {
        "status": r.status,
        "reward": r.reward,
        "pattern": r.pattern,
        "tcl": r.tcl,
        "wallclock_s": round(r.wallclock_s, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out",
                    default=str(EXPERIMENTS_DIR / "results" / "oracle_tier1.json"))
    args = ap.parse_args()

    import random
    random.seed(args.seed)

    # Load Tier-1 samples
    samples = []
    with open(TIER1_POOL) as f:
        for line in f:
            r = json.loads(line)
            samples.append({
                "id": r["id"],
                "sva": r["sva"],
                "rtl": r["rtl_module"],
                "tcl": r["expected_tcl"],
            })
    random.shuffle(samples)
    samples = samples[:args.n]
    print(f"[oracle] running on {len(samples)} Tier-1 samples, "
          f"workers={args.workers}")

    tasks = [(i, s["sva"], s["rtl"]) for i, s in enumerate(samples)]

    results = {}
    done = 0
    with mp.Pool(args.workers) as pool:
        for idx, r in pool.imap_unordered(_run_one, tasks, chunksize=1):
            results[idx] = r
            done += 1
            if done % 10 == 0:
                status_counts = Counter(v["status"] for v in results.values())
                print(f"  {done}/{len(tasks)}  {dict(status_counts)}")

    # Aggregate
    status_counts = Counter(v["status"] for v in results.values())
    by_pattern = defaultdict(lambda: Counter())
    by_tcl = defaultdict(lambda: Counter())
    for i, v in results.items():
        by_pattern[v.get("pattern")][v["status"]] += 1
        by_tcl[samples[i]["tcl"]][v["status"]] += 1

    pass_rate = status_counts["PASS"] / len(results) * 100
    print("\n" + "=" * 60)
    print(f"ORACLE RESULT: PASS rate = {pass_rate:.1f}%  (target ≥60%)")
    print(f"status counts: {dict(status_counts)}")
    print("\nBy pattern:")
    for p, c in by_pattern.items():
        total = sum(c.values())
        pp = 100 * c["PASS"] / max(total, 1)
        print(f"  {p!s:<25} n={total:>3}  PASS={c['PASS']:>3} ({pp:.0f}%)  "
              f"other={dict({k:v for k,v in c.items() if k!='PASS'})}")
    print("\nBy TCL:")
    for lv in sorted(by_tcl):
        c = by_tcl[lv]
        total = sum(c.values())
        pp = 100 * c["PASS"] / max(total, 1)
        print(f"  L{lv}: n={total:>3}  PASS={c['PASS']:>3} ({pp:.0f}%)  "
              f"other={dict({k:v for k,v in c.items() if k!='PASS'})}")

    # Save
    report = {
        "n": len(results),
        "pass_rate_pct": round(pass_rate, 2),
        "status_counts": dict(status_counts),
        "by_pattern": {str(k): dict(v) for k, v in by_pattern.items()},
        "by_tcl": {str(k): dict(v) for k, v in by_tcl.items()},
        "samples": [{**samples[i], **results[i]} for i in range(len(results))],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
