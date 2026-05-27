#!/usr/bin/env python3
"""
label_tier1_golden_status.py — run the formal verifier on every Tier-1
GRPO sample (verifier-parseable full-RTL) and label each record with
`golden_status` / `golden_reward`.

Output:
  data/train/grpo/verifiable_parseable_labeled.jsonl   (input + 2 new fields)

This is the one-time pre-pass needed by `build_grpo_pool_tiered.py
--require-golden-pass`. Re-run when the lowering or verifier changes.

Usage:
  source ${OSS_CAD_SUITE}/environment
  PYTHONPATH=. python3 scripts/label_tier1_golden_status.py \\
      --workers 8 --timeout 12 --depth 8
"""
import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXPERIMENTS_DIR))
from src.formal_verify import formal_verify

IN_PATH = EXPERIMENTS_DIR / "data" / "train" / "grpo" / "verifiable_parseable.jsonl"
OUT_PATH = EXPERIMENTS_DIR / "data" / "train" / "grpo" / "verifiable_parseable_labeled.jsonl"


def _work(args):
    idx, sva, rtl, timeout, depth = args
    t0 = time.time()
    r = formal_verify(sva, rtl, timeout=timeout, depth=depth)
    return idx, r.status, r.reward, round(time.time() - t0, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=12)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0,
                    help="debug: label only first N records")
    args = ap.parse_args()

    records = []
    with open(IN_PATH) as f:
        for line in f:
            records.append(json.loads(line))
    if args.limit:
        records = records[:args.limit]
    print(f"[label] loading {len(records)} Tier-1 records from {IN_PATH.name}")

    tasks = [
        (i, r["sva"], r["rtl_module"], args.timeout, args.depth)
        for i, r in enumerate(records)
    ]

    t0 = time.time()
    done = 0
    status_counts = {}
    with mp.Pool(args.workers) as pool:
        for idx, status, reward, dt in pool.imap_unordered(_work, tasks, chunksize=2):
            records[idx]["golden_status"] = status
            records[idx]["golden_reward"] = reward
            records[idx]["golden_verify_seconds"] = dt
            status_counts[status] = status_counts.get(status, 0) + 1
            done += 1
            if done % 25 == 0 or done == len(records):
                elapsed = time.time() - t0
                eta = elapsed / done * (len(records) - done)
                print(f"  {done}/{len(records)}  "
                      f"elapsed={elapsed:.0f}s  eta={eta:.0f}s  "
                      f"counts={status_counts}")

    with open(OUT_PATH, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"\n[label] wrote {len(records)} labeled records → {OUT_PATH}")
    print("[label] golden status distribution:")
    for s, n in sorted(status_counts.items(), key=lambda x: -x[1]):
        print(f"  {s:>14s}  {n:>4d}  ({100*n/len(records):.1f}%)")


if __name__ == "__main__":
    main()
