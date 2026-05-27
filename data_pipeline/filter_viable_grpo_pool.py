#!/usr/bin/env python3
"""
filter_viable_grpo_pool.py — given a screening result from
run_funcatk_eval.py (8 samples per task with PEC verdicts), keep only
the tasks that fall in the GRPO "golden zone" — neither already-solved
by the policy nor totally hopeless.

A task is `viable` iff   1 ≤ EQUIVALENT_count ≤ N-1  (for N samples).
That guarantees, in expectation, a 4-rollout GRPO batch on this task
will see reward variance > 0 (mix of EQUIVALENT and non-EQUIVALENT
rollouts), so GRPO has gradient signal.

Three output buckets are written next to the source pool:
  * <stem>_viable.jsonl       (1 ≤ eq_count ≤ N-1)
  * <stem>_too_easy.jsonl     (eq_count == N)
  * <stem>_too_hard.jsonl     (eq_count == 0)

Plus a histogram of eq_count distribution.

Usage:
    python scripts/filter_viable_grpo_pool.py \
        --screen results/grpo_logs/screen_C2_sft132954.json \
        --pool   data/CodeV-SVA-datasets/grpo/codev_grpo_unified_C2.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", required=True, help="funcatk result JSON")
    ap.add_argument("--pool", required=True, help="source GRPO pool jsonl")
    ap.add_argument("--out-dir", default="",
                    help="output directory (default: alongside --pool)")
    args = ap.parse_args()

    screen_path = Path(args.screen)
    pool_path = Path(args.pool)
    out_dir = Path(args.out_dir) if args.out_dir else pool_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[filter] screen JSON: {screen_path}")
    print(f"[filter] source pool: {pool_path}")

    screen = json.load(open(screen_path))
    num_samples = screen["num_samples"]

    # Build (id -> eq_count) from per_sample
    eq_count_by_id: dict[str, int] = {}
    for row in screen["per_sample"]:
        if not row.get("evaluable", True):
            continue
        c = sum(1 for cand in row["candidates"]
                if cand.get("pec_verdict") == "EQUIVALENT")
        eq_count_by_id[row["id"]] = c

    # Histogram
    hist = Counter(eq_count_by_id.values())
    print(f"\n[filter] eq_count histogram (N={num_samples}):")
    for k in range(num_samples + 1):
        n = hist.get(k, 0)
        bar = "█" * (n * 60 // max(1, max(hist.values())))
        print(f"  c={k}  {n:5d}  {bar}")

    # Bucket and write
    n_viable = n_too_easy = n_too_hard = n_missing = 0
    out_paths = {
        "viable": out_dir / f"{pool_path.stem}_viable.jsonl",
        "too_easy": out_dir / f"{pool_path.stem}_too_easy.jsonl",
        "too_hard": out_dir / f"{pool_path.stem}_too_hard.jsonl",
    }
    handles = {k: open(p, "w") for k, p in out_paths.items()}
    n_total_pool = 0
    with open(pool_path) as f:
        for line in f:
            row = json.loads(line)
            n_total_pool += 1
            tid = row.get("id")
            c = eq_count_by_id.get(tid)
            if c is None:
                n_missing += 1
                continue
            row["screen_eq_count"] = c
            row["screen_n"] = num_samples
            line_out = json.dumps(row, ensure_ascii=False) + "\n"
            if c == 0:
                handles["too_hard"].write(line_out)
                n_too_hard += 1
            elif c == num_samples:
                handles["too_easy"].write(line_out)
                n_too_easy += 1
            else:
                handles["viable"].write(line_out)
                n_viable += 1
    for h in handles.values():
        h.close()

    print(f"\n[filter] total in pool:    {n_total_pool}")
    print(f"[filter] missing in screen: {n_missing}")
    print(f"[filter] too_hard (c=0):    {n_too_hard}")
    print(f"[filter] viable (1..{num_samples-1}): {n_viable}")
    print(f"[filter] too_easy (c={num_samples}): {n_too_easy}")
    print()
    for k, p in out_paths.items():
        print(f"  -> {p}")


if __name__ == "__main__":
    main()
