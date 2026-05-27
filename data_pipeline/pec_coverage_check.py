#!/usr/bin/env python3
"""
pec_coverage_check.py — sanity-check the PEC oracle on the NL2SVA-Human
test set by running reference-vs-reference equivalence on every task.

Ideal outcome: 79/79 EQUIVALENT (every reference is equivalent to itself).
Anything else exposes a coverage hole in our lowering / Yosys flow.

Usage:
  source ${OSS_CAD_SUITE}/environment
  PYTHONPATH=. python3 scripts/pec_coverage_check.py --workers 8
"""
import argparse
import json
import multiprocessing as mp
import sys
from collections import Counter
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXPERIMENTS_DIR))
from src.pec_yosys import prop_equivalence

TEST_PATH = EXPERIMENTS_DIR / "data" / "test" / "nl2sva_human.jsonl"
OUT_PATH = EXPERIMENTS_DIR / "results" / "pec_coverage_check.json"


def _work(args):
    idx, sva, rtl, depth, timeout = args
    r = prop_equivalence(sva, sva, rtl, depth=depth, timeout=timeout)
    return idx, r.verdict, r.fwd_status, r.bwd_status, round(r.wallclock_s, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--depth", type=int, default=15)
    ap.add_argument("--timeout", type=int, default=30)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    tasks = []
    with open(TEST_PATH) as f:
        for line in f:
            tasks.append(json.loads(line))
    if args.limit:
        tasks = tasks[:args.limit]
    print(f"[pec-coverage] {len(tasks)} tasks from NL2SVA-Human")

    work = [(i, t["reference_sva"], t.get("rtl_context", ""),
             args.depth, args.timeout) for i, t in enumerate(tasks)]

    by_tcl = {}
    by_verdict = Counter()
    failures = []
    rows = []
    with mp.Pool(args.workers) as pool:
        for done, (idx, verdict, fwd, bwd, dt) in enumerate(
                pool.imap_unordered(_work, work, chunksize=2), 1):
            t = tasks[idx]
            tcl = int(t.get("expected_tcl", 0))
            by_verdict[verdict] += 1
            by_tcl.setdefault(tcl, Counter())[verdict] += 1
            rows.append({
                "id": t.get("id"), "tcl": tcl,
                "verdict": verdict, "fwd": fwd, "bwd": bwd, "seconds": dt,
            })
            if verdict != "EQUIVALENT":
                failures.append({
                    "id": t.get("id"), "tcl": tcl, "verdict": verdict,
                    "fwd": fwd, "bwd": bwd,
                    "ref_sva": t["reference_sva"][:200],
                })
            if done % 10 == 0:
                print(f"  {done}/{len(tasks)}  verdict counts: {dict(by_verdict)}")

    print()
    print("=" * 60)
    print("REFERENCE-VS-REFERENCE COVERAGE")
    print("=" * 60)
    print(f"Total: {len(tasks)}")
    for v, n in sorted(by_verdict.items(), key=lambda x: -x[1]):
        print(f"  {v:<22s}  {n:>4d}  ({100*n/len(tasks):.1f}%)")
    print()
    print("Per-TCL breakdown:")
    print(f"  {'TCL':<5s} {'total':<7s}  EQUIVALENT  NOT_EQUIV  PARSE_ERR  UNSUPP  TIMEOUT  OTHER")
    for tcl in sorted(by_tcl):
        c = by_tcl[tcl]
        total = sum(c.values())
        eq = c.get("EQUIVALENT", 0)
        ne = c.get("NOT_EQUIVALENT", 0)
        pe = c.get("PARSE_ERROR", 0)
        un = c.get("UNSUPPORTED", 0)
        to = c.get("TIMEOUT", 0)
        other = total - eq - ne - pe - un - to
        print(f"  L{tcl:<4d} {total:<7d}  {eq:>10d}  {ne:>9d}  {pe:>9d}  {un:>6d}  {to:>7d}  {other:>5d}")

    print()
    if failures:
        print(f"Non-EQUIVALENT samples ({len(failures)}):")
        for f in failures[:15]:
            print(f"  L{f['tcl']}  {f['verdict']:<18s}  ({f['fwd']}/{f['bwd']})  "
                  f"{f['id']}")
            print(f"     {f['ref_sva']}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump({
            "n_tasks": len(tasks),
            "verdict_counts": dict(by_verdict),
            "per_tcl": {str(k): dict(v) for k, v in by_tcl.items()},
            "failures": failures,
            "rows": rows,
        }, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
