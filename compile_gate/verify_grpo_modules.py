#!/usr/bin/env python3
"""
verify_grpo_modules.py — run yosys-slang --ignore-assertions on every record
in grpo/verifiable.jsonl in parallel. Add `parseable` boolean field and emit:

  data/train/grpo/verifiable_parseable.jsonl  (only parseable=True rows)
  data/train/grpo/verifiable_with_flag.jsonl  (all rows, parseable field added)
"""
import argparse
import json
import multiprocessing as mp
import subprocess
import tempfile
from pathlib import Path

IN = Path("data/train/grpo/verifiable.jsonl")
OUT_ALL = Path("data/train/grpo/verifiable_with_flag.jsonl")
OUT_OK  = Path("data/train/grpo/verifiable_parseable.jsonl")
TIMEOUT = 12


def check_one(task):
    idx, rtl = task
    try:
        with tempfile.TemporaryDirectory(prefix=f"v_{idx}_") as d:
            p = Path(d) / "m.sv"
            p.write_text(rtl)
            rc = subprocess.call(
                ["yosys", "-q", "-m", "slang", "-p",
                 f"read_slang --ignore-assertions {p}; hierarchy"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=TIMEOUT,
            )
            return idx, rc == 0
    except Exception:
        return idx, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=mp.cpu_count() // 2)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    recs = []
    with open(IN) as f:
        for i, line in enumerate(f):
            recs.append(json.loads(line))
            if args.limit and i + 1 >= args.limit:
                break
    print(f"[verify] {len(recs)} records, workers={args.workers}")
    tasks = [(i, r["rtl_module"]) for i, r in enumerate(recs)]

    results = {}
    with mp.Pool(args.workers) as pool:
        for k, (idx, ok) in enumerate(pool.imap_unordered(check_one, tasks, chunksize=4), 1):
            results[idx] = ok
            if k % 200 == 0:
                passed = sum(1 for v in results.values() if v)
                print(f"  progress {k}/{len(tasks)}  passed={passed} "
                      f"({100*passed/k:.1f}%)")

    n_ok = 0
    per_src = {}
    with open(OUT_ALL, "w") as fa, open(OUT_OK, "w") as fo:
        for i, r in enumerate(recs):
            r["parseable"] = results.get(i, False)
            fa.write(json.dumps(r) + "\n")
            src = r["source"]
            per_src.setdefault(src, [0, 0])
            per_src[src][1] += 1
            if r["parseable"]:
                fo.write(json.dumps(r) + "\n")
                n_ok += 1
                per_src[src][0] += 1

    print(f"\n[verify] parseable: {n_ok} / {len(recs)} "
          f"({100*n_ok/len(recs):.1f}%)")
    print(f"[verify] per-source (parseable / total):")
    for s, (ok, total) in sorted(per_src.items(), key=lambda x: -x[1][0]):
        print(f"  {s:<22} {ok:>5} / {total:<5}  ({100*ok/max(total,1):.1f}%)")
    print(f"\nOutputs:")
    print(f"  {OUT_OK}")
    print(f"  {OUT_ALL}")


if __name__ == "__main__":
    main()
