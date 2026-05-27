#!/usr/bin/env python3
"""
remove_codev83k_from_expand_tcl.py — strip records that originated from
CodeV-SVA-dataset-83K out of expand_tcl/all_methods_merged.jsonl
(in place, atomic rewrite via .tmp).

A row is considered codev-83K-origin iff:
  - origin_dataset == "CodeV-SVA-dataset-83K"
  - id contains "codev" or "83k" (case-insensitive)
  - parent_id contains "codev" or "83k" (for downstream method3 rows)

Idempotent: rerunning on a cleaned file is a no-op.

Usage:
    python scripts/remove_codev83k_from_expand_tcl.py
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_INPUTS = [
    ROOT / "data" / "expand_tcl" / "all_methods_merged.jsonl",
]


def is_codev_origin(row: dict) -> bool:
    if row.get("origin_dataset") == "CodeV-SVA-dataset-83K":
        return True
    for k in ("id", "parent_id", "source_path", "origin_path"):
        v = row.get(k)
        if isinstance(v, str):
            low = v.lower()
            if "codev" in low or "83k" in low:
                return True
    return False


def clean_in_place(src: Path):
    tmp = src.with_suffix(src.suffix + ".tmp")
    n_in = n_out = n_dropped = 0
    with open(src) as fin, open(tmp, "w") as fout:
        for line in fin:
            line_stripped = line.strip()
            if not line_stripped:
                fout.write(line)
                continue
            row = json.loads(line_stripped)
            n_in += 1
            if is_codev_origin(row):
                n_dropped += 1
                continue
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_out += 1
    os.replace(tmp, src)
    return {"n_in": n_in, "n_dropped": n_dropped, "n_out": n_out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="*", default=[str(p) for p in DEFAULT_INPUTS])
    args = ap.parse_args()
    for path in args.inputs:
        src = Path(path)
        if not src.exists():
            print(f"[skip] missing: {src}")
            continue
        print(f"\n[clean] {src.name}")
        stats = clean_in_place(src)
        print(f"  rows in:      {stats['n_in']}")
        print(f"  dropped:      {stats['n_dropped']} (CodeV-83K origin)")
        print(f"  rows out:     {stats['n_out']}")


if __name__ == "__main__":
    main()
