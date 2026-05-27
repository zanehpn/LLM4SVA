#!/usr/bin/env python3
"""
annotate_codev_unified_with_tc.py — add a `temporal_class` field to every
row of the CodeV-SVA SFT / GRPO unified jsonl files IN PLACE, using the
same 3-class collapse policy as data/test/manifest_tc_split.json:

    C1 = {L1}   (combinational)
    C2 = {L2, L3, L4}   (bounded temporal)
    C3 = {L5}   (liveness)

Rewrites each file atomically (write to `.tmp` then rename). Rows whose
`expected_tcl` is not in {1..5} are passed through unchanged without a
class tag (count reported).

Usage:
    python scripts/annotate_codev_unified_with_tc.py
    # or targeted:
    python scripts/annotate_codev_unified_with_tc.py --inputs <path> ...
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_INPUTS = [
    ROOT / "data" / "CodeV-SVA-datasets" / "grpo" / "codev_grpo_unified.jsonl",
    ROOT / "data" / "CodeV-SVA-datasets" / "sft" / "codev_sft_unified.jsonl",
]

L_TO_C = {1: "C1", 2: "C2", 3: "C2", 4: "C2", 5: "C3"}


def annotate_in_place(src: Path):
    tmp = src.with_suffix(src.suffix + ".tmp")
    per_class = Counter()
    per_tcl = Counter()
    n_rows = 0
    n_untagged = 0
    with open(src) as fin, open(tmp, "w") as fout:
        for line in fin:
            line = line.rstrip("\n")
            if not line.strip():
                fout.write(line + "\n")
                continue
            row = json.loads(line)
            n_rows += 1
            tcl = row.get("expected_tcl")
            per_tcl[tcl] += 1
            cls = L_TO_C.get(tcl)
            if cls is not None:
                row["temporal_class"] = cls
                per_class[cls] += 1
            else:
                n_untagged += 1
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, src)
    return {
        "n_rows": n_rows,
        "n_untagged": n_untagged,
        "per_tcl": dict(sorted(per_tcl.items(), key=lambda x: (x[0] is None, x[0]))),
        "per_class": {c: per_class[c] for c in ("C1", "C2", "C3")},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="*", default=[str(p) for p in DEFAULT_INPUTS])
    args = ap.parse_args()

    for path in args.inputs:
        src = Path(path)
        if not src.exists():
            print(f"[skip] missing: {src}")
            continue
        print(f"\n[annotate] {src.name}")
        stats = annotate_in_place(src)
        print(f"  rows:          {stats['n_rows']}")
        print(f"  untagged:      {stats['n_untagged']} "
              f"(expected_tcl not in 1..5)")
        print(f"  by expected_tcl: {stats['per_tcl']}")
        print(f"  by temporal_class: {stats['per_class']}")


if __name__ == "__main__":
    main()
