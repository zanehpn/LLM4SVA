#!/usr/bin/env python3
"""
split_codev_unified_by_tc.py — shard the CodeV-SVA SFT / GRPO unified
jsonl files into C1/C2/C3 partitions using the same 3-class collapse
policy as data/test/manifest_tc_split.json:

    C1 = {L1}   (combinational)
    C2 = {L2, L3, L4}   (bounded temporal)
    C3 = {L5}   (liveness)

Adds a `temporal_class` field to each output row, matching test_C1/C2/C3's
schema. Writes siblings of the source file:

    <dir>/<stem>_C1.jsonl
    <dir>/<stem>_C2.jsonl
    <dir>/<stem>_C3.jsonl
    <dir>/<stem>_tc_split_manifest.json

Usage:
    python scripts/split_codev_unified_by_tc.py
    # or targeted:
    python scripts/split_codev_unified_by_tc.py --inputs <path> ...
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_INPUTS = [
    ROOT / "data" / "CodeV-SVA-datasets" / "grpo" / "codev_grpo_unified.jsonl",
    ROOT / "data" / "CodeV-SVA-datasets" / "sft" / "codev_sft_unified.jsonl",
]

L_TO_C = {1: "C1", 2: "C2", 3: "C2", 4: "C2", 5: "C3"}


def split_file(src: Path):
    out_handles = {}
    per_class = Counter()
    per_tcl = Counter()
    n = 0
    skipped_unknown_tcl = 0

    def _open(cls: str):
        if cls in out_handles:
            return out_handles[cls]
        out_path = src.with_name(f"{src.stem}_{cls}.jsonl")
        out_handles[cls] = open(out_path, "w")
        return out_handles[cls]

    with open(src) as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            n += 1
            tcl = row.get("expected_tcl")
            per_tcl[tcl] += 1
            cls = L_TO_C.get(tcl)
            if cls is None:
                skipped_unknown_tcl += 1
                continue
            row["temporal_class"] = cls
            per_class[cls] += 1
            _open(cls).write(json.dumps(row, ensure_ascii=False) + "\n")

    for fh in out_handles.values():
        fh.close()

    manifest = {
        "source": str(src),
        "policy": "C1={L1}, C2={L2,L3,L4}, C3={L5}",
        "n_rows": n,
        "n_written": sum(per_class.values()),
        "skipped_unknown_tcl": skipped_unknown_tcl,
        "per_tcl": {str(k): v for k, v in sorted(
            per_tcl.items(), key=lambda x: (x[0] is None, x[0]))},
        "per_class": {c: per_class[c] for c in ("C1", "C2", "C3")},
        "outputs": {
            c: str(src.with_name(f"{src.stem}_{c}.jsonl"))
            for c in per_class
        },
    }
    manifest_path = src.with_name(f"{src.stem}_tc_split_manifest.json")
    with open(manifest_path, "w") as fp:
        json.dump(manifest, fp, indent=2)
    return manifest, manifest_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="*", default=[str(p) for p in DEFAULT_INPUTS])
    args = ap.parse_args()

    for path in args.inputs:
        src = Path(path)
        if not src.exists():
            print(f"[skip] missing: {src}")
            continue
        print(f"\n[split] {src}")
        manifest, mpath = split_file(src)
        print(f"  rows:     {manifest['n_rows']} (written {manifest['n_written']}, "
              f"skipped {manifest['skipped_unknown_tcl']})")
        print(f"  by TCL:   {manifest['per_tcl']}")
        print(f"  by class: {manifest['per_class']}")
        for c, out in manifest["outputs"].items():
            print(f"    -> {Path(out).name}")
        print(f"  manifest: {mpath.name}")


if __name__ == "__main__":
    main()
