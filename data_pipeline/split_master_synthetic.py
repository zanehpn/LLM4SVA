#!/usr/bin/env python3
"""
split_master_synthetic.py — split master_train.jsonl into synthetic vs
real-engineering subsets, based on the provenance field.

A row is **synthetic** iff every file in its `provenance` came from a
synthetic source (expand_tcl/method3 + their aggregates).

A row is **real** iff at least one provenance entry is a non-synthetic
source (scrapers, handcrafted, opentitan, named_properties, sft_train,
grpo_pool_*, etc.). Rows that appear in BOTH a real and synthetic file
are treated as real — the SVA exists in a real-engineering corpus and
the synthetic file just happens to also include it.

Outputs (siblings of master_train.jsonl):
    master_train_real.jsonl
    master_train_synthetic.jsonl
    master_train_split_manifest.json

Usage:
    python scripts/split_master_synthetic.py
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "data" / "master"

# Provenance file paths that count as "synthetic" — purely programmatic
# expansions where the NL is template-generated and the SVA is constructed
# by transforming an L4 reference into L3/L5 variants.
SYNTHETIC_SOURCES = {
    "expand_tcl/method3_synthetic_l3_l5_with_rtl_filled.jsonl",
    "expand_tcl/all_methods_merged.jsonl",
    "expand_tcl/expand_tc_C1.jsonl",
    "expand_tcl/expand_tc_C2.jsonl",
    "expand_tcl/expand_tc_C3.jsonl",
    "expand_tcl/method1_existing_l3_l5.jsonl",
    "expand_tcl/method2_github_high_tcl.jsonl",
}


def is_synthetic(prov: list) -> bool:
    """Return True iff all provenance entries are synthetic sources."""
    if not prov:
        return False
    return all(p in SYNTHETIC_SOURCES for p in prov)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(ROOT / "master_train.jsonl"))
    ap.add_argument("--out-dir", default=str(ROOT))
    args = ap.parse_args()

    src = Path(args.input)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    real_path = out / "master_train_real.jsonl"
    syn_path = out / "master_train_synthetic.jsonl"

    n_real = n_syn = 0
    real_class = Counter(); syn_class = Counter()
    real_tcl = Counter(); syn_tcl = Counter()

    with open(src) as fin, \
         open(real_path, "w") as f_real, \
         open(syn_path, "w") as f_syn:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prov = row.get("provenance", [])
            if is_synthetic(prov):
                f_syn.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_syn += 1
                syn_class[row.get("temporal_class")] += 1
                syn_tcl[row.get("expected_tcl")] += 1
            else:
                f_real.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_real += 1
                real_class[row.get("temporal_class")] += 1
                real_tcl[row.get("expected_tcl")] += 1

    manifest = {
        "policy": (
            "split master_train.jsonl into real vs synthetic by provenance. "
            "synthetic = every provenance entry is in expand_tcl/method*; "
            "real = at least one non-synthetic provenance entry. Rows in "
            "BOTH classes are placed in real (real corpus is authoritative)."
        ),
        "synthetic_sources": sorted(SYNTHETIC_SOURCES),
        "n_real": n_real,
        "n_synthetic": n_syn,
        "n_total": n_real + n_syn,
        "real": {
            "temporal_class": {str(k): v for k, v in real_class.items()},
            "expected_tcl": {str(k): v for k, v in sorted(
                real_tcl.items(), key=lambda x: (x[0] is None, x[0]))},
        },
        "synthetic": {
            "temporal_class": {str(k): v for k, v in syn_class.items()},
            "expected_tcl": {str(k): v for k, v in sorted(
                syn_tcl.items(), key=lambda x: (x[0] is None, x[0]))},
        },
        "outputs": {
            "real": str(real_path),
            "synthetic": str(syn_path),
        },
    }
    manifest_path = out / "master_train_split_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[split] master_train.jsonl ({n_real + n_syn} rows)")
    print(f"  real      : {n_real:>7d}  ({100*n_real/(n_real+n_syn):.1f}%)")
    print(f"    -> {real_path}")
    print(f"  synthetic : {n_syn:>7d}  ({100*n_syn/(n_real+n_syn):.1f}%)")
    print(f"    -> {syn_path}")
    print(f"  manifest  : {manifest_path}")
    print(f"\n[real] temporal_class: {dict(real_class)}")
    print(f"[real] expected_tcl: "
          f"{dict(sorted(real_tcl.items(), key=lambda x: (x[0] is None, x[0])))}")
    print(f"\n[synthetic] temporal_class: {dict(syn_class)}")
    print(f"[synthetic] expected_tcl: "
          f"{dict(sorted(syn_tcl.items(), key=lambda x: (x[0] is None, x[0])))}")


if __name__ == "__main__":
    main()
