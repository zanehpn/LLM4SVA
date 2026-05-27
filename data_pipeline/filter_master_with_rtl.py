#!/usr/bin/env python3
"""
filter_master_with_rtl.py — keep only rows whose original rtl_context
is non-empty. For master_train_synthetic_grpo.jsonl (which has 100%
rtl after synthesis), use the corresponding row in
master_train_synthetic.jsonl as the authority on whether the rtl was
originally present.

Files are rewritten atomically in place. Manifests are updated.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "data" / "master"


def has_real_rtl(row: dict) -> bool:
    rtl = (row.get("rtl_context") or "").strip()
    return bool(rtl)


def filter_in_place(src: Path, keep_pred):
    tmp = src.with_suffix(src.suffix + ".tmp")
    n_in = n_out = 0
    with open(src) as fin, open(tmp, "w") as fout:
        for line in fin:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            row = json.loads(line)
            n_in += 1
            if keep_pred(row):
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_out += 1
    os.replace(tmp, src)
    return n_in, n_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master-dir", default=str(ROOT))
    args = ap.parse_args()
    md = Path(args.master_dir)

    # Step 1: filter master_train.jsonl, real, synthetic — by rtl_context presence
    targets_simple = [
        md / "master_train.jsonl",
        md / "master_train_real.jsonl",
        md / "master_train_synthetic.jsonl",
    ]
    print("[step 1] filtering files by has(rtl_context):")
    for src in targets_simple:
        if not src.exists():
            print(f"  [skip] missing: {src.name}")
            continue
        n_in, n_out = filter_in_place(src, has_real_rtl)
        print(f"  {src.name}: {n_in} → {n_out}  (-{n_in-n_out})")

    # Step 2: filter master_train_synthetic_grpo.jsonl to ids that survived
    # filtering of master_train_synthetic.jsonl (i.e. originally had rtl).
    syn_path = md / "master_train_synthetic.jsonl"
    grpo_path = md / "master_train_synthetic_grpo.jsonl"
    print("\n[step 2] filtering synthetic_grpo by survival of synthetic.jsonl:")
    if not syn_path.exists() or not grpo_path.exists():
        print("  [skip] required files missing")
    else:
        kept_ids = set()
        with open(syn_path) as f:
            for line in f:
                if line.strip():
                    kept_ids.add(json.loads(line)["id"])
        n_in, n_out = filter_in_place(
            grpo_path,
            lambda r: r.get("id") in kept_ids,
        )
        print(f"  {grpo_path.name}: {n_in} → {n_out}  (-{n_in-n_out})")

    # Step 3: refresh manifest
    print("\n[step 3] refreshing manifests:")
    from collections import Counter
    for f in [md / "master_train.jsonl", md / "master_train_real.jsonl",
              md / "master_train_synthetic.jsonl",
              md / "master_train_synthetic_grpo.jsonl"]:
        if not f.exists(): continue
        n = 0; cls = Counter(); tcl = Counter()
        for line in open(f):
            if not line.strip(): continue
            r = json.loads(line)
            n += 1
            cls[r.get("temporal_class")] += 1
            tcl[r.get("expected_tcl")] += 1
        print(f"  {f.name}: {n} rows, "
              f"class={dict(cls)}, "
              f"tcl={dict(sorted(tcl.items(), key=lambda x:(x[0] is None, x[0])))}")


if __name__ == "__main__":
    main()
