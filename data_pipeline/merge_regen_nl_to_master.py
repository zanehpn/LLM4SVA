#!/usr/bin/env python3
"""
merge_regen_nl_to_master.py — apply GPT-5.1 NL regenerations into the
master_train.jsonl, preserving original engineer/curated NL for rows
that weren't in the regen scope.

Each output row gets a new field `nl_origin`:
  - "engineer"          : kept verbatim from S/A tier (no LLM rewrite)
  - "gpt5_regen"        : NL replaced by validated GPT-5.1 output
  - "regen_failed_keep" : regen attempted but produced sva_token_leak;
                          original NL kept (logged for follow-up)
  - "untouched"         : neither in regen scope nor S/A engineer (rare;
                          B-tier LLM-filled / unknown provenance)

Atomic write via .tmp + os.replace.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", default="data/master/master_train.jsonl")
    ap.add_argument("--regen", default="data/master/master_train.nl_regen.jsonl")
    ap.add_argument("--output", default="data/master/master_train.jsonl",
                    help="defaults to overwriting master in-place")
    ap.add_argument("--needs-regen-list",
                    default="data/master/master_train_needs_regen.jsonl",
                    help="ids in this file = had been queued for regen")
    args = ap.parse_args()

    # Build {id: regen_record}
    regen_by_id = {}
    with open(args.regen) as f:
        for line in f:
            r = json.loads(line)
            regen_by_id[r["id"]] = r

    needs_regen_ids = set()
    with open(args.needs_regen_list) as f:
        for line in f:
            needs_regen_ids.add(json.loads(line)["id"])

    print(f"[in] regen records: {len(regen_by_id)}")
    print(f"[in] needs_regen ids: {len(needs_regen_ids)}")

    src = Path(args.master)
    dst = Path(args.output)
    tmp = dst.with_suffix(dst.suffix + ".tmp")

    origin_counts = Counter()
    with open(src) as fin, open(tmp, "w") as fout:
        for line in fin:
            r = json.loads(line)
            tid = r["id"]
            if tid in regen_by_id:
                rec = regen_by_id[tid]
                if rec["status"] == "ok":
                    r["nl_original"] = r.get("nl", "")
                    r["nl"] = rec["nl_new"]
                    r["nl_origin"] = "gpt5_regen"
                else:
                    r["nl_origin"] = "regen_failed_keep"
            elif tid in needs_regen_ids:
                # In needs-regen list but somehow no regen record
                r["nl_origin"] = "regen_missing"
            else:
                r["nl_origin"] = "engineer"
            origin_counts[r["nl_origin"]] += 1
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, dst)

    print(f"\n[out] wrote {dst}")
    print(f"[out] nl_origin distribution:")
    total = sum(origin_counts.values())
    for k, v in origin_counts.most_common():
        print(f"  {k:<22s} {v:>6d}  ({100*v/total:.2f}%)")


if __name__ == "__main__":
    main()
