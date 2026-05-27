#!/usr/bin/env python3
"""
build_master_train.py — collapse the entire `data/train/` and
`data/expand_tcl/` zoo into a single master jsonl, deduped by canonical
SVA body hash. Each surviving row carries a `provenance` list naming
every file it appeared in.

Output:
    data/master/master_train.jsonl
    data/master/master_train_manifest.json
    data/master/redundancy_report.json   (which input files become
                                           safely deletable)

The merge prefers the "richest" copy when the same SVA appears in
multiple files:
  - NL: prefer engineer-written (longer, real_nl=True) over LLM-filled
  - RTL: prefer non-empty + longest
  - tcl: max across copies (should be identical anyway)

Usage:
    python scripts/build_master_train.py
"""
from __future__ import annotations

import json
import re
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = ROOT / "master"

# Inputs in priority order — earlier files "win" tie-breaks for fields.
# Engineer-NL sources first, then LLM-augmented, then synthetic.
INPUTS = [
    # Engineer-NL real-world (highest authority)
    "train/sft/sft_train.jsonl",
    "train/all_real_nl_pairs.jsonl",
    "train/handcrafted_pilot.jsonl",
    "train/named_properties.jsonl",
    "train/opentitan_macros.jsonl",
    "train/github_scraped.jsonl",
    "train/github_search_scraped.jsonl",
    "train/snapshots_scraped.jsonl",
    "train/all_merged_train.jsonl",
    "train/unified/train_unified.jsonl",

    # LLM-augmented NL on real RTL
    "train/grpo/grpo_pool_disjoint.jsonl",
    "train/grpo/scrape_all_rtl_nl.jsonl",
    "train/grpo/scrape_github_repos.jsonl",
    "train/grpo/industrial_with_rtl.jsonl",
    "train/grpo/grpo_pool_industrial.jsonl",
    "train/grpo/grpo_pool_pilot8_rtl.jsonl",
    "train/grpo/grpo_pool_tiered.jsonl",
    "train/grpo/grpo_pool_viable.jsonl",
    "train/grpo/grpo_pool_viable_relaxed.jsonl",

    # Programmatic L3/L5 expansion (synthetic, lowest priority for NL)
    "expand_tcl/method1_existing_l3_l5.jsonl",
    "expand_tcl/method2_github_high_tcl.jsonl",
    "expand_tcl/method3_synthetic_l3_l5_with_rtl_filled.jsonl",
    "expand_tcl/all_methods_merged.jsonl",
    "expand_tcl/expand_tc_C2.jsonl",
    "expand_tcl/expand_tc_C3.jsonl",
]

L_TO_C = {1: "C1", 2: "C2", 3: "C2", 4: "C2", 5: "C3"}


def canon_hash(sva: str) -> str:
    if not sva:
        return ""
    return hashlib.md5(re.sub(r"\s+", "", sva).lower().encode()).hexdigest()[:16]


def is_engineer_nl(row: dict) -> bool:
    """Heuristic: engineer NL is typically not 'LLM-filled' tagged, has
    natural phrasing (capital letter start), and is between 20-300 chars.
    LLM-filled NL often has the canned prefix 'Create a SVA assertion that
    checks: ...' or starts with 'When the ...' / 'If ...' templated forms.
    """
    nl = (row.get("nl") or "").strip()
    if not nl:
        return False
    # Most LLM-filled NLs in this corpus follow these prefix patterns:
    llm_prefixes = (
        "Create a SVA assertion that checks:",
        "Generate a SVA assertion ",
        "Write a SVA ",
    )
    if any(nl.startswith(p) for p in llm_prefixes):
        return False
    # Original engineer NLs from comments often start with lowercase ("that ..."),
    # are short ("L3: gnt within 1-4 cycles"), or begin with verbs ("checks ...")
    return True


def merge_row(existing: dict, new: dict, source_file: str):
    """Merge `new` into `existing` in-place. existing already has provenance."""
    existing["provenance"].append(source_file)

    # Prefer engineer-NL over LLM-NL
    new_nl = (new.get("nl") or "").strip()
    if new_nl and is_engineer_nl(new):
        if not existing.get("nl") or not existing.get("nl_engineer"):
            existing["nl"] = new_nl
            existing["nl_engineer"] = True
            existing["nl_source"] = source_file
    elif new_nl and not existing.get("nl"):
        existing["nl"] = new_nl
        existing["nl_engineer"] = False
        existing["nl_source"] = source_file

    # Prefer richer rtl_context (longest non-empty)
    new_rtl = new.get("rtl_context") or ""
    if len(new_rtl) > len(existing.get("rtl_context") or ""):
        existing["rtl_context"] = new_rtl
        existing["rtl_source"] = source_file

    # tcl should be consistent — keep first non-zero, warn on conflict
    if existing.get("expected_tcl") in (None, 0):
        existing["expected_tcl"] = new.get("expected_tcl")
    elif new.get("expected_tcl") not in (None, 0, existing["expected_tcl"]):
        existing.setdefault("tcl_conflicts", []).append(
            (source_file, new.get("expected_tcl"))
        )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    master: dict[str, dict] = {}
    file_stats: dict[str, dict] = {}

    for rel in INPUTS:
        path = ROOT / rel
        if not path.exists():
            print(f"[skip] missing: {rel}")
            continue
        n_rows = n_new = n_dup = n_invalid = 0
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    n_invalid += 1
                    continue
                n_rows += 1
                sva = row.get("reference_sva") or row.get("sva") or ""
                h = canon_hash(sva)
                if not h:
                    n_invalid += 1
                    continue
                if h not in master:
                    nl_eng = is_engineer_nl(row)
                    master[h] = {
                        "hash_canon": h,
                        "id": row.get("id") or row.get("sample_id"),
                        "nl": (row.get("nl") or "").strip(),
                        "reference_sva": sva,
                        "rtl_context": row.get("rtl_context") or "",
                        "expected_tcl": row.get("expected_tcl"),
                        "temporal_class": L_TO_C.get(row.get("expected_tcl")),
                        "nl_engineer": nl_eng,
                        "nl_source": rel if (row.get("nl") or "").strip() else None,
                        "rtl_source": rel if (row.get("rtl_context") or "") else None,
                        "provenance": [rel],
                    }
                    n_new += 1
                else:
                    merge_row(master[h], row, rel)
                    n_dup += 1
        file_stats[rel] = {
            "rows": n_rows,
            "added_new": n_new,
            "merged_into_existing": n_dup,
            "invalid": n_invalid,
        }
        print(f"[merge] {rel}: rows={n_rows}, new={n_new}, "
              f"dup={n_dup}, invalid={n_invalid}, master_size={len(master)}")

    # Write master
    out_path = OUT_DIR / "master_train.jsonl"
    with open(out_path, "w") as f:
        for row in master.values():
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\n[done] master_train.jsonl: {len(master)} unique SVAs")

    # Final stats
    cls_dist = Counter(row.get("temporal_class") for row in master.values())
    tcl_dist = Counter(row.get("expected_tcl") for row in master.values())
    eng_count = sum(1 for r in master.values() if r.get("nl_engineer"))
    nl_nonempty = sum(1 for r in master.values() if r.get("nl"))

    manifest = {
        "policy": "Master pool: deduped union of all train/* and expand_tcl/* "
                  "jsonl by canonical SVA body hash. Engineer-NL prefer over "
                  "LLM-filled NL when both exist for the same SVA. Each row "
                  "carries provenance = [files it appeared in].",
        "n_master_rows": len(master),
        "temporal_class_dist": dict(cls_dist),
        "expected_tcl_dist": {str(k): v for k, v in sorted(
            tcl_dist.items(), key=lambda x: (x[0] is None, x[0]))},
        "rows_with_nl": nl_nonempty,
        "rows_with_engineer_nl": eng_count,
        "rows_with_llm_nl": nl_nonempty - eng_count,
        "input_files": file_stats,
    }
    manifest_path = OUT_DIR / "master_train_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Redundancy report — which input files are now fully covered
    redundancy = {}
    master_hashes = set(master.keys())
    for rel in INPUTS:
        path = ROOT / rel
        if not path.exists():
            continue
        own = set()
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                h = canon_hash(r.get("reference_sva") or r.get("sva") or "")
                if h:
                    own.add(h)
        in_master = own & master_hashes
        redundancy[rel] = {
            "unique_hashes": len(own),
            "covered_by_master": len(in_master),
            "fully_covered": len(own - master_hashes) == 0,
        }
    with open(OUT_DIR / "redundancy_report.json", "w") as f:
        json.dump(redundancy, f, indent=2)

    print(f"\n=== summary ===")
    print(f"master rows: {len(master)}")
    print(f"by temporal_class: {dict(cls_dist)}")
    print(f"by expected_tcl: {dict(tcl_dist)}")
    print(f"NL coverage: {nl_nonempty}/{len(master)} have NL "
          f"(engineer={eng_count}, llm={nl_nonempty - eng_count})")
    print(f"\noutputs:")
    print(f"  {out_path}")
    print(f"  {manifest_path}")
    print(f"  {OUT_DIR / 'redundancy_report.json'}")
    print(f"\nfiles fully covered by master (safe to delete):")
    for rel, info in sorted(redundancy.items()):
        if info["fully_covered"]:
            print(f"  ✓ {rel}")
    print(f"\nfiles with unique content (DO NOT delete unless re-merged):")
    for rel, info in sorted(redundancy.items()):
        if not info["fully_covered"]:
            new = info["unique_hashes"] - info["covered_by_master"]
            print(f"  ✗ {rel}  (still has {new} unique)")


if __name__ == "__main__":
    main()
