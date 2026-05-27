#!/usr/bin/env python3
"""
build_train_unified.py — merge the 6 per-source train JSONLs into one
deduplicated corpus + per-TCL stratified shards ready for curriculum SFT.

Dedup policy (cross-source, keep best record per SVA body):
  1. Normalize SVA body (collapse whitespace, strip wrapping asserts for
     hashing purposes — but keep original form in the output).
  2. If the same normalized body appears in multiple files, keep the record
     with the richest NL context (longest non-empty `nl`, ties broken by
     longest `rtl_context`, then by source priority below).

Source priority (for tie-breaking only):
  1. handcrafted_pilot   (human-authored NL, highest quality)
  2. nl2sva_machine      (GPT-generated NL aligned to SVA)
  3. named_properties    (preceding code comment as NL)
  4. github_scraped      (code comment as NL, often empty)
  5. opentitan_macros    (placeholder NL like "[ASSERT] xxx")
  6. snapshots_scraped   (same strategy, more noisy)

Outputs under data/train/unified/:
  train_unified.jsonl           — all dedup'd samples in one file
  train_unified_L1.jsonl ..  L5.jsonl — per-TCL shards for curriculum
  manifest.json                 — counts, TCL distribution, dedup report
"""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = EXPERIMENTS_DIR / "data" / "train"
OUT_DIR = TRAIN_DIR / "unified"

SOURCE_PRIORITY = {
    "handcrafted_pilot":     1,
    "nl2sva_machine":        2,
    "named_properties":      3,
    "github_scraped":        4,
    "opentitan_macros":      5,
    "snapshots_scraped":     6,
    "github_search_scraped": 7,
}

PER_SOURCE_FILES = [
    "handcrafted_pilot.jsonl",
    "nl2sva_machine.jsonl",
    "named_properties.jsonl",
    "github_scraped.jsonl",
    "opentitan_macros.jsonl",
    "snapshots_scraped.jsonl",
    "github_search_scraped.jsonl",
]


def normalize_body(sva: str) -> str:
    """Collapse whitespace in the SVA body for dedup."""
    return re.sub(r"\s+", " ", sva).strip()


def quality_score(rec: dict) -> tuple:
    """Higher tuple = richer record. Used to pick the best rec per body."""
    nl_len = len(rec.get("nl", "") or "")
    rtl_len = len(rec.get("rtl_context", "") or "")
    # Placeholder NL from opentitan_macros like "[ASSERT] xyz" — demote
    nl = rec.get("nl", "") or ""
    is_placeholder = bool(re.match(r"^\[(?:ASSERT|ASSUME|COVER)[^\]]*\]", nl))
    src_rank = SOURCE_PRIORITY.get(rec.get("source", ""), 99)
    # Priority: non-placeholder NL length > rtl length > source rank (lower=better)
    return (not is_placeholder, nl_len, rtl_len, -src_rank)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    by_body: dict[str, dict] = {}
    per_source_raw = Counter()
    per_source_kept = Counter()

    for fname in PER_SOURCE_FILES:
        path = TRAIN_DIR / fname
        if not path.exists():
            print(f"  [skip] {fname}: missing")
            continue
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                per_source_raw[rec["source"]] += 1
                body = normalize_body(rec["reference_sva"])
                if not body:
                    continue
                if body in by_body:
                    if quality_score(rec) > quality_score(by_body[body]):
                        by_body[body] = rec
                else:
                    by_body[body] = rec

    # Record which source each final winner came from
    for rec in by_body.values():
        per_source_kept[rec["source"]] += 1

    # Stratify by TCL
    per_tcl: dict[int, list] = defaultdict(list)
    for body, rec in by_body.items():
        # Ensure we use the normalized body in the output to avoid whitespace noise
        rec_out = dict(rec)
        rec_out["reference_sva"] = body
        per_tcl[int(rec.get("expected_tcl", 0))].append(rec_out)

    # Write combined
    all_records = []
    for lvl in (1, 2, 3, 4, 5):
        all_records.extend(per_tcl.get(lvl, []))
    with open(OUT_DIR / "train_unified.jsonl", "w") as f:
        for r in all_records:
            f.write(json.dumps(r) + "\n")

    # Write per-TCL shards
    for lvl in (1, 2, 3, 4, 5):
        with open(OUT_DIR / f"train_unified_L{lvl}.jsonl", "w") as f:
            for r in per_tcl.get(lvl, []):
                f.write(json.dumps(r) + "\n")

    # Manifest
    manifest = {
        "total_raw_samples": sum(per_source_raw.values()),
        "total_unique_bodies": len(by_body),
        "dedup_ratio": round(len(by_body) / max(sum(per_source_raw.values()), 1), 3),
        "per_source_raw": dict(per_source_raw),
        "per_source_kept_after_dedup": dict(per_source_kept),
        "per_tcl_counts": {f"L{lvl}": len(per_tcl.get(lvl, []))
                           for lvl in (1, 2, 3, 4, 5)},
        "nl_populated": sum(1 for r in by_body.values() if (r.get("nl") or "").strip()),
        "rtl_context_populated": sum(
            1 for r in by_body.values() if (r.get("rtl_context") or "").strip()
        ),
        "schema_fields": sorted(
            {k for r in list(by_body.values())[:100] for k in r.keys()}
        ),
        "output_files": [
            "train_unified.jsonl",
            *[f"train_unified_L{lvl}.jsonl" for lvl in (1, 2, 3, 4, 5)],
        ],
    }
    with open(OUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Console summary
    print("=" * 60)
    print("UNIFIED TRAIN BUILD")
    print("=" * 60)
    print(f"Raw samples (with cross-source dup): "
          f"{sum(per_source_raw.values()):>6}")
    print(f"Unique SVA bodies after dedup:       "
          f"{len(by_body):>6}")
    print(f"Dedup ratio:                         "
          f"{len(by_body)/max(sum(per_source_raw.values()),1):.3f}")
    print()
    print(f"Per-source (raw → kept after dedup):")
    for src in SOURCE_PRIORITY:
        raw = per_source_raw.get(src, 0)
        kept = per_source_kept.get(src, 0)
        print(f"  {src:<22} {raw:>6} → {kept:>6}")
    print()
    print(f"Per-TCL shards:")
    for lvl in (1, 2, 3, 4, 5):
        print(f"  L{lvl}: {len(per_tcl.get(lvl, [])):>5}   "
              f"→ {OUT_DIR}/train_unified_L{lvl}.jsonl")
    print()
    print(f"NL populated:          "
          f"{manifest['nl_populated']}/{len(by_body)} "
          f"({100*manifest['nl_populated']/len(by_body):.1f}%)")
    print(f"RTL context populated: "
          f"{manifest['rtl_context_populated']}/{len(by_body)} "
          f"({100*manifest['rtl_context_populated']/len(by_body):.1f}%)")
    print()
    print(f"Manifest: {OUT_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()
