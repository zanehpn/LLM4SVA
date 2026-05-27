#!/usr/bin/env python3
"""
merge_all_train.py — aggregate every jsonl under data/train/ into one unified
file with a canonical schema, dedup by SVA body hash, and keep the record
with the richest non-empty fields (prefer: real NL, longer rtl_context,
non-generic source).

Output schema per record:
  {
    "id":            stable identifier (first source's id or hash prefix)
    "source":        "+" -joined list of source files/tags
    "nl":            natural-language description (possibly empty)
    "reference_sva": SVA body
    "rtl_context":   surrounding HDL text (possibly empty)
    "expected_tcl":  TCL level 1-5
    "hash":          SHA-256 prefix of normalized SVA body
    "nl_filled_by":  LLM id if NL is LLM-generated (optional)
  }

Sources ingested:
  - data/train/*.jsonl                       (7 source jsonls)
  - data/train/sft/sft_train_L{1..5}.jsonl   (curriculum SFT pool)
  - data/train/sft/sft_train.jsonl           (merged pre-stratification)
  - data/train/grpo/grpo_pool_*.jsonl        (all GRPO pools)
  - data/train/grpo/scrape_all_rtl_nl.jsonl  (latest NL+RTL scrape)
  - data/train/ipo/ipo_pairs.jsonl           (preference pairs — only chosen)
  - data/train/unified/train_unified*.jsonl  (canonical unified)
  - data/train/all_real_nl_pairs.jsonl       (meaningful-NL aggregate)
"""
import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = EXPERIMENTS_DIR / "data" / "train"
OUT_FILE = TRAIN_DIR / "all_merged_train.jsonl"
MANIFEST = TRAIN_DIR / "all_merged_manifest.json"


def body_hash(sva: str) -> str:
    s = re.sub(r"\s+", " ", sva or "").strip().lower()
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def classify_tcl(sva: str) -> int:
    s = (sva or "").lower()
    if any(k in s for k in ("s_eventually", "s_until", "s_always", "nexttime")):
        return 5
    if "|->" in sva or "|=>" in sva:
        return 4
    if "##[" in sva:
        return 3
    if re.search(r"##\s*\d+", sva):
        return 2
    return 1


def is_valid(sva: str) -> bool:
    s = (sva or "").lower().strip()
    return "assert" in s and "property" in s and 30 <= len(s) <= 4000


PLACEHOLDER_RE = re.compile(r"^\s*\[(ASSERT|ASSUME|COVER)", re.I)


def is_real_nl(nl: str) -> bool:
    nl = (nl or "").strip()
    return len(nl) >= 10 and not PLACEHOLDER_RE.match(nl)


def normalize(rec: dict, source: str) -> dict | None:
    """Extract canonical fields; return None if record can't be normalized."""
    sva = (rec.get("reference_sva") or rec.get("sva") or
           rec.get("gold_sva") or "")
    if not is_valid(sva):
        return None
    nl = (rec.get("nl") or rec.get("nl_comment") or rec.get("description")
          or "").strip()
    rtl = (rec.get("rtl_context") or rec.get("rtl") or "").strip()
    tcl = rec.get("expected_tcl") or rec.get("tcl") or classify_tcl(sva)
    try:
        tcl = int(tcl)
    except (ValueError, TypeError):
        tcl = classify_tcl(sva)
    h = body_hash(sva)
    # pass through optional LLM-fill marker
    nl_filled_by = rec.get("nl_filled_by") or rec.get("nl_source", "")
    return {
        "id": rec.get("id", f"{source[:8]}_{h}"),
        "source": source,
        "nl": nl,
        "reference_sva": sva,
        "rtl_context": rtl,
        "expected_tcl": tcl,
        "hash": h,
        "nl_filled_by": nl_filled_by,
    }


def score_richness(r: dict) -> int:
    """Higher score = richer record; used for dedup tiebreaking."""
    s = 0
    if is_real_nl(r["nl"]):
        s += 4
    if r["rtl_context"] and len(r["rtl_context"]) >= 500:
        s += 2
    elif r["rtl_context"]:
        s += 1
    if r.get("nl_filled_by"):
        # slight preference for clearly-labelled records
        s += 1
    return s


SOURCES = [
    # (glob relative to TRAIN_DIR, source-tag, priority)
    ("*.jsonl",                             "raw",       1),
    ("sft/sft_train.jsonl",                 "sft_merged", 2),
    ("sft/sft_train_L*.jsonl",              "sft_L",     2),
    ("grpo/grpo_pool_*.jsonl",              "grpo_pool", 3),
    ("grpo/scrape_all_rtl_nl.jsonl",        "scrape_all_rtl_nl", 4),
    ("ipo/ipo_pairs.jsonl",                 "ipo",       3),
    ("unified/train_unified*.jsonl",        "unified",   1),
    ("all_real_nl_pairs.jsonl",             "real_nl",   5),   # highest priority
]


def expand(glob_pat: str):
    return sorted((TRAIN_DIR / glob_pat).parent.glob(Path(glob_pat).name))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_FILE))
    ap.add_argument("--manifest", default=str(MANIFEST))
    args = ap.parse_args()

    merged: dict[str, dict] = {}   # hash -> best record
    stats = Counter()
    source_counts = Counter()

    for pat, tag, priority in SOURCES:
        files = expand(pat) if "*" in pat else [TRAIN_DIR / pat]
        files = [f for f in files if f.exists()]
        for f in files:
            fn = f.name
            ingested = 0
            for line in open(f):
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    stats["json_err"] += 1
                    continue
                # IPO pair: pull chosen.content as SVA
                if "chosen" in raw and isinstance(raw["chosen"], list):
                    content = raw["chosen"][-1].get("content", "") if raw["chosen"] else ""
                    m = re.search(r"```(?:systemverilog|sv)?\s*(.+?)```", content, re.DOTALL)
                    if m:
                        content = m.group(1).strip()
                    m = re.search(r"(assert\s+property\s*\(.*?\)\s*;)", content, re.DOTALL | re.I)
                    sva_from_ipo = m.group(1).strip() if m else content.strip()
                    rec = {
                        "sva": sva_from_ipo,
                        "nl": "",  # IPO prompt has NL but we don't parse it back here
                    }
                else:
                    rec = raw
                norm = normalize(rec, tag)
                if norm is None:
                    stats["invalid_sva"] += 1
                    continue
                ingested += 1
                h = norm["hash"]
                if h not in merged:
                    merged[h] = norm
                    stats["new"] += 1
                else:
                    stats["dup"] += 1
                    # prefer the richer record
                    if score_richness(norm) > score_richness(merged[h]):
                        # keep id from the earlier seen (stable identity)
                        norm["id"] = merged[h]["id"]
                        norm["source"] = merged[h]["source"] + "+" + tag
                        merged[h] = norm
                    else:
                        merged[h]["source"] = merged[h]["source"] + "+" + tag
            source_counts[fn] = ingested

    # Stratify stats
    tcl_dist = Counter(r["expected_tcl"] for r in merged.values())
    n_real_nl = sum(1 for r in merged.values() if is_real_nl(r["nl"]))
    n_with_rtl = sum(1 for r in merged.values()
                     if r["rtl_context"] and len(r["rtl_context"]) >= 500)
    n_gold = sum(1 for r in merged.values()
                 if is_real_nl(r["nl"]) and len(r["rtl_context"]) >= 500)

    # Write
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in merged.values():
            f.write(json.dumps(r) + "\n")

    manifest = {
        "total_unique": len(merged),
        "tcl_dist": dict(tcl_dist),
        "real_nl": n_real_nl,
        "real_nl_pct": round(100 * n_real_nl / max(1, len(merged)), 2),
        "with_rtl_ge_500": n_with_rtl,
        "gold_triple": n_gold,
        "source_ingested_counts": dict(source_counts),
        "dedup_stats": dict(stats),
    }
    with open(args.manifest, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[write] {args.out}: {len(merged)} unique records")
    print(f"[tcl]   {dict(tcl_dist)}")
    print(f"[nl]    real NL: {n_real_nl} ({100*n_real_nl/len(merged):.1f}%)")
    print(f"[rtl]   with RTL≥500: {n_with_rtl} ({100*n_with_rtl/len(merged):.1f}%)")
    print(f"[gold]  real NL + RTL≥500: {n_gold} ({100*n_gold/len(merged):.1f}%)")
    print(f"[stats] {dict(stats)}")
    print(f"[source counts] {dict(sorted(source_counts.items(), key=lambda x: -x[1])[:15])}")
    print(f"[manifest] {args.manifest}")


if __name__ == "__main__":
    main()
