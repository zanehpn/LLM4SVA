#!/usr/bin/env python3
"""
scrape_snapshots.py — aggressive three-way SVA extraction over local RTL
snapshot pools (PR-based checkouts of SV/V projects):

  ${SVA_CORPUS_ROOT}/HWE-train/snapshots   (486 snapshots, ~20 GB)
  ${SVA_CORPUS_ROOT}/HDL_all/snapshots     (135 snapshots, ~3 GB)

For every .sv / .svh / .v file we run ALL THREE extraction strategies in a
single walk:

  (A) inline  — from scrape_github_sva.extract_from_file
  (B) macro   — from expand_opentitan_macros.scan_file
  (C) named   — from scrape_named_properties.extract_from_file

Dedup: SHA256(whitespace-collapsed SVA body) — one entry per unique SVA
across all snapshots and strategies.

Output: data/raw/snapshots_scraped/scraped.jsonl
  Schema: {source_snapshot, strategy, sva, nl_comment, module, file, line,
           body_hash}
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scrape_github_sva import extract_from_file as inline_extract
from scrape_named_properties import extract_from_file as named_extract
from expand_opentitan_macros import scan_file as macro_scan

OUT_DIR = EXPERIMENTS_DIR / "data" / "raw" / "snapshots_scraped"
OUT_JSONL = OUT_DIR / "scraped.jsonl"

SV_EXTS = {".sv", ".svh", ".v", ".vh"}
DEFAULT_ROOTS = [
    Path("${SVA_CORPUS_ROOT}/HWE-train/snapshots"),              # 486 snaps
    Path("${SVA_CORPUS_ROOT}/HDL_all/snapshots"),                # 135
    Path("${SVA_CORPUS_ROOT}/HDL_verified_new/testbench"),       # 78, verified
    Path("${SVA_CORPUS_ROOT}/HDL_verified_new_useless/snapshots_top100"),  # 100
    Path("${SVA_CORPUS_ROOT}/Local_agent/tmp/iter_memory1"),     # 1243
    Path("${SVA_CORPUS_ROOT}/Other_agent/KGCompass/workdir_hdl"),# 752
]


def _norm(sva: str) -> str:
    return re.sub(r"\s+", " ", sva).strip()


def _hash(sva: str) -> str:
    return hashlib.sha256(_norm(sva).encode()).hexdigest()[:16]


def walk_and_extract(root: Path, seen: set) -> Iterable[Dict]:
    """Walk one snapshot root, yield normalized SVA records."""
    for sn_dir in sorted(root.iterdir()):
        if not sn_dir.is_dir():
            continue
        repo_keep = 0
        for f in sn_dir.rglob("*"):
            if not f.is_file() or f.suffix.lower() not in SV_EXTS:
                continue
            # Skip vendored / obvious noise
            if any(p in (".git", "build", "_out", "node_modules")
                   for p in f.parts):
                continue
            # --- (A) inline ---
            try:
                for rec in inline_extract(f):
                    sva = rec.get("sva", "")
                    if not sva:
                        continue
                    h = _hash(sva)
                    if h in seen:
                        continue
                    seen.add(h)
                    repo_keep += 1
                    yield {
                        "source_snapshot": sn_dir.name,
                        "strategy": "inline",
                        "sva": _norm(sva),
                        "nl_comment": rec.get("nl_comment", ""),
                        "module": rec.get("module", ""),
                        "file": str(f.relative_to(sn_dir)),
                        "line": rec.get("line", 0),
                        "body_hash": h,
                    }
            except Exception:
                pass
            # --- (B) macro ---
            try:
                for rec in macro_scan(f):
                    sva = rec.get("sva", "")
                    if not sva:
                        continue
                    h = _hash(sva)
                    if h in seen:
                        continue
                    seen.add(h)
                    repo_keep += 1
                    yield {
                        "source_snapshot": sn_dir.name,
                        "strategy": f"macro:{rec.get('macro','?')}",
                        "sva": _norm(sva),
                        "nl_comment": "",
                        "module": "",
                        "file": str(f.relative_to(sn_dir)),
                        "line": rec.get("line", 0),
                        "body_hash": h,
                    }
            except Exception:
                pass
            # --- (C) named property ---
            try:
                for rec in named_extract(f):
                    sva = rec.get("sva", "")
                    if not sva:
                        continue
                    h = _hash(sva)
                    if h in seen:
                        continue
                    seen.add(h)
                    repo_keep += 1
                    yield {
                        "source_snapshot": sn_dir.name,
                        "strategy": "named",
                        "sva": _norm(sva),
                        "nl_comment": rec.get("nl_comment", ""),
                        "module": rec.get("module", ""),
                        "file": str(f.relative_to(sn_dir)),
                        "line": rec.get("line", 0),
                        "body_hash": h,
                    }
            except Exception:
                pass
        if repo_keep:
            print(f"  [{root.name}/{sn_dir.name}] +{repo_keep}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", default=[str(r) for r in DEFAULT_ROOTS])
    ap.add_argument("--limit", type=int, default=0,
                    help="debug: only process first N snapshots per root")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seen = set()
    n_total = 0
    by_strategy = {}
    by_snapshot = {}

    with open(OUT_JSONL, "w") as out:
        for root_str in args.roots:
            root = Path(root_str)
            if not root.exists():
                print(f"[warn] root missing: {root}")
                continue
            print(f"\n=== scanning {root} ===")
            snaps = sorted([d for d in root.iterdir() if d.is_dir()])
            if args.limit:
                snaps = snaps[:args.limit]
            for sn in snaps:
                before = n_total
                for rec in walk_and_extract_one(sn, seen):
                    out.write(json.dumps(rec) + "\n")
                    n_total += 1
                    by_strategy[rec["strategy"]] = by_strategy.get(rec["strategy"], 0) + 1
                    by_snapshot[sn.name] = by_snapshot.get(sn.name, 0) + 1
                added = n_total - before
                if added:
                    print(f"  [{root.name}/{sn.name}] +{added}")

    print(f"\n[snapshots] total unique SVAs written: {n_total}")
    print(f"[snapshots] output: {OUT_JSONL}")
    print(f"\nBy strategy:")
    for s, n in sorted(by_strategy.items(), key=lambda x: -x[1]):
        print(f"  {s:<22} {n}")
    print(f"\nTop 15 snapshots:")
    for s, n in sorted(by_snapshot.items(), key=lambda x: -x[1])[:15]:
        print(f"  {s:<55} {n}")


def walk_and_extract_one(sn_dir: Path, seen: set):
    """Same as walk_and_extract but for a single snapshot dir."""
    for f in sn_dir.rglob("*"):
        if not f.is_file() or f.suffix.lower() not in SV_EXTS:
            continue
        if any(p in (".git", "build", "_out", "node_modules") for p in f.parts):
            continue
        # inline
        try:
            for rec in inline_extract(f):
                sva = rec.get("sva", "")
                if not sva: continue
                h = _hash(sva)
                if h in seen: continue
                seen.add(h)
                yield {"source_snapshot": sn_dir.name, "strategy": "inline",
                       "sva": _norm(sva), "nl_comment": rec.get("nl_comment", ""),
                       "module": rec.get("module", ""),
                       "file": str(f.relative_to(sn_dir)),
                       "line": rec.get("line", 0), "body_hash": h}
        except Exception: pass
        # macro
        try:
            for rec in macro_scan(f):
                sva = rec.get("sva", "")
                if not sva: continue
                h = _hash(sva)
                if h in seen: continue
                seen.add(h)
                yield {"source_snapshot": sn_dir.name,
                       "strategy": f"macro:{rec.get('macro','?')}",
                       "sva": _norm(sva), "nl_comment": "", "module": "",
                       "file": str(f.relative_to(sn_dir)),
                       "line": rec.get("line", 0), "body_hash": h}
        except Exception: pass
        # named
        try:
            for rec in named_extract(f):
                sva = rec.get("sva", "")
                if not sva: continue
                h = _hash(sva)
                if h in seen: continue
                seen.add(h)
                yield {"source_snapshot": sn_dir.name, "strategy": "named",
                       "sva": _norm(sva), "nl_comment": rec.get("nl_comment", ""),
                       "module": rec.get("module", ""),
                       "file": str(f.relative_to(sn_dir)),
                       "line": rec.get("line", 0), "body_hash": h}
        except Exception: pass


if __name__ == "__main__":
    main()
