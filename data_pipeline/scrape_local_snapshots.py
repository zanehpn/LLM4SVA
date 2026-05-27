#!/usr/bin/env python3
"""
scrape_local_snapshots.py — Wave 6 SVA extraction.

Walks ${SVA_CORPUS_ROOT}/HWE-train/snapshots:
  - 671 `_prN` directories (extracted PR snapshots)
  - 169 `.tar.gz` archives (full-repo snapshots at specific commits)

For each, runs the three extractors used by scrape_clone_delete.py
(inline, macro, named) on every .sv/.svh/.v/.vh file. Deduplicates
against existing scraped SVA hashes (across all train sources).

Output: appended to data/raw/github_search_scraped/scraped.jsonl
(same destination as Wave 3/4/5 scrapes — keeps a single canonical
raw scrape file).

Usage:
  PYTHONPATH=. python3 scripts/scrape_local_snapshots.py \\
      [--snapshots /path/to/snapshots] [--limit-archives N]
"""
import argparse
import hashlib
import json
import re
import sys
import tarfile
import tempfile
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scrape_github_sva import extract_from_file as inline_extract
from expand_opentitan_macros import scan_file as macro_scan
from scrape_named_properties import extract_from_file as named_extract

UNIFIED = EXPERIMENTS_DIR / "data" / "train" / "unified" / "train_unified.jsonl"
OUT_DIR = EXPERIMENTS_DIR / "data" / "raw" / "github_search_scraped"
OUT_JSONL = OUT_DIR / "scraped.jsonl"

SV_EXTS = {".sv", ".svh", ".v", ".vh"}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _hash(s: str) -> str:
    return hashlib.sha256(_norm(s).encode()).hexdigest()[:16]


def load_existing_hashes() -> set:
    seen = set()
    if UNIFIED.exists():
        for line in open(UNIFIED):
            r = json.loads(line)
            seen.add(_hash(r["reference_sva"]))
    if OUT_JSONL.exists():
        for line in open(OUT_JSONL):
            try:
                r = json.loads(line)
                seen.add(r.get("body_hash", _hash(r.get("sva", ""))))
            except Exception:
                pass
    print(f"[seed] {len(seen)} existing SVA hashes loaded")
    return seen


def walk_dir(root: Path, seen: set, source_label: str):
    """Run all 3 extractors on every SV/V file under root.
    Returns list of new unique SVA records."""
    records = []
    for f in root.rglob("*"):
        if not f.is_file() or f.suffix.lower() not in SV_EXTS:
            continue
        if any(p in (".git", "build", "_out", "node_modules", "__pycache__")
               for p in f.parts):
            continue
        try:
            for extractor, strategy in [
                (inline_extract, "inline"),
                (macro_scan, "macro"),
                (named_extract, "named"),
            ]:
                try:
                    items = extractor(f)
                except Exception:
                    continue
                for it in items or []:
                    sva = it.get("sva") or it.get("body") or ""
                    if not sva or len(sva) < 30 or len(sva) > 4000:
                        continue
                    h = _hash(sva)
                    if h in seen:
                        continue
                    seen.add(h)
                    records.append({
                        "source_repo": source_label,
                        "strategy": strategy,
                        "sva": sva,
                        "nl_comment": it.get("nl_comment", ""),
                        "module": it.get("module", ""),
                        "file": str(f.relative_to(root)),
                        "line": it.get("line", 0),
                        "body_hash": h,
                        "license": "Apache-2.0",
                        "stars": 0,
                    })
        except Exception as e:
            continue
    return records


def repo_label_from_dir(name: str) -> str:
    """Map 'chipsalliance_adams-bridge_pr175' → 'chipsalliance/adams-bridge'."""
    s = re.sub(r"_pr\d+$", "", name)
    parts = s.split("_", 1)
    if len(parts) == 2:
        return f"{parts[0]}/{parts[1]}"
    return s


def repo_label_from_archive(name: str) -> str:
    """'alexforencich_verilog-pcie_<sha>.tar.gz' → 'alexforencich/verilog-pcie'."""
    s = name.replace(".tar.gz", "")
    s = re.sub(r"_[0-9a-f]{40}$", "", s)
    parts = s.split("_", 1)
    if len(parts) == 2:
        return f"{parts[0]}/{parts[1]}"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshots",
                    default="${SVA_CORPUS_ROOT}/HWE-train/snapshots")
    ap.add_argument("--limit-archives", type=int, default=0,
                    help="cap number of .tar.gz archives processed (0 = all)")
    ap.add_argument("--limit-prdirs", type=int, default=0,
                    help="cap number of PR dirs processed (0 = all)")
    args = ap.parse_args()

    seen = load_existing_hashes()
    snap_dir = Path(args.snapshots)

    pr_dirs = sorted(d for d in snap_dir.iterdir() if d.is_dir())
    archives = sorted(f for f in snap_dir.iterdir()
                      if f.suffix == ".gz" and f.name.endswith(".tar.gz"))
    if args.limit_prdirs:
        pr_dirs = pr_dirs[:args.limit_prdirs]
    if args.limit_archives:
        archives = archives[:args.limit_archives]
    print(f"[scan] {len(pr_dirs)} PR dirs + {len(archives)} archives")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = open(OUT_JSONL, "a")
    total_new = 0
    per_repo_new = {}

    # Process PR dirs first (no extraction needed)
    for i, d in enumerate(pr_dirs, 1):
        label = repo_label_from_dir(d.name)
        recs = walk_dir(d, seen, label)
        for r in recs:
            out.write(json.dumps(r) + "\n")
        total_new += len(recs)
        per_repo_new[label] = per_repo_new.get(label, 0) + len(recs)
        if i % 50 == 0 or len(recs) > 0:
            print(f"  [{i}/{len(pr_dirs)}] {d.name}: +{len(recs)} new "
                  f"(total {total_new})")
    out.flush()

    # Process .tar.gz archives
    for i, arc in enumerate(archives, 1):
        label = repo_label_from_archive(arc.name)
        try:
            with tempfile.TemporaryDirectory(prefix="sva_snap_") as tmp:
                tmp_path = Path(tmp)
                # Extract only SV/V files to save IO/disk
                with tarfile.open(arc, "r:gz") as tf:
                    members = [m for m in tf.getmembers()
                               if Path(m.name).suffix.lower() in SV_EXTS]
                    tf.extractall(tmp_path, members=members)
                recs = walk_dir(tmp_path, seen, label)
                for r in recs:
                    out.write(json.dumps(r) + "\n")
                total_new += len(recs)
                per_repo_new[label] = per_repo_new.get(label, 0) + len(recs)
                if i % 20 == 0 or len(recs) > 0:
                    print(f"  arc[{i}/{len(archives)}] {arc.name}: "
                          f"+{len(recs)} new (total {total_new})")
        except Exception as e:
            print(f"  arc[{i}] {arc.name}: ERR {e}")
            continue
    out.close()

    print(f"\n[done] Wave 6 total new unique SVAs: {total_new}")
    print(f"[done] Top contributing repos:")
    for repo, n in sorted(per_repo_new.items(), key=lambda x: -x[1])[:20]:
        if n == 0:
            continue
        print(f"  {n:5d}  {repo}")


if __name__ == "__main__":
    main()
