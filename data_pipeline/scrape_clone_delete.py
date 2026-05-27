#!/usr/bin/env python3
"""
scrape_clone_delete.py — for each candidate repo in
data/raw/github_search/candidates.json:

  1. Shallow-clone to a temp dir.
  2. Run all three extractors (inline, macro, named).
  3. Write unique SVAs to data/raw/github_search_scraped/scraped.jsonl
     (deduped against the current unified train hash set).
  4. Delete the clone dir immediately.

Keeps disk footprint bounded at one-repo-at-a-time.
"""
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scrape_github_sva import extract_from_file as inline_extract
from expand_opentitan_macros import scan_file as macro_scan
from scrape_named_properties import extract_from_file as named_extract

import argparse

_ARG_PARSER = argparse.ArgumentParser()
_ARG_PARSER.add_argument(
    "--candidates",
    default=str(EXPERIMENTS_DIR / "data" / "raw" / "github_search" / "candidates.json"),
    help="path to candidates JSON produced by github_search_sva.py")
_ARGS, _ = _ARG_PARSER.parse_known_args()
CAND_JSON = Path(_ARGS.candidates)
OUT_DIR = EXPERIMENTS_DIR / "data" / "raw" / "github_search_scraped"
OUT_JSONL = OUT_DIR / "scraped.jsonl"
UNIFIED = EXPERIMENTS_DIR / "data" / "train" / "unified" / "train_unified.jsonl"

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
    print(f"[seed] {len(seen)} existing SVA hashes loaded from unified train")
    return seen


def walk_repo(repo_dir: Path, seen: set, repo_full_name: str):
    records = []
    for f in repo_dir.rglob("*"):
        if not f.is_file() or f.suffix.lower() not in SV_EXTS:
            continue
        if any(p in (".git", "build", "_out", "node_modules") for p in f.parts):
            continue
        rel = str(f.relative_to(repo_dir))
        # (A) inline
        try:
            for rec in inline_extract(f):
                sva = rec.get("sva", "")
                if not sva:
                    continue
                h = _hash(sva)
                if h in seen:
                    continue
                seen.add(h)
                records.append({
                    "source_repo": repo_full_name,
                    "strategy": "inline",
                    "sva": _norm(sva),
                    "nl_comment": rec.get("nl_comment", ""),
                    "module": rec.get("module", ""),
                    "file": rel, "line": rec.get("line", 0),
                    "body_hash": h,
                })
        except Exception:
            pass
        # (B) macro
        try:
            for rec in macro_scan(f):
                sva = rec.get("sva", "")
                if not sva:
                    continue
                h = _hash(sva)
                if h in seen:
                    continue
                seen.add(h)
                records.append({
                    "source_repo": repo_full_name,
                    "strategy": f"macro:{rec.get('macro','?')}",
                    "sva": _norm(sva),
                    "nl_comment": "", "module": "",
                    "file": rel, "line": rec.get("line", 0),
                    "body_hash": h,
                })
        except Exception:
            pass
        # (C) named
        try:
            for rec in named_extract(f):
                sva = rec.get("sva", "")
                if not sva:
                    continue
                h = _hash(sva)
                if h in seen:
                    continue
                seen.add(h)
                records.append({
                    "source_repo": repo_full_name,
                    "strategy": "named",
                    "sva": _norm(sva),
                    "nl_comment": rec.get("nl_comment", ""),
                    "module": rec.get("module", ""),
                    "file": rel, "line": rec.get("line", 0),
                    "body_hash": h,
                })
        except Exception:
            pass
    return records


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Don't clobber earlier run — append on re-run
    mode = "a" if OUT_JSONL.exists() else "w"

    cand = json.load(open(CAND_JSON))
    print(f"[scan] {len(cand)} candidate repos")

    seen = load_existing_hashes()
    # Also include anything already in OUT_JSONL from a previous run
    if OUT_JSONL.exists():
        for line in open(OUT_JSONL):
            try:
                seen.add(json.loads(line)["body_hash"])
            except Exception:
                pass
        print(f"[seed] +{sum(1 for _ in open(OUT_JSONL))} already in "
              f"github_search_scraped")

    total_kept = 0
    per_repo = {}
    with open(OUT_JSONL, mode) as out:
        for i, c in enumerate(cand, 1):
            repo = c["repo"]
            stars = c["stars"]
            lic = c["license"]
            print(f"\n[{i}/{len(cand)}] {repo}  {stars}★  {lic}")
            with tempfile.TemporaryDirectory(prefix="sva_scrape_") as tmp:
                target = Path(tmp) / repo.replace("/", "__")
                rc = subprocess.call(
                    ["git", "clone", "--depth", "1", "--quiet",
                     f"https://github.com/{repo}.git", str(target)],
                    timeout=600,
                )
                if rc != 0 or not target.exists():
                    print(f"  [skip] clone failed")
                    continue
                size_mb = sum(f.stat().st_size
                              for f in target.rglob("*") if f.is_file()) / 1e6
                print(f"  cloned ({size_mb:.0f} MB); scanning...")
                recs = walk_repo(target, seen, repo)
                for r in recs:
                    r["license"] = lic
                    r["stars"] = stars
                    out.write(json.dumps(r) + "\n")
                per_repo[repo] = len(recs)
                total_kept += len(recs)
                print(f"  → {len(recs)} new unique SVAs; deleting clone.")
                # tempfile context auto-deletes

    print(f"\n{'='*60}")
    print(f"DONE")
    print(f"{'='*60}")
    print(f"Total new unique SVAs: {total_kept}")
    print(f"Output: {OUT_JSONL}")
    print(f"\nTop contributors:")
    for r, n in sorted(per_repo.items(), key=lambda x: -x[1])[:20]:
        if n > 0:
            print(f"  {r:<48} {n}")


if __name__ == "__main__":
    main()
