#!/usr/bin/env python3
"""
scrape_named_properties.py — extract SVAs declared as named `property` blocks
(the style used by cva6, cv32e40x verif/, and many UVM testbenches):

    property MY_PROP;
       @(posedge clk) disable iff (!rst_n) <LHS> |-> <RHS>;
    endproperty

Wraps each body into `assert property (<body>);` and writes to
data/raw/named_properties/scraped.jsonl, then dedup-merges into the training
corpus via fetch_benchmarks.py.

Complements scripts/scrape_github_sva.py (inline asserts) and
scripts/expand_opentitan_macros.py (macro-expanded SVAs).
"""
import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import List

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent
REPOS_DIR = EXPERIMENTS_DIR / "data" / "raw" / "github_repos"
OUT_DIR = EXPERIMENTS_DIR / "data" / "raw" / "named_properties"
OUT_JSONL = OUT_DIR / "scraped.jsonl"

SV_EXTS = {".sv", ".svh", ".v", ".vh"}

# property NAME [(args)]; BODY endproperty
_PROP_RE = re.compile(
    r"\bproperty\s+([A-Za-z_]\w*)\s*(?:\([^)]*\))?\s*;(.*?)\bendproperty\b",
    re.DOTALL,
)

MODULE_RE = re.compile(r"\bmodule\s+([A-Za-z_]\w*)\b")


def _enclosing_module(text: str, pos: int) -> str:
    last = ""
    for m in MODULE_RE.finditer(text, 0, pos):
        last = m.group(1)
    return last


def _grab_comment_above(text: str, pos: int) -> str:
    # Find previous newline; walk lines backward
    lines = text[:pos].splitlines()
    out = []
    for line in reversed(lines):
        s = line.strip()
        if s.startswith("//"):
            out.append(s.lstrip("/").strip())
        elif s == "":
            if out:
                break
            else:
                continue
        else:
            break
        if len(out) >= 6:
            break
    return " ".join(reversed(out)).strip()


def extract_from_file(path: Path) -> List[dict]:
    try:
        text = path.read_text(errors="ignore")
    except Exception:
        return []
    if "property" not in text or "endproperty" not in text:
        return []
    out = []
    for m in _PROP_RE.finditer(text):
        name = m.group(1)
        body = m.group(2).strip().rstrip(";").strip()
        # Skip empty / trivial single-identifier bodies
        if not body or re.fullmatch(r"[A-Za-z_]\w*", body):
            continue
        # Skip property defs inside package declarations that are just data types
        if "typedef" in body or "enum" in body and "{" in body:
            continue
        # Collapse whitespace
        body_one = re.sub(r"\s+", " ", body)
        sva = f"assert property ({body_one});"
        line_no = text.count("\n", 0, m.start()) + 1
        out.append({
            "name": name,
            "sva": sva,
            "body": body_one,
            "file": str(path),
            "line": line_no,
            "module": _enclosing_module(text, m.start()),
            "nl_comment": _grab_comment_above(text, m.start()),
        })
    return out


def walk_repo(repo_dir: Path) -> List[dict]:
    out = []
    for f in repo_dir.rglob("*"):
        if not f.is_file() or f.suffix.lower() not in SV_EXTS:
            continue
        if any(part in ("build", "_out", ".git") for part in f.parts):
            continue
        out.extend(extract_from_file(f))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos", nargs="*", default=None,
                    help="repo subdir names; default = all cloned")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.repos:
        repo_dirs = [REPOS_DIR / r for r in args.repos]
    else:
        repo_dirs = sorted([d for d in REPOS_DIR.iterdir() if d.is_dir()])

    seen = set()
    kept = 0
    skipped_dupes = 0
    by_repo = {}
    with open(OUT_JSONL, "w") as out:
        for rdir in repo_dirs:
            if not rdir.exists():
                print(f"  [skip] {rdir.name}: missing")
                continue
            recs = walk_repo(rdir)
            repo_kept = 0
            for r in recs:
                h = hashlib.sha256(r["sva"].encode()).hexdigest()[:16]
                if h in seen:
                    skipped_dupes += 1
                    continue
                seen.add(h)
                r["source_repo"] = rdir.name.replace("__", "/")
                r["body_hash"] = h
                r["file"] = str(Path(r["file"]).relative_to(rdir))
                out.write(json.dumps(r) + "\n")
                kept += 1
                repo_kept += 1
            by_repo[rdir.name] = repo_kept
            print(f"  [scan] {rdir.name:<42} {len(recs):>5} found, "
                  f"{repo_kept:>5} unique")

    print(f"\n[named-props] total unique SVAs written: {kept}")
    print(f"[named-props] duplicates skipped:         {skipped_dupes}")
    print(f"[named-props] output: {OUT_JSONL}")
    print(f"\nPer-repo:")
    for r, n in sorted(by_repo.items(), key=lambda x: -x[1]):
        if n > 0:
            print(f"  {r:<45} {n}")


if __name__ == "__main__":
    main()
