#!/usr/bin/env python3
"""
scrape_github_sva.py — harvest SystemVerilog assertions from star-popular
open-source RTL repositories on GitHub.

Policy:
  - Only repos with stars > MIN_STARS (default 100) are cloned.
  - Shallow clones (--depth 1).
  - Scraped SVAs are written to data/raw/github_scraped/scraped.jsonl with
    full provenance (repo, license, file, line, surrounding comment).
  - `fetch_benchmarks.py` has a matching `github_scraped` loader that routes
    them into data/train/ with the contamination filter applied.

Usage:
  python3 scripts/scrape_github_sva.py --check      # just show which seeds
                                                    # pass the stars bar
  python3 scripts/scrape_github_sva.py --clone      # shallow-clone passers
  python3 scripts/scrape_github_sva.py --extract    # scan cloned repos
  python3 scripts/scrape_github_sva.py --all        # all three steps
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = EXPERIMENTS_DIR / "data" / "raw" / "github_repos"
OUT_DIR = EXPERIMENTS_DIR / "data" / "raw" / "github_scraped"
OUT_JSONL = OUT_DIR / "scraped.jsonl"
PASSERS_JSON = OUT_DIR / "_passers.json"

MIN_STARS = 100

# -----------------------------------------------------------------------------
# Seed list — curated repos known to carry SVAs. All expected to be >100★.
# -----------------------------------------------------------------------------
SEEDS = [
    # (owner/repo, short note)
    # Wave 1 (already scraped)
    ("lowRISC/opentitan",        "Google/lowRISC root-of-trust chip — formal/ dirs"),
    ("lowRISC/ibex",             "RISC-V core — fv/ and dv/ with SVAs"),
    ("openhwgroup/cva6",         "Ariane RISC-V core"),
    ("openhwgroup/cv32e40p",     "RISC-V core, OpenHW"),
    ("openhwgroup/cv32e40x",     "RISC-V core, OpenHW"),
    ("openhwgroup/cvw",          "Wally RISC-V core (Harvey Mudd)"),
    ("pulp-platform/snitch_cluster", "PULP Snitch"),
    ("pulp-platform/axi",        "PULP AXI library"),
    ("pulp-platform/common_cells", "PULP primitives"),
    ("chipsalliance/Surelog",    "SV parser with SVA self-tests"),
    ("chipsalliance/Caliptra-RTL", "Caliptra security RTL"),
    ("steveicarus/ivtest",       "Icarus Verilog test suite"),
    ("verilator/verilator",      "Verilator — large SV test corpus"),
    ("YosysHQ/sby",              "SymbiYosys — examples with SVAs"),
    ("SpinalHDL/VexRiscv",       "Generated SV from SpinalHDL; some SVAs"),
    # Wave 2 (new)
    ("black-parrot/black-parrot", "Linux-capable RISC-V multicore (BP)"),
    ("riscv-boom/riscv-boom",    "Berkeley Berkeley Out-of-Order core (BOOM)"),
    ("pulp-platform/cheshire",   "PULP Linux-capable SoC"),
    ("pulp-platform/cvfpu",      "FPNew FPU core"),
    ("pulp-platform/idma",       "PULP iDMA"),
    ("pulp-platform/axi_riscv_atomics", "AXI RISC-V atomics"),
    ("chipsalliance/t1",         "RISC-V V vector unit"),
    ("chipsalliance/Caliptra-SS", "Caliptra subsystem (includes more RTL)"),
    ("zachjs/sv2v",              "SystemVerilog → Verilog preprocessor (tests)"),
    ("bespoke-silicon-group/basejump_stl", "BaseJump STL (BYU/UW primitives)"),
]


# -----------------------------------------------------------------------------
# GitHub API stars check (unauthenticated; 60 req/hr is plenty for 16 repos)
# -----------------------------------------------------------------------------
def get_stars(owner_repo: str) -> Optional[int]:
    url = f"https://api.github.com/repos/{owner_repo}"
    req = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github+json",
        "User-Agent": "SVA4DAC-scraper",
    })
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read())
        return int(data.get("stargazers_count", 0))
    except Exception as e:
        print(f"  [stars] {owner_repo}: ERROR {e}")
        return None


def check_all(min_stars: int = MIN_STARS) -> List[Dict]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = []
    for owner_repo, note in SEEDS:
        stars = get_stars(owner_repo)
        status = "SKIP" if stars is None or stars < min_stars else "OK"
        star_str = f"{stars}★" if stars is not None else "?"
        print(f"  [{status}] {owner_repo:<40} {star_str:>8}  — {note}")
        result.append({
            "repo": owner_repo, "stars": stars, "note": note,
            "passes": status == "OK",
        })
        time.sleep(0.5)  # be polite to unauthenticated API
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(PASSERS_JSON, "w") as f:
        json.dump(result, f, indent=2)
    return result


# -----------------------------------------------------------------------------
# Shallow clone
# -----------------------------------------------------------------------------
def clone_passers():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not PASSERS_JSON.exists():
        print("No passers file. Run --check first.")
        return
    passers = [p for p in json.load(open(PASSERS_JSON)) if p["passes"]]
    for p in passers:
        owner, repo = p["repo"].split("/")
        target = RAW_DIR / f"{owner}__{repo}"
        if target.exists():
            print(f"  [clone] {p['repo']}: already at {target}")
            continue
        url = f"https://github.com/{p['repo']}.git"
        print(f"  [clone] {p['repo']} → {target}")
        rc = subprocess.call(["git", "clone", "--depth", "1", url, str(target)])
        if rc != 0:
            print(f"  [clone] FAILED {p['repo']}")


# -----------------------------------------------------------------------------
# SVA extraction
# -----------------------------------------------------------------------------
SV_EXTS = {".sv", ".svh", ".v", ".vh"}
KEYWORDS = ("assert property", "assume property", "cover property")

# Module header regex (first ~80 lines of each file only needed)
MODULE_RE = re.compile(r"\bmodule\s+([A-Za-z_]\w*)\b", re.IGNORECASE)


def _balanced_capture(text: str, open_idx: int) -> Optional[int]:
    """Given text and the index of an opening '(', return the index of the
    matching ')' or None if unbalanced."""
    depth = 0
    i = open_idx
    n = len(text)
    in_blk_cmt = False
    in_line_cmt = False
    in_str = False
    while i < n:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if in_blk_cmt:
            if ch == "*" and nxt == "/":
                in_blk_cmt = False
                i += 2
                continue
        elif in_line_cmt:
            if ch == "\n":
                in_line_cmt = False
        elif in_str:
            if ch == "\\" and i + 1 < n:
                i += 2
                continue
            if ch == '"':
                in_str = False
        else:
            if ch == "/" and nxt == "*":
                in_blk_cmt = True
                i += 2
                continue
            if ch == "/" and nxt == "/":
                in_line_cmt = True
                i += 2
                continue
            if ch == '"':
                in_str = True
            elif ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    return i
        i += 1
    return None


def _grab_comment_above(text: str, assert_line_start: int) -> str:
    """Walk upward collecting // or /* */ comment lines immediately above."""
    lines = text[:assert_line_start].splitlines()
    out = []
    for line in reversed(lines):
        s = line.strip()
        if s.startswith("//"):
            out.append(s.lstrip("/").strip())
        elif s.endswith("*/"):
            out.append(s.rstrip("*/").lstrip("/*").strip())
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


def _enclosing_module(text: str, pos: int) -> str:
    """Return name of nearest preceding `module X` declaration, or ''."""
    last = ""
    for m in MODULE_RE.finditer(text, 0, pos):
        last = m.group(1)
    return last


def extract_from_file(path: Path) -> List[Dict]:
    try:
        text = path.read_text(errors="ignore")
    except Exception:
        return []
    if not any(kw in text for kw in KEYWORDS):
        return []

    out = []
    lowered = text  # we search case-sensitive; SV asserts are lowercase anyway
    for kw in KEYWORDS:
        start = 0
        while True:
            idx = lowered.find(kw, start)
            if idx < 0:
                break
            # find first '(' after keyword
            paren_open = lowered.find("(", idx + len(kw))
            if paren_open < 0:
                start = idx + len(kw)
                continue
            paren_close = _balanced_capture(lowered, paren_open)
            start = idx + len(kw)
            if paren_close is None:
                continue
            # Expression including outer parens:
            expr = text[paren_open:paren_close + 1]
            # Full statement body (keyword + expr + ';' if present)
            tail = paren_close + 1
            # advance past optional 'else' clause? skip — keep the minimal form.
            while tail < len(text) and text[tail] in " \t":
                tail += 1
            semi = text.find(";", tail)
            if 0 < semi - tail < 200:
                stmt = text[idx:semi + 1]
            else:
                stmt = text[idx:paren_close + 1] + ";"

            # Skip trivial / non-concurrent-style lines:
            # require presence of an edge or a temporal op
            if not re.search(
                r"@\s*\(\s*(?:posedge|negedge)|"
                r"\|->|\|=>|##|\bs_eventually\b|\bs_until\b|\bthroughout\b|\beventually\b|\buntil\b",
                stmt,
            ):
                # Keep it still (TCL-1 combinational implications are valid),
                # but skip if looks like an immediate-style `assert property(x)`
                # with a bare variable — that's a reference, not a new assertion.
                if re.fullmatch(r"\s*assert\s+property\s*\(\s*[A-Za-z_]\w*\s*\)\s*;\s*",
                                stmt):
                    continue

            # Line number (1-based)
            line_no = text.count("\n", 0, idx) + 1
            nl = _grab_comment_above(text, idx - text[:idx].rfind("\n") - 1 + (text[:idx].rfind("\n") + 1))
            mod = _enclosing_module(text, idx)
            out.append({
                "sva": re.sub(r"\s+", " ", stmt).strip(),
                "nl_comment": nl,
                "module": mod,
                "file": str(path),
                "line": line_no,
            })
    return out


def scan_repo(repo_dir: Path, license_txt: str) -> List[Dict]:
    """Walk a cloned repo, extract all SVAs."""
    results = []
    owner_repo = repo_dir.name.replace("__", "/")
    for f in repo_dir.rglob("*"):
        if not f.is_file():
            continue
        if f.suffix.lower() not in SV_EXTS:
            continue
        # Skip common vendored / large generated dirs to be quick
        if any(part in ("build", "_out", "node_modules", ".git") for part in f.parts):
            continue
        for rec in extract_from_file(f):
            rec["source_repo"] = owner_repo
            rec["license"] = license_txt
            rec["file"] = str(f.relative_to(repo_dir))
            results.append(rec)
    return results


def detect_license(repo_dir: Path) -> str:
    for name in ("LICENSE", "LICENSE.txt", "LICENSE.md", "COPYING"):
        p = repo_dir / name
        if p.exists():
            head = p.read_text(errors="ignore")[:400].lower()
            for spdx in ("apache license, version 2.0", "bsd", "mit license",
                         "gnu general public", "solderpad", "mozilla public"):
                if spdx in head:
                    if "apache" in spdx:
                        return "Apache-2.0"
                    if "bsd" in spdx:
                        return "BSD"
                    if "mit" in spdx:
                        return "MIT"
                    if "solderpad" in spdx:
                        return "Solderpad-2.0"
                    if "gnu" in spdx:
                        return "GPL"
                    if "mozilla" in spdx:
                        return "MPL-2.0"
            return "UNKNOWN"
    return "UNKNOWN"


def extract_all():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not RAW_DIR.exists():
        print("No cloned repos at", RAW_DIR, "— run --clone first.")
        return
    seen_hashes = set()
    kept = 0
    skipped_dupes = 0
    by_repo = {}
    with open(OUT_JSONL, "w") as out:
        for repo_dir in sorted(RAW_DIR.iterdir()):
            if not repo_dir.is_dir():
                continue
            lic = detect_license(repo_dir)
            print(f"  [scan] {repo_dir.name}  license={lic}")
            recs = scan_repo(repo_dir, lic)
            repo_kept = 0
            for r in recs:
                body_hash = hashlib.sha256(
                    re.sub(r"\s+", " ", r["sva"]).encode()
                ).hexdigest()[:16]
                if body_hash in seen_hashes:
                    skipped_dupes += 1
                    continue
                seen_hashes.add(body_hash)
                r["body_hash"] = body_hash
                out.write(json.dumps(r) + "\n")
                kept += 1
                repo_kept += 1
            by_repo[repo_dir.name] = repo_kept
            print(f"         → {repo_kept} unique SVAs")
    print(f"\n[scrape] total unique SVAs written: {kept}")
    print(f"[scrape] duplicates skipped:         {skipped_dupes}")
    print(f"[scrape] output: {OUT_JSONL}")
    print(f"\nPer-repo counts:")
    for r, n in sorted(by_repo.items(), key=lambda kv: -kv[1]):
        print(f"  {r:<45} {n}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="check stars; write passers list")
    ap.add_argument("--clone", action="store_true", help="shallow-clone passers")
    ap.add_argument("--extract", action="store_true", help="scan + write jsonl")
    ap.add_argument("--all", action="store_true", help="check + clone + extract")
    ap.add_argument("--min-stars", type=int, default=MIN_STARS)
    args = ap.parse_args()

    if args.all or args.check:
        print(f"=== stars check (min {args.min_stars}★) ===")
        check_all(args.min_stars)
    if args.all or args.clone:
        print(f"\n=== clone passers ===")
        clone_passers()
    if args.all or args.extract:
        print(f"\n=== extract SVAs ===")
        extract_all()
    if not (args.all or args.check or args.clone or args.extract):
        print(__doc__)


if __name__ == "__main__":
    main()
