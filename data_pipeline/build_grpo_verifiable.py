#!/usr/bin/env python3
"""
build_grpo_verifiable.py — produce a GRPO-ready subset of training data where
each record carries the **complete enclosing RTL module** that the SVA lives
in, not just the SVA body. These triples are what the GRPO verifier needs.

Sources (all have `file` + `source_repo` / `source_snapshot`):
  - data/raw/github_scraped/scraped.jsonl       (wave-1 github repos)
  - data/raw/opentitan_macros/expanded.jsonl    (macro-expanded SVAs)
  - data/raw/named_properties/scraped.jsonl     (property...endproperty blocks)
  - data/raw/snapshots_scraped/scraped.jsonl    (PR snapshots pool)

Explicitly EXCLUDED:
  - data/raw/assertion_data_for_LLM             (AssertionBench) — THIS IS THE
    TEST SET. Pulling it into training would leak labels. Removed 2026-04-20.
  - data/raw/github_search_scraped/scraped.jsonl — clones deleted after extraction

Output: data/train/grpo/verifiable.jsonl
Schema:
  {
    "id":           "<source>_<file>_L<line>_<module>_<idx>",
    "source":       "github_scraped" | ...,
    "source_repo":  "lowRISC/opentitan",
    "license":      "Apache-2.0",
    "nl":           "... (may be empty)",
    "sva":          "assert property (@(posedge clk) req |=> gnt);",
    "module_name":  "bar",
    "rtl_module":   "<full module bar (...) ... endmodule>",
    "expected_tcl": 4,
    "file":         "rtl/bar.sv",
    "line":         123,
    "parseable":    true/false   (yosys -m slang --ignore-assertions check)
  }

Parseable check (optional, with --verify): run yosys with slang plugin on the
extracted module; records where parsing fails are dropped.
"""
import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = EXPERIMENTS_DIR / "data" / "raw"
REPOS_DIR = RAW_DIR / "github_repos"
OUT_DIR = EXPERIMENTS_DIR / "data" / "train" / "grpo"
OUT_JSONL = OUT_DIR / "verifiable.jsonl"

# Snapshot pools — used to resolve `source_snapshot` back to a file
SNAPSHOT_ROOTS = [
    Path("${SVA_CORPUS_ROOT}/HWE-train/snapshots"),
    Path("${SVA_CORPUS_ROOT}/HDL_all/snapshots"),
    Path("${SVA_CORPUS_ROOT}/HDL_verified_new/testbench"),
    Path("${SVA_CORPUS_ROOT}/HDL_verified_new_useless/snapshots_top100"),
    Path("${SVA_CORPUS_ROOT}/Local_agent/tmp/iter_memory1"),
    Path("${SVA_CORPUS_ROOT}/Other_agent/KGCompass/workdir_hdl"),
]

# --- module extraction ---------------------------------------------------

def find_modules(text: str) -> List[Tuple[int, int, str]]:
    """Return list of (start_line, end_line_inclusive, module_name) for every
    top-level `module <name> ... endmodule` in the file. Handles nested
    begin/end but not nested module decls (SV permits them but it's rare)."""
    modules = []
    lines = text.splitlines()
    mod_re = re.compile(r"^\s*module\s+([A-Za-z_]\w*)")
    end_re = re.compile(r"^\s*endmodule\b")
    start = None; name = None
    for i, line in enumerate(lines):
        if start is None:
            m = mod_re.match(line)
            if m:
                # Skip interface / class keyword matches by re-checking
                if re.match(r"^\s*(interface|class|package)\b", line):
                    continue
                start = i
                name = m.group(1)
        else:
            if end_re.match(line):
                modules.append((start, i, name))
                start = None; name = None
    return modules


def enclosing_module(text: str, sva_line: int) -> Optional[Tuple[str, str]]:
    """Return (module_name, module_body_text) containing the given 1-based
    line number; None if not inside any module."""
    lines = text.splitlines()
    for (s, e, name) in find_modules(text):
        if s <= (sva_line - 1) <= e:
            body = "\n".join(lines[s:e + 1])
            return name, body
    return None


# --- source-specific resolvers -----------------------------------------

def resolve_github_like(source_repo: str, file_rel: str) -> Optional[Path]:
    """github_scraped / opentitan_macros / named_properties style."""
    repo_dir = REPOS_DIR / source_repo.replace("/", "__")
    if not repo_dir.exists():
        return None
    p = repo_dir / file_rel
    return p if p.exists() else None


def resolve_snapshot(source_snapshot: str, file_rel: str) -> Optional[Path]:
    for root in SNAPSHOT_ROOTS:
        base = root / source_snapshot
        if base.exists():
            p = base / file_rel
            if p.exists():
                return p
    return None


# --- iterate raw per-source jsonl + produce triples -------------------

def iter_source(
    jsonl_path: Path,
    get_file: callable,
    get_repo: callable,
    source_tag: str,
):
    if not jsonl_path.exists():
        return
    # Cache module-extractions per file
    mods_by_file: Dict[Path, List[Tuple[int, int, str]]] = {}
    file_text: Dict[Path, str] = {}
    with open(jsonl_path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            sva = rec.get("sva", "").strip()
            sva_line = rec.get("line", 0)
            if not sva or not sva_line:
                continue
            fpath = get_file(rec)
            if fpath is None or not fpath.exists():
                continue
            if fpath not in file_text:
                try:
                    file_text[fpath] = fpath.read_text(errors="ignore")
                    mods_by_file[fpath] = find_modules(file_text[fpath])
                except Exception:
                    file_text[fpath] = ""
                    mods_by_file[fpath] = []
            text = file_text[fpath]
            if not text:
                continue
            # find enclosing module
            mod = None
            for (s, e, name) in mods_by_file[fpath]:
                if s <= (sva_line - 1) <= e:
                    lines = text.splitlines()
                    mod = (name, "\n".join(lines[s:e + 1]))
                    break
            if mod is None:
                continue
            mod_name, mod_body = mod
            yield {
                "source": source_tag,
                "source_repo": get_repo(rec),
                "nl_comment": rec.get("nl_comment", "") or "",
                "sva": sva,
                "module_name": mod_name,
                "rtl_module": mod_body,
                "file": str(fpath),
                "line": sva_line,
                "strategy": rec.get("strategy", "") or rec.get("macro", ""),
            }


# --- AssertionBench paired loader --------------------------------------

def iter_assertionbench():
    """For every .gold file under AssertionBench, pair parsed assertions with
    the sibling .v file(s)."""
    ab_root = RAW_DIR / "assertion_data_for_LLM" / "verified_assertions"
    if not ab_root.exists():
        return
    # Import parser from sibling data_pipeline module.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from fetch_benchmarks import _parse_gold_file  # type: ignore

    # Prefer *_filtered.gold
    gold_files = sorted(ab_root.rglob("*_filtered.gold"))
    if not gold_files:
        gold_files = sorted(ab_root.rglob("*.gold"))
    for gold in gold_files:
        sub_dir = gold.parent
        v_files = list(sub_dir.glob("*.v"))
        if not v_files:
            continue
        # Concatenate up to 3 neighbouring .v files as module context
        rtl_chunks = []
        for v in v_files[:3]:
            try:
                rtl_chunks.append(f"// === {v.name} ===\n"
                                  + v.read_text(errors="ignore"))
            except Exception:
                pass
        rtl = "\n".join(rtl_chunks)
        if not rtl:
            continue
        for k, (kind, body) in enumerate(_parse_gold_file(gold)):
            if kind == "property":
                sva = f"assert property ({body});"
            else:
                sva = f"assert property (@(posedge clk) {body});"
            yield {
                "source": "assertionbench",
                "source_repo": f"opencores/{sub_dir.parent.name}",
                "nl_comment": "",
                "sva": sva,
                "module_name": sub_dir.name,
                "rtl_module": rtl,
                "file": str(gold.relative_to(RAW_DIR)),
                "line": 0,
                "strategy": kind,
            }


# --- yosys-slang parseability sanity check -----------------------------

def yosys_slang_parseable(rtl: str, timeout: int = 15) -> bool:
    """Return True iff yosys -m slang can read the module (SVAs ignored)."""
    with tempfile.TemporaryDirectory(prefix="sva_verify_") as tmp:
        t = Path(tmp)
        src = t / "m.sv"
        src.write_text(rtl)
        try:
            rc = subprocess.call(
                ["yosys", "-q", "-m", "slang", "-p",
                 f"read_slang --ignore-assertions {src}; hierarchy"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=timeout,
            )
            return rc == 0
        except subprocess.TimeoutExpired:
            return False
        except FileNotFoundError:
            return False


# --- orchestrator ------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="run yosys-slang parse check on each extracted module")
    ap.add_argument("--limit", type=int, default=0,
                    help="debug: stop after N emitted records")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # TCL classifier for relabelling
    sys.path.insert(0, str(EXPERIMENTS_DIR))
    from src.tcl_classifier import TCLClassifier
    clf = TCLClassifier()

    seen_body = set()
    written = 0
    parseable_ok = parseable_fail = 0
    per_source = {}

    def record_to_output(rec: dict) -> Optional[dict]:
        nonlocal parseable_ok, parseable_fail
        body = re.sub(r"\s+", " ", rec["sva"]).strip()
        h = hashlib.sha256(body.encode()).hexdigest()[:16]
        if h in seen_body:
            return None
        seen_body.add(h)
        level, _ = clf.classify(rec["sva"])
        out = {
            "id": f"{rec['source']}_{Path(rec.get('file','?')).name}_"
                  f"L{rec.get('line','0')}_{rec.get('module_name','?')}_{h}",
            "source": rec["source"],
            "source_repo": rec.get("source_repo", ""),
            "nl": rec.get("nl_comment", ""),
            "sva": body,
            "module_name": rec.get("module_name", ""),
            "rtl_module": rec["rtl_module"],
            "expected_tcl": int(level),
            "file": rec.get("file", ""),
            "line": rec.get("line", 0),
            "strategy": rec.get("strategy", ""),
        }
        if args.verify:
            ok = yosys_slang_parseable(rec["rtl_module"])
            out["parseable"] = ok
            if ok:
                parseable_ok += 1
            else:
                parseable_fail += 1
                return None      # drop non-parseable samples
        return out

    with open(OUT_JSONL, "w") as out:
        feeders = [
            ("github_scraped",
             iter_source(RAW_DIR / "github_scraped" / "scraped.jsonl",
                         get_file=lambda r: resolve_github_like(
                             r.get("source_repo", ""), r.get("file", "")),
                         get_repo=lambda r: r.get("source_repo", ""),
                         source_tag="github_scraped")),
            ("opentitan_macros",
             iter_source(RAW_DIR / "opentitan_macros" / "expanded.jsonl",
                         get_file=lambda r: resolve_github_like(
                             r.get("source_repo", ""), r.get("file", "")),
                         get_repo=lambda r: r.get("source_repo", ""),
                         source_tag="opentitan_macros")),
            ("named_properties",
             iter_source(RAW_DIR / "named_properties" / "scraped.jsonl",
                         get_file=lambda r: resolve_github_like(
                             r.get("source_repo", ""), r.get("file", "")),
                         get_repo=lambda r: r.get("source_repo", ""),
                         source_tag="named_properties")),
            ("snapshots_scraped",
             iter_source(RAW_DIR / "snapshots_scraped" / "scraped.jsonl",
                         get_file=lambda r: resolve_snapshot(
                             r.get("source_snapshot", ""), r.get("file", "")),
                         get_repo=lambda r: r.get("source_snapshot", ""),
                         source_tag="snapshots_scraped")),
            # assertionbench DELIBERATELY OMITTED — it is the test set.
        ]
        for src_tag, gen in feeders:
            cnt = 0
            for rec in gen:
                row = record_to_output(rec)
                if row is None:
                    continue
                out.write(json.dumps(row) + "\n")
                written += 1
                cnt += 1
                if args.limit and written >= args.limit:
                    break
            per_source[src_tag] = cnt
            print(f"  [{src_tag}] +{cnt}")
            if args.limit and written >= args.limit:
                break

    print(f"\n[grpo] total verifiable records: {written}")
    if args.verify:
        print(f"[grpo] yosys-slang parseable: {parseable_ok} / "
              f"{parseable_ok + parseable_fail}")
    print(f"[grpo] output: {OUT_JSONL}")
    # TCL distribution
    from collections import Counter
    c = Counter()
    for line in open(OUT_JSONL):
        c[json.loads(line)["expected_tcl"]] += 1
    print(f"[grpo] TCL dist: L1={c[1]} L2={c[2]} L3={c[3]} L4={c[4]} L5={c[5]}")


if __name__ == "__main__":
    main()
