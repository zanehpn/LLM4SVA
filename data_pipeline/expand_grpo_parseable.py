#!/usr/bin/env python3
"""
expand_grpo_parseable.py — retry yosys-slang parseability by bundling the
module with its likely dependencies (packages, interfaces, same-dir siblings).

Bundle strategy per sample:
  1. All `*_pkg.sv` / `*_pkg.svh` in the **same repo**  (cap 25 files, 500KB)
  2. All `*_if.sv` / `*_intf.sv` in the same repo       (cap 10 files)
  3. All .sv/.svh in the SAME DIRECTORY as the module file (same submodules)
  4. Finally the module body itself

Pre-processes:
  - Drop `` `include `` lines (we're concatenating instead)
  - Keep `` `define `` lines as-is (slang needs them before use)

Input:  data/train/grpo/verifiable_with_flag.jsonl
Output:
  data/train/grpo/verifiable_expanded.jsonl   — all samples w/ parseable flag
  data/train/grpo/grpo_pool.jsonl             — parseable subset (pool for GRPO)
"""
import argparse
import json
import multiprocessing as mp
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent
REPOS_DIR = EXPERIMENTS_DIR / "data" / "raw" / "github_repos"

SNAPSHOT_ROOTS = [
    Path("${SVA_CORPUS_ROOT}/HWE-train/snapshots"),
    Path("${SVA_CORPUS_ROOT}/HDL_all/snapshots"),
    Path("${SVA_CORPUS_ROOT}/HDL_verified_new/testbench"),
    Path("${SVA_CORPUS_ROOT}/HDL_verified_new_useless/snapshots_top100"),
    Path("${SVA_CORPUS_ROOT}/Local_agent/tmp/iter_memory1"),
    Path("${SVA_CORPUS_ROOT}/Other_agent/KGCompass/workdir_hdl"),
]

IN_JSONL = EXPERIMENTS_DIR / "data" / "train" / "grpo" / "verifiable_with_flag.jsonl"
OUT_EXPANDED = EXPERIMENTS_DIR / "data" / "train" / "grpo" / "verifiable_expanded.jsonl"
OUT_POOL = EXPERIMENTS_DIR / "data" / "train" / "grpo" / "grpo_pool.jsonl"

PKG_CAP_FILES = 25
PKG_CAP_BYTES = 500_000
IF_CAP_FILES = 10
TIMEOUT = 20


def find_repo_root(file_path: Path) -> Optional[Path]:
    """Given an absolute SV file path, return its cloned-repo root."""
    p = file_path
    for _ in range(15):
        if p.parent == p:
            break
        # Our clones are under data/raw/github_repos/<owner>__<repo>/
        if p.parent == REPOS_DIR:
            return p
        # Snapshots: any of the 6 SNAPSHOT_ROOTS
        if p.parent in SNAPSHOT_ROOTS:
            return p
        # AssertionBench: verified_assertions/<family>/<sub_design>
        if "assertion_data_for_LLM" in str(p) and p.name == "verified_assertions":
            return p
        p = p.parent
    return None


def collect_bundle(file_path: Path) -> str:
    """Build a concatenated dependency preamble for a single SV file."""
    repo = find_repo_root(file_path)
    chunks = []

    # 1. Packages in the whole repo
    if repo is not None:
        pkg_files = sorted(repo.rglob("*_pkg.sv")) + sorted(repo.rglob("*_pkg.svh"))
        total_bytes = 0
        for p in pkg_files[:PKG_CAP_FILES]:
            try:
                txt = p.read_text(errors="ignore")
            except Exception:
                continue
            if total_bytes + len(txt) > PKG_CAP_BYTES:
                break
            chunks.append(f"// === PKG: {p.name} ===")
            chunks.append(_strip_includes(txt))
            total_bytes += len(txt)

        # 2. Interfaces
        if_files = (sorted(repo.rglob("*_if.sv"))
                    + sorted(repo.rglob("*_intf.sv")))[:IF_CAP_FILES]
        for p in if_files:
            try:
                chunks.append(f"// === IF: {p.name} ===")
                chunks.append(_strip_includes(p.read_text(errors="ignore")))
            except Exception:
                pass

    # 3. Same-dir siblings
    for p in sorted(file_path.parent.glob("*.sv")):
        if p == file_path:
            continue
        try:
            chunks.append(f"// === SIB: {p.name} ===")
            chunks.append(_strip_includes(p.read_text(errors="ignore")))
        except Exception:
            pass
    for p in sorted(file_path.parent.glob("*.svh")):
        try:
            chunks.append(f"// === SIB: {p.name} ===")
            chunks.append(_strip_includes(p.read_text(errors="ignore")))
        except Exception:
            pass

    return "\n".join(chunks) + "\n"


def _strip_includes(text: str) -> str:
    return re.sub(r"(?m)^\s*`include\s+.*$", "", text)


def _parseable(source: str, timeout: int = TIMEOUT) -> bool:
    with tempfile.TemporaryDirectory(prefix="sva_expand_") as d:
        p = Path(d) / "m.sv"
        p.write_text(source)
        try:
            rc = subprocess.call(
                ["yosys", "-q", "-m", "slang", "-p",
                 f"read_slang --ignore-assertions --allow-use-before-declare {p}; hierarchy"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=timeout,
            )
            return rc == 0
        except Exception:
            return False


def process_one(args: Tuple[int, dict]) -> Tuple[int, bool, int]:
    idx, rec = args
    if rec.get("parseable"):
        return idx, True, len(rec["rtl_module"])
    file_path = Path(rec.get("file", ""))
    if not file_path.exists():
        return idx, False, 0
    bundle = collect_bundle(file_path)
    combined = bundle + "\n" + rec["rtl_module"]
    ok = _parseable(combined)
    return idx, ok, len(combined)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=mp.cpu_count() // 2)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    recs = []
    with open(IN_JSONL) as f:
        for i, line in enumerate(f):
            recs.append(json.loads(line))
            if args.limit and i + 1 >= args.limit:
                break
    print(f"[expand] {len(recs)} records, workers={args.workers}")

    tasks = [(i, r) for i, r in enumerate(recs)]
    results: Dict[int, bool] = {}
    with mp.Pool(args.workers) as pool:
        for n, (idx, ok, nbytes) in enumerate(
                pool.imap_unordered(process_one, tasks, chunksize=2), 1):
            results[idx] = ok
            if n % 200 == 0:
                passed = sum(1 for v in results.values() if v)
                print(f"  progress {n}/{len(tasks)}  passed={passed} "
                      f"({100*passed/n:.1f}%)")

    # Annotate + write
    kept = 0
    per_src_ok = {}
    per_src_total = {}
    with open(OUT_EXPANDED, "w") as fe, open(OUT_POOL, "w") as fp:
        for i, r in enumerate(recs):
            ok = results.get(i, False)
            r["parseable"] = bool(ok)
            per_src_total[r["source"]] = per_src_total.get(r["source"], 0) + 1
            fe.write(json.dumps(r) + "\n")
            if ok:
                fp.write(json.dumps(r) + "\n")
                per_src_ok[r["source"]] = per_src_ok.get(r["source"], 0) + 1
                kept += 1

    print()
    print("=" * 60)
    print(f"Expanded GRPO pool: {kept} / {len(recs)}  "
          f"({100*kept/len(recs):.1f}%)")
    print(f"Per-source (parseable after bundling):")
    for src in sorted(per_src_total.keys()):
        ok_n = per_src_ok.get(src, 0)
        tot_n = per_src_total[src]
        print(f"  {src:<22} {ok_n:>5} / {tot_n:<5}  "
              f"({100*ok_n/max(tot_n,1):.1f}%)")
    print()
    print(f"Outputs:")
    print(f"  {OUT_POOL}")
    print(f"  {OUT_EXPANDED}")


if __name__ == "__main__":
    main()
