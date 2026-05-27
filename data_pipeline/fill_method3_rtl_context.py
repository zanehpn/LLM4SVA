#!/usr/bin/env python3
"""
fill_method3_rtl_context.py — backfill missing rtl_context for method3 synthetic
samples by tracing parent_id to local source files under data/raw/.

Outputs:
  experiments/data/expand_tcl/method3_synthetic_l3_l5_rtl_filled.jsonl
  experiments/data/expand_tcl/method3_synthetic_l3_l5_rtl_fill_manifest.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
GITHUB_REPOS = RAW_DIR / "github_repos"
GITHUB_SEARCH_SCRAPED = RAW_DIR / "github_search_scraped" / "scraped.jsonl"
DEFAULT_IN = ROOT / "data" / "expand_tcl" / "method3_synthetic_l3_l5.jsonl"
DEFAULT_OUT = ROOT / "data" / "expand_tcl" / "method3_synthetic_l3_l5_rtl_filled.jsonl"
DEFAULT_MANIFEST = ROOT / "data" / "expand_tcl" / "method3_synthetic_l3_l5_rtl_fill_manifest.json"

MODULE_RE = re.compile(r"\bmodule\s+([A-Za-z_]\w*)\b")
ENDMODULE_RE = re.compile(r"\bendmodule\b")
LINE_SUFFIX_RE = re.compile(r"^(.*)_(\d+)$")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def build_file_indexes() -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
    by_name: dict[str, list[Path]] = defaultdict(list)
    by_suffix: dict[str, list[Path]] = defaultdict(list)
    if not GITHUB_REPOS.exists():
        return by_name, by_suffix
    for path in GITHUB_REPOS.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(GITHUB_REPOS).as_posix()
        by_name[path.name].append(path)
        parts = rel.split("/")
        for i in range(len(parts)):
            suf = "/".join(parts[i:])
            by_suffix[suf].append(path)
    return by_name, by_suffix


def load_scraped_index() -> dict[str, list[dict]]:
    idx: dict[str, list[dict]] = defaultdict(list)
    if not GITHUB_SEARCH_SCRAPED.exists():
        return idx
    with open(GITHUB_SEARCH_SCRAPED) as f:
        for line in f:
            rec = json.loads(line)
            idx[Path(rec.get("file", "")).name].append(rec)
    return idx


def extract_module_or_window(path: Path, line_no: int | None = None) -> str:
    try:
        text = path.read_text(errors="ignore")
    except Exception:
        return ""
    lines = text.splitlines()
    if not lines:
        return ""
    if line_no is None:
        line_no = 1
    line_idx = max(0, min(len(lines) - 1, line_no - 1))

    # Find enclosing module around line.
    start = None
    for i in range(line_idx, -1, -1):
        if MODULE_RE.search(lines[i]):
            start = i
            break
    if start is not None:
        for j in range(start, min(len(lines), start + 1200)):
            if ENDMODULE_RE.search(lines[j]):
                return "\n".join(lines[start:j + 1])

    lo = max(0, line_idx - 40)
    hi = min(len(lines), line_idx + 80)
    return "\n".join(lines[lo:hi])


def parse_parent(parent_id: str) -> tuple[str | None, int | None]:
    m = LINE_SUFFIX_RE.match(parent_id)
    if not m:
        return None, None
    return m.group(1), int(m.group(2))


def candidate_paths(parent_path: str, by_name: dict[str, list[Path]], by_suffix: dict[str, list[Path]]) -> list[Path]:
    out: list[Path] = []
    clean = parent_path
    if clean.startswith("wave3_"):
        clean = clean[len("wave3_"):]
    if clean.startswith("github_search_scraped_?_"):
        clean = clean[len("github_search_scraped_?_"):]
    clean = clean.replace("__", "/")

    if clean in by_suffix:
        out.extend(by_suffix[clean])
    for part in clean.split("/"):
        if "/" in part:
            continue
    name = Path(clean).name
    if name in by_name:
        out.extend(by_name[name])

    uniq = []
    seen = set()
    for p in out:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


def fill_rows(rows: list[dict]) -> tuple[list[dict], dict]:
    by_name, by_suffix = build_file_indexes()
    scraped = load_scraped_index()
    counts = Counter()
    filled_rows = []

    for rec in rows:
        rec2 = dict(rec)
        if (rec2.get("rtl_context") or "").strip():
            rec2["rtl_fill_status"] = "already_present"
            filled_rows.append(rec2)
            counts["already_present"] += 1
            continue

        parent_id = rec2.get("parent_id", "")
        parent_path, line_no = parse_parent(parent_id)
        filled = ""
        status = "unfilled"

        if parent_path:
            paths = candidate_paths(parent_path, by_name, by_suffix)
            if len(paths) == 1:
                filled = extract_module_or_window(paths[0], line_no)
                status = "filled_from_local_file" if filled else "unfilled"
            elif len(paths) > 1:
                target_name = Path(parent_path).name
                same_name = [p for p in paths if p.name == target_name]
                if len(same_name) == 1:
                    filled = extract_module_or_window(same_name[0], line_no)
                    status = "filled_from_local_file" if filled else "unfilled"
                else:
                    status = "ambiguous_local_file"
            else:
                target_name = Path(parent_path).name
                scraped_hits = scraped.get(target_name, [])
                if scraped_hits:
                    status = "scraped_only_no_local_file"
                else:
                    status = "no_matching_file"
        else:
            status = "unparseable_parent_id"

        if filled:
            rec2["rtl_context"] = filled
        rec2["rtl_fill_status"] = status
        filled_rows.append(rec2)
        counts[status] += 1

    manifest = {
        "counts": dict(counts),
        "total": len(rows),
        "filled_total": sum(1 for r in filled_rows if (r.get("rtl_context") or "").strip()),
    }
    return filled_rows, manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(DEFAULT_IN))
    ap.add_argument("--output", default=str(DEFAULT_OUT))
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    args = ap.parse_args()

    with open(args.input) as f:
        rows = [json.loads(line) for line in f]

    filled_rows, manifest = fill_rows(rows)
    write_jsonl(Path(args.output), filled_rows)
    with open(args.manifest, "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
