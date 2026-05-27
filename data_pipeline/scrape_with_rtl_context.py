#!/usr/bin/env python3
"""
scrape_with_rtl_context.py — re-extract SVAs from local snapshot dirs with
the enclosing module as rtl_context.

Root cause we're fixing: the NL2SVA-Human test set has 100% coverage of real
testbench RTL (mean 2616 chars), but the existing SFT pool has 97% empty
rtl_context and the Pilot 7 industrial pool has 100% empty. Training
without real RTL → policy never learns to ground SVAs in the surrounding
HDL → eval RTL becomes distracting context rather than grounding.

For each snapshot dir:
  1. Walk all .sv/.svh/.v/.vh files
  2. Find every `assert property(...)` (inline) and `property ... endproperty;
     <name> : assert property(<name>);` (named)
  3. Capture the enclosing `module|package|class|interface` block as
     rtl_context (truncated to 4000 chars to match typical test-set length)
  4. Dedup by body hash
  5. Write per-record: {id, source, sva, rtl_context, module, line, hash}

Output: data/train/grpo/industrial_with_rtl.jsonl
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent
OUT = EXPERIMENTS_DIR / "data" / "train" / "grpo" / "industrial_with_rtl.jsonl"

SV_EXTS = {".sv", ".svh", ".v", ".vh"}
MAX_RTL = 4000

# Find enclosing block: module|package|class|interface ... endmodule|endpackage|...
BLOCK_OPENERS = ("module", "package", "class", "interface", "program")
BLOCK_CLOSERS = {
    "module": "endmodule", "package": "endpackage",
    "class": "endclass", "interface": "endinterface",
    "program": "endprogram",
}
OPENER_RE = re.compile(
    r"\b(module|package|class|interface|program)\s+([A-Za-z_]\w*)", re.M
)

# SVA patterns
INLINE_ASSERT_RE = re.compile(
    r"(?:\b(\w+)\s*:\s*)?(assert|assume|cover)\s+property\s*\(",
    re.IGNORECASE,
)
NAMED_PROP_RE = re.compile(
    r"\bproperty\s+([A-Za-z_]\w*)\s*(?:\([^)]*\))?\s*;(.+?)\bendproperty\b",
    re.DOTALL,
)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _hash(s: str) -> str:
    return hashlib.sha256(_norm(s).encode()).hexdigest()[:16]


def _find_matching_paren(text: str, open_idx: int) -> int:
    """Returns index of the ')' that matches the '(' at open_idx. Returns -1
    if unbalanced."""
    depth = 0
    i = open_idx
    n = len(text)
    in_str = False
    in_line_comment = False
    in_block_comment = False
    while i < n:
        c = text[i]
        if in_line_comment:
            if c == "\n":
                in_line_comment = False
        elif in_block_comment:
            if c == "*" and i + 1 < n and text[i + 1] == "/":
                in_block_comment = False
                i += 1
        elif in_str:
            if c == '"' and text[i - 1] != "\\":
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "/" and i + 1 < n and text[i + 1] == "/":
                in_line_comment = True
            elif c == "/" and i + 1 < n and text[i + 1] == "*":
                in_block_comment = True
                i += 1
            elif c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    return i
        i += 1
    return -1


def _grab_preceding_comment(text: str, pos: int) -> str:
    """Walk backward from `pos` line-by-line; accumulate consecutive comment
    lines ending just before the SVA. Returns cleaned comment text.
    Stops at the first non-comment non-blank line, or at a limit of 8 lines."""
    # Find start of the line containing pos
    line_start = text.rfind("\n", 0, pos)
    if line_start < 0:
        return ""
    lines_above = []
    cursor = line_start
    for _ in range(8):
        prev_nl = text.rfind("\n", 0, cursor)
        line = text[prev_nl + 1:cursor].rstrip() if prev_nl >= 0 else text[:cursor].rstrip()
        stripped = line.strip()
        if stripped.startswith("//"):
            lines_above.append(stripped.lstrip("/").strip())
        elif stripped.startswith("/*") or stripped.endswith("*/"):
            # simple block-comment line capture
            s = stripped.strip("/*").strip("*/").strip()
            if s:
                lines_above.append(s)
        elif not stripped:
            # blank line stops the comment chain if we've collected any
            if lines_above:
                break
        else:
            break
        cursor = prev_nl
        if cursor < 0:
            break
    if not lines_above:
        return ""
    # lines_above is in reverse order (bottom→up) — flip back
    return " ".join(reversed(lines_above)).strip()


def _find_enclosing_block(text: str, pos: int):
    """Walk backward from `pos` to find an opener; then forward to its matching
    closer. Returns (block_text, block_kind) or (None, None)."""
    last_open = None
    last_kind = None
    for m in OPENER_RE.finditer(text, 0, pos):
        last_open = m
        last_kind = m.group(1)
    if not last_open:
        return None, None
    # walk forward to its 'end<kind>'
    closer = BLOCK_CLOSERS[last_kind]
    # naive: find first occurrence of `end<kind>` after pos, at the closer depth
    # a proper impl would track nested blocks, but good enough for most designs
    end_idx = text.find(closer, pos)
    if end_idx < 0:
        return None, None
    start = last_open.start()
    block = text[start:end_idx + len(closer)]
    return block, last_kind


def _extract_inline(text: str):
    """Yield (sva_text, full_line_pos) for each inline `assert property(...)`."""
    for m in INLINE_ASSERT_RE.finditer(text):
        paren_start = text.find("(", m.end() - 1)
        if paren_start < 0:
            continue
        paren_end = _find_matching_paren(text, paren_start)
        if paren_end < 0:
            continue
        sva_start = m.start()
        sva_end = paren_end + 1
        # include trailing ';' if present
        rest = text[sva_end: sva_end + 200]
        # also include 'else <stmt>;' suffix for clean formatting
        semi = rest.find(";")
        if semi >= 0 and semi < 150:
            sva_end += semi + 1
        sva_text = text[sva_start:sva_end]
        yield sva_text, sva_start


def _extract_named(text: str):
    """Yield (named-assert-text, pos) — wraps `property X; <body> endproperty`
    into `assert property(<body>);`."""
    for m in NAMED_PROP_RE.finditer(text):
        name = m.group(1)
        body = m.group(2).strip().rstrip(";").strip()
        if not body or re.fullmatch(r"[A-Za-z_]\w*", body):
            continue
        sva = f"assert property ({body});"
        yield sva, m.start()


def extract_records(path: Path, source_repo: str, seen: set):
    """Return list of records with rtl_context + sva."""
    try:
        text = path.read_text(errors="ignore")
    except Exception:
        return []
    if "assert" not in text and "property" not in text:
        return []
    records = []
    for extractor_name, gen in [("inline", _extract_inline(text)),
                                 ("named", _extract_named(text))]:
        for sva_text, pos in gen:
            if len(sva_text) < 30 or len(sva_text) > 4000:
                continue
            h = _hash(sva_text)
            if h in seen:
                continue
            seen.add(h)
            block, kind = _find_enclosing_block(text, pos)
            if block and len(block) > 80:
                # Truncate long modules but preserve header (inputs/outputs)
                rtl_context = block[:MAX_RTL]
            else:
                rtl_context = ""
            nl_comment = _grab_preceding_comment(text, pos)
            records.append({
                "sva": sva_text,
                "rtl_context": rtl_context,
                "nl_comment": nl_comment,
                "strategy": extractor_name,
                "file": str(path),
                "line": text[:pos].count("\n") + 1,
                "module_kind": kind or "",
                "source_repo": source_repo,
                "body_hash": h,
            })
    return records


def repo_from_dir(d: Path) -> str:
    name = d.name
    name = re.sub(r"_pr\d+$", "", name)
    parts = name.split("_", 1)
    return f"{parts[0]}/{parts[1]}" if len(parts) == 2 else name


def repo_from_archive(name: str) -> str:
    s = name.replace(".tar.gz", "")
    s = re.sub(r"_[0-9a-f]{40}$", "", s)
    parts = s.split("_", 1)
    return f"{parts[0]}/{parts[1]}" if len(parts) == 2 else s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot-roots", nargs="+",
                    default=[
                        "${SVA_CORPUS_ROOT}/HWE-train/snapshots",
                        "${SVA_CORPUS_ROOT}/HDL_all/snapshots",
                        "${SVA_CORPUS_ROOT}/HDL_verified_new/snapshots",
                    ])
    ap.add_argument("--limit-dirs", type=int, default=0,
                    help="cap how many snapshot dirs to scan (for smoke test)")
    args = ap.parse_args()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    total_records = 0
    per_repo = {}

    import tarfile, tempfile
    out = open(OUT, "w")
    for root in args.snapshot_roots:
        root = Path(root)
        if not root.exists():
            print(f"[skip] {root}")
            continue
        pr_dirs = sorted(d for d in root.iterdir() if d.is_dir())
        archives = sorted(f for f in root.iterdir()
                          if f.is_file() and f.name.endswith(".tar.gz"))
        if args.limit_dirs:
            pr_dirs = pr_dirs[:args.limit_dirs]
            archives = archives[:args.limit_dirs]
        print(f"\n[{root.name}] {len(pr_dirs)} dirs + {len(archives)} archives")
        for i, d in enumerate(pr_dirs, 1):
            repo = repo_from_dir(d)
            n_before = len(seen)
            for f in d.rglob("*"):
                if not f.is_file() or f.suffix.lower() not in SV_EXTS:
                    continue
                if any(p in (".git", "build", "_out", "node_modules")
                       for p in f.parts):
                    continue
                recs = extract_records(f, repo, seen)
                for r in recs:
                    out.write(json.dumps(r) + "\n")
                    total_records += 1
            new_this = len(seen) - n_before
            per_repo[repo] = per_repo.get(repo, 0) + new_this
            if i % 50 == 0 or new_this > 20:
                print(f"  [{i}/{len(pr_dirs)}] {d.name}: +{new_this} "
                      f"(total {total_records})")
        # Process .tar.gz archives
        for i, arc in enumerate(archives, 1):
            repo = repo_from_archive(arc.name)
            n_before = len(seen)
            try:
                with tempfile.TemporaryDirectory(prefix="sva_snap_") as tmp:
                    tmp_path = Path(tmp)
                    with tarfile.open(arc, "r:gz") as tf:
                        members = [m for m in tf.getmembers()
                                   if Path(m.name).suffix.lower() in SV_EXTS]
                        tf.extractall(tmp_path, members=members)
                    for f in tmp_path.rglob("*"):
                        if not f.is_file() or f.suffix.lower() not in SV_EXTS:
                            continue
                        recs = extract_records(f, repo, seen)
                        for r in recs:
                            out.write(json.dumps(r) + "\n")
                            total_records += 1
            except Exception as e:
                print(f"  arc[{i}] {arc.name}: ERR {e}")
                continue
            new_this = len(seen) - n_before
            per_repo[repo] = per_repo.get(repo, 0) + new_this
            if i % 50 == 0 or new_this > 20:
                print(f"  arc[{i}/{len(archives)}] {arc.name}: +{new_this} "
                      f"(total {total_records})")
    out.close()

    print(f"\n[done] wrote {total_records} records to {OUT}")
    print(f"[done] Top 20 repos by new SVAs:")
    for repo, n in sorted(per_repo.items(), key=lambda x: -x[1])[:20]:
        if n:
            print(f"  {n:5d}  {repo}")


if __name__ == "__main__":
    main()
