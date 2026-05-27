"""backfill_master_rtl.py
Find the original source files in data/raw/github_repos and replace
truncated / placeholder rtl_context with the real module body.

The IDs in master_train.jsonl encode the source file path for several
prefixes:
    named_properties_<org>_<repo>_<path>_<file>.sv_L<line>_<name>
    github_scraped_<org>_<repo>_<path>_<file>.sv_L<line>
    snapshots_scraped_<org>_<repo>_pr<pr>_<inline|named>_<path>.sv_L<line>
    github_search_scraped_?_<inline|named>_<path>.sv_L<line>   (no org/repo)

Synthetic rows (synthetic_l3 / synthetic_l5 / scrape_a / rtl_aware) carry
no source path in their IDs and are NOT handled here — they need a
separate parent-trace pipeline.

For matchable rows we extract the smallest module {block} that contains
the SVA at the given line. If we cannot find the file or the module
boundary, we leave rtl_context untouched and record a status.

Usage:
    python scripts/backfill_master_rtl.py \
        --input  data/master/master_train_norm.jsonl \
        --output data/master/master_train_rtl_filled.jsonl
"""
from __future__ import annotations
import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GITHUB_REPOS = ROOT / "data" / "raw" / "github_repos"


# Parse "<file>.sv_L<line>" — the unambiguous tail token in the ID.
FILE_LINE_RE = re.compile(r"_([A-Za-z0-9][\w\-]*?\.(?:sv|svh|v|vh))_L(\d+)(?=_|$)")

# Strict module-name extractor. Two forms:
#   - placeholder comment line:   `// module <name>` (whole line)
#   - real RTL line:              `module <name> (` / `;` / `#` / `import` /
#                                  `//` — i.e. SV-valid token follows the name
# Avoids matching English-prose lines like
#   `module expects both AW and W valid before it does something`
# where the next char is just another word.
RTL_PLACEHOLDER_RE = re.compile(
    r"^\s*//\s*module\s+([A-Za-z_]\w*)\s*$", re.MULTILINE)
RTL_REAL_MODULE_RE = re.compile(
    r"^\s*module\s+([A-Za-z_]\w*)\b\s*(?:\(|;|#|import\b|/\*|//|$)",
    re.MULTILINE)
RTL_INTERFACE_RE = re.compile(
    r"^\s*interface\s+([A-Za-z_]\w*)\b\s*(?:\(|;|#|import\b|/\*|//|$)",
    re.MULTILINE)
RTL_PACKAGE_RE = re.compile(
    r"^\s*package\s+([A-Za-z_]\w*)\b\s*;",
    re.MULTILINE)
# These are common English words / SV keywords that must never be accepted as
# a module-name pointer. Any 1- or 2-character name is also rejected.
BAD_NAMES = {
    "is", "has", "a", "an", "the", "this", "that", "it", "we", "you",
    "expects", "boundary", "code", "needs", "should", "can", "must",
    "and", "or", "but", "if", "then", "else", "when", "where",
    # SV reserved words
    "module", "endmodule", "interface", "endinterface", "package",
    "endpackage", "class", "endclass", "function", "endfunction",
    "task", "endtask", "begin", "end", "logic", "wire", "reg",
    "input", "output", "inout", "parameter", "localparam", "assign",
    "always", "always_ff", "always_comb", "always_latch", "initial",
    "generate", "endgenerate", "import", "export", "typedef",
}


def extract_decl_name_from_rtl(rtl: str) -> tuple[str, str] | None:
    """Return (kind, name) where kind is 'module' / 'interface' / 'package'.
    Tries placeholder + real module first, then interface, then package.
    Returns None if nothing valid is found."""
    for regex, kind in [
        (RTL_PLACEHOLDER_RE, "module"),
        (RTL_REAL_MODULE_RE, "module"),
        (RTL_INTERFACE_RE, "interface"),
        (RTL_PACKAGE_RE, "package"),
    ]:
        m = regex.search(rtl)
        if not m:
            continue
        name = m.group(1)
        if len(name) < 3:
            continue
        if name.lower() in BAD_NAMES:
            continue
        return kind, name
    return None


SCRAPED_FILES = [
    ROOT / "data" / "raw" / "named_properties" / "scraped.jsonl",
    ROOT / "data" / "raw" / "github_scraped" / "scraped.jsonl",
    ROOT / "data" / "raw" / "snapshots_scraped" / "scraped.jsonl",
    ROOT / "data" / "raw" / "github_search_scraped" / "scraped.jsonl",
]


def normalize_sva_for_match(sva: str) -> str:
    """Normalize an SVA string for cross-file lookup. Strips whitespace and
    lowercases. Ignores the wrapping `assert property` and trailing `;`."""
    if not sva:
        return ""
    s = sva.lower()
    s = re.sub(r"\s+", "", s)
    return s


def build_scraped_index() -> dict[str, list[dict]]:
    """Map normalized-SVA-string -> list of scraped row dicts (each with
    source_repo, file, line)."""
    idx: dict[str, list[dict]] = defaultdict(list)
    for path in SCRAPED_FILES:
        if not path.exists():
            continue
        with open(path) as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                row = json.loads(ln)
                key = normalize_sva_for_match(row.get("sva") or "")
                if not key:
                    continue
                row["_origin_file"] = str(path.relative_to(ROOT))
                idx[key].append(row)
    return idx
MODULE_START_RE = re.compile(r"^\s*module\s+([A-Za-z_]\w*)", re.MULTILINE)
ENDMODULE_RE = re.compile(r"^\s*endmodule\b", re.MULTILINE)
END_DECL_RE = re.compile(
    r"^\s*(?:endmodule|endinterface|endpackage)\b", re.MULTILINE)


def build_basename_index() -> dict[str, list[Path]]:
    idx: dict[str, list[Path]] = defaultdict(list)
    for p in GITHUB_REPOS.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() in (".sv", ".v", ".svh", ".vh"):
            idx[p.name].append(p)
    return idx


def build_module_index() -> dict[str, list[tuple[Path, int]]]:
    """Map declaration-name -> list[(file_path, declaration_line)] for
    module / interface / package declarations across github_repos."""
    idx: dict[str, list[tuple[Path, int]]] = defaultdict(list)
    patterns = [
        re.compile(r"^\s*module\s+([A-Za-z_]\w*)\b\s*(?:\(|;|#|import\b|/\*|//|$)"),
        re.compile(r"^\s*interface\s+([A-Za-z_]\w*)\b\s*(?:\(|;|#|import\b|/\*|//|$)"),
        re.compile(r"^\s*package\s+([A-Za-z_]\w*)\b\s*;"),
    ]
    for p in GITHUB_REPOS.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in (".sv", ".v", ".svh", ".vh"):
            continue
        try:
            with open(p, errors="replace") as f:
                for line_no, line in enumerate(f, start=1):
                    for pat in patterns:
                        m = pat.match(line)
                        if m:
                            idx[m.group(1)].append((p, line_no))
                            break
        except Exception:
            continue
    return idx


def parse_id_to_source(id_str: str, basename_index: dict[str, list[Path]]):
    """Try to extract (basename, line_no, hint_tokens) from id_str.

    The captured group of FILE_LINE_RE concatenates the encoded path with
    `_` separators, e.g.,
        'properties_chipsalliance_Caliptra-RTL_src_..._el2_lsu.sv'
    The actual file basename is some suffix of this string. We split by
    `_`, then try suffixes from shortest to longest, picking the longest
    suffix whose `*.sv` (or `.v`/`.svh`/`.vh`) is actually a known file
    basename in the github_repos index. This lets us correctly resolve
    files like `el2_lsu.sv` whose basename itself contains underscores.
    """
    if not id_str:
        return None
    m = FILE_LINE_RE.search(id_str)
    if not m:
        return None
    captured = m.group(1)
    line_no = int(m.group(2))
    parts = captured.split("_")
    best = None  # (basename, hint_tokens)
    for i in range(1, len(parts) + 1):
        candidate = "_".join(parts[-i:])
        if candidate in basename_index:
            best = (candidate, parts[:-i])
            # keep going to favor longer, more specific basenames
    if best is None:
        return None
    basename, hint = best
    drop = {"", "named", "properties", "github", "scraped", "search",
            "snapshots", "inline", "pr"}
    hint = [t for t in hint if t and t not in drop and not t.isdigit()]
    return basename, line_no, hint


def best_file_match(candidates: list[Path], hint: list[str]) -> Path | None:
    """Pick the candidate whose path contains the most hint tokens in order."""
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    if not hint:
        return candidates[0]

    def score(p: Path) -> tuple[int, int]:
        # token-overlap count + longer-path-as-tiebreaker
        path_str = str(p).lower()
        return (sum(1 for h in hint if h.lower() in path_str), -len(path_str))

    return max(candidates, key=score)


def extract_module_around_line(text: str, line_no: int) -> str | None:
    """Return the module/interface/package block whose body covers
    `line_no`. If we can't find a containing decl, return None."""
    lines = text.splitlines(keepends=True)
    if line_no < 1 or line_no > len(lines):
        return text if 0 < len(text) <= 32_000 else None

    decl_re = re.compile(
        r"\s*(module|interface|package)\s+([A-Za-z_]\w*)")
    decl_starts = []
    for idx, ln in enumerate(lines):
        m = decl_re.match(ln)
        if m:
            decl_starts.append((idx, m.group(1), m.group(2)))

    target_idx = line_no - 1
    candidate = None
    for entry in decl_starts:
        if entry[0] <= target_idx:
            candidate = entry
        else:
            break
    if not candidate:
        return None
    rest = "".join(lines[candidate[0]:])
    em = END_DECL_RE.search(rest)
    if not em:
        return rest if rest else None
    return rest[:em.end()]


def _try_repo_path(source_repo: str, file_rel: str,
                   line_no: int, idx: dict[str, list[Path]]):
    """Given (source_repo, file_rel, line) attempt to locate the file
    inside data/raw/github_repos and extract the module containing
    `line_no`. Returns the new rtl_context body or None."""
    if not source_repo or not file_rel:
        return None
    repo_dir = source_repo.replace("/", "__")
    candidate = GITHUB_REPOS / repo_dir / file_rel
    if not candidate.is_file():
        # Try basename-only match within the repo dir
        parts = file_rel.split("/")
        basename = parts[-1]
        repo_root = GITHUB_REPOS / repo_dir
        if not repo_root.is_dir():
            # The repo isn't cloned at all; fall back to global basename idx
            for cand in idx.get(basename, []):
                rel = cand.relative_to(GITHUB_REPOS)
                if any(seg in rel.as_posix() for seg in parts[:-1]):
                    candidate = cand
                    break
            else:
                return None
        else:
            matches = list(repo_root.rglob(basename))
            if not matches:
                return None
            candidate = matches[0]
    try:
        text = candidate.read_text(errors="replace")
    except Exception:
        return None
    body = extract_module_around_line(text, line_no)
    if not body:
        return None
    return body, str(candidate.relative_to(GITHUB_REPOS))


def backfill_row(row: dict, idx: dict[str, list[Path]],
                 scraped_idx: dict[str, list[dict]],
                 module_idx: dict[str, list[tuple[Path, int]]],
                 stats: Counter):
    rtl = row.get("rtl_context") or ""
    if "endmodule" in rtl and len(rtl) > 200:
        stats["already_proper"] += 1
        return row, "already_proper"

    # ---- Strategy 1: parse the master_train ID for a file path ----
    parsed = parse_id_to_source(row.get("id"), idx)
    if parsed is not None:
        basename, line_no, hint = parsed
        candidates = idx.get(basename) or []
        if candidates:
            chosen = best_file_match(candidates, hint)
            try:
                text = chosen.read_text(errors="replace")
                body = extract_module_around_line(text, line_no)
            except Exception:
                body = None
            if body:
                new_row = dict(row)
                new_row["rtl_context"] = body
                new_row["_rtl_backfill"] = {
                    "via": "id_parse",
                    "source_file": str(chosen.relative_to(GITHUB_REPOS)),
                    "line": line_no,
                    "module_chars": len(body),
                }
                stats["backfilled_via_id"] += 1
                return new_row, "backfilled_via_id"

    # ---- Strategy 2: SVA-string match against scraped corpora ----
    key = normalize_sva_for_match(row.get("reference_sva") or "")
    matches = scraped_idx.get(key, []) if key else []
    for m in matches:
        sr = m.get("source_repo") or m.get("source_snapshot")
        fp = m.get("file")
        try:
            line_no = int(m.get("line") or 0)
        except (TypeError, ValueError):
            line_no = 0
        result = _try_repo_path(sr, fp, line_no, idx)
        if result is None:
            continue
        body, rel = result
        new_row = dict(row)
        new_row["rtl_context"] = body
        new_row["_rtl_backfill"] = {
            "via": "scraped_sva_match",
            "source_repo": sr,
            "source_file": rel,
            "scraped_origin": m.get("_origin_file"),
            "line": line_no,
            "module_chars": len(body),
        }
        stats["backfilled_via_scraped"] += 1
        return new_row, "backfilled_via_scraped"

    # ---- Strategy 3: extract decl name (module/interface/package) ----
    name_match = extract_decl_name_from_rtl(rtl)
    if name_match:
        kind, modname = name_match
        candidates = module_idx.get(modname) or []
        for cand_path, decl_line in candidates:
            try:
                text = cand_path.read_text(errors="replace")
            except Exception:
                continue
            body = extract_module_around_line(text, decl_line)
            if body:
                new_row = dict(row)
                new_row["rtl_context"] = body
                new_row["_rtl_backfill"] = {
                    "via": "module_name",
                    "decl_kind": kind,
                    "module_name": modname,
                    "source_file": str(cand_path.relative_to(GITHUB_REPOS)),
                    "module_chars": len(body),
                }
                stats["backfilled_via_module_name"] += 1
                return new_row, "backfilled_via_module_name"

    if not parsed and not name_match:
        stats["id_unparseable_no_scraped"] += 1
        return row, "id_unparseable"
    if matches:
        stats["scraped_repo_missing"] += 1
        return row, "scraped_repo_missing"
    if name_match:
        stats["module_name_not_in_repos"] += 1
        return row, "module_name_not_in_repos"
    stats["file_not_in_repos"] += 1
    return row, "file_not_in_repos"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",
                    default=str(ROOT / "data" / "master" / "master_train_norm.jsonl"))
    ap.add_argument("--output",
                    default=str(ROOT / "data" / "master" / "master_train_rtl_filled.jsonl"))
    ap.add_argument("--max-module-chars", type=int, default=32_000,
                    help="hard cap on the backfilled rtl_context size")
    ap.add_argument("--drop-noise-placeholders", action="store_true",
                    help="drop rows whose rtl_context is a `// module <X>` "
                         "placeholder where X is an English word / 1-2 char "
                         "junk (BAD_NAMES) and we couldn't backfill it")
    args = ap.parse_args()

    src = Path(args.input)
    dst = Path(args.output)
    if not src.exists():
        sys.exit(f"missing input: {src}")
    if not GITHUB_REPOS.exists():
        sys.exit(f"missing github_repos at: {GITHUB_REPOS}")

    print(f"[index] scanning {GITHUB_REPOS} for *.sv/*.v files...")
    t0 = time.time()
    idx = build_basename_index()
    print(f"[index] {sum(len(v) for v in idx.values())} files indexed under "
          f"{len(idx)} unique basenames in {time.time()-t0:.1f}s")
    t0 = time.time()
    scraped_idx = build_scraped_index()
    print(f"[scraped] {sum(len(v) for v in scraped_idx.values())} scraped "
          f"rows indexed under {len(scraped_idx)} unique normalized SVAs "
          f"in {time.time()-t0:.1f}s")
    t0 = time.time()
    module_idx = build_module_index()
    print(f"[modules] {sum(len(v) for v in module_idx.values())} module "
          f"declarations under {len(module_idx)} unique names "
          f"in {time.time()-t0:.1f}s")

    stats = Counter()
    by_id_prefix = defaultdict(Counter)
    n_in = n_out = 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src) as fin, open(dst, "w") as fout:
        for line in fin:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            n_in += 1
            row = json.loads(line)
            new_row, status = backfill_row(row, idx, scraped_idx,
                                           module_idx, stats)
            prefix = (row.get("id") or "").split("_")[0]
            by_id_prefix[prefix][status] += 1
            rtl = new_row.get("rtl_context", "") or ""
            if len(rtl) > args.max_module_chars:
                new_row["rtl_context"] = rtl[: args.max_module_chars]
                new_row["_rtl_backfill"] = {
                    **(new_row.get("_rtl_backfill") or {}),
                    "truncated_to": args.max_module_chars,
                }
                stats["truncated_oversize"] += 1
            # Optional drop: rtl_context is a `// module <english>` noise
            # placeholder that backfill could not recover.
            if args.drop_noise_placeholders and len(rtl) < 200:
                rtl_lower = rtl.lower().strip()
                m = re.match(r"^//\s*module\s+([A-Za-z_]\w*)", rtl_lower)
                if m and (len(m.group(1)) < 3 or m.group(1) in BAD_NAMES):
                    stats["dropped_noise_placeholder"] += 1
                    continue
            fout.write(json.dumps(new_row, ensure_ascii=False) + "\n")
            n_out += 1

    print(f"\n[backfill_master_rtl]")
    print(f"  input : {src}")
    print(f"  output: {dst}")
    print(f"  rows in : {n_in}")
    print(f"  rows out: {n_out}")
    print(f"  status:")
    for s, c in stats.most_common():
        print(f"    {s:<25} {c:>6}  ({100*c/max(n_in,1):5.1f}%)")
    print(f"\n  by id-prefix:")
    for px, ctr in sorted(by_id_prefix.items(), key=lambda kv: -sum(kv[1].values())):
        total = sum(ctr.values())
        print(f"    {px:<22} (total {total})")
        for st, c in ctr.most_common():
            print(f"        {st:<22} {c:>5}  ({100*c/total:.0f}%)")


if __name__ == "__main__":
    main()
