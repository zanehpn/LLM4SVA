#!/usr/bin/env python3
"""
expand_opentitan_macros.py — find `ASSERT_* / `ASSUME / `COVER macro
invocations in the cloned OpenTitan tree (and any other repo that uses the
same prim_assert.sv macro vocabulary), expand them to concrete SVA strings,
and append them to the training corpus.

Templates (from hw/ip/prim/rtl/prim_assert_standard_macros.svh):

  `ASSERT(name, prop, clk, rst)
     → name: assert property (@(posedge clk) disable iff ((rst) !== '0) (prop));
  `ASSERT_NEVER(name, prop, clk, rst)
     → name: assert property (@(posedge clk) disable iff ((rst) !== '0) not (prop));
  `ASSERT_KNOWN(name, sig, clk, rst)
     → expands to `ASSERT(name, !$isunknown(sig), clk, rst)
  `ASSUME(name, prop, clk, rst)
     → name: assume property (@(posedge clk) disable iff ((rst) !== '0) (prop));
  `COVER(name, prop, clk, rst)
     → name: cover property (@(posedge clk) disable iff ((rst) !== '0) (prop));
  `ASSERT_AT_RESET(name, prop, rst)
     → name: assert property (@(posedge rst) $isunknown(rst) || (prop));

Default clock/reset signals (when only 2 args supplied): clk_i, !rst_ni
(matching OpenTitan's `ASSERT_DEFAULT_CLK` / `ASSERT_DEFAULT_RST`).

Output: data/raw/opentitan_macros/expanded.jsonl
Register as `opentitan_macros` source in fetch_benchmarks.py.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import List, Optional

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent
REPOS_DIR = EXPERIMENTS_DIR / "data" / "raw" / "github_repos"
OUT_DIR = EXPERIMENTS_DIR / "data" / "raw" / "opentitan_macros"
OUT_JSONL = OUT_DIR / "expanded.jsonl"

SV_EXTS = {".sv", ".svh", ".v", ".vh"}
MACROS = ("ASSERT", "ASSERT_NEVER", "ASSERT_KNOWN", "ASSERT_AT_RESET",
          "ASSUME", "COVER")
DEFAULT_CLK = "clk_i"
DEFAULT_RST = "!rst_ni"


# -----------------------------------------------------------------------------
# Argument extraction with balanced parens + comma-respecting split
# -----------------------------------------------------------------------------
def _balanced_close(text: str, open_idx: int) -> Optional[int]:
    depth = 0
    i = open_idx
    n = len(text)
    in_blk_cmt = in_line_cmt = in_str = False
    while i < n:
        ch = text[i]; nxt = text[i + 1] if i + 1 < n else ""
        if in_blk_cmt:
            if ch == "*" and nxt == "/":
                in_blk_cmt = False; i += 2; continue
        elif in_line_cmt:
            if ch == "\n": in_line_cmt = False
        elif in_str:
            if ch == "\\" and i + 1 < n:
                i += 2; continue
            if ch == '"': in_str = False
        else:
            if ch == "/" and nxt == "*": in_blk_cmt = True; i += 2; continue
            if ch == "/" and nxt == "/": in_line_cmt = True; i += 2; continue
            if ch == '"': in_str = True
            elif ch == "(": depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0: return i
        i += 1
    return None


def split_args(arg_str: str) -> List[str]:
    """Split at top-level commas — respecting nested (), [], {}, strings, and
    line comments.  Trailing line-continuation backslashes are stripped."""
    parts = []
    buf = []
    depth_p = depth_b = depth_c = 0
    in_str = False
    i = 0
    n = len(arg_str)
    while i < n:
        ch = arg_str[i]; nxt = arg_str[i + 1] if i + 1 < n else ""
        if in_str:
            buf.append(ch)
            if ch == "\\" and i + 1 < n:
                buf.append(nxt); i += 2; continue
            if ch == '"': in_str = False
            i += 1; continue
        if ch == "/" and nxt == "/":
            # skip line comment
            while i < n and arg_str[i] != "\n": i += 1
            continue
        if ch == '"': in_str = True; buf.append(ch); i += 1; continue
        if ch == "(": depth_p += 1
        elif ch == ")": depth_p -= 1
        elif ch == "[": depth_b += 1
        elif ch == "]": depth_b -= 1
        elif ch == "{": depth_c += 1
        elif ch == "}": depth_c -= 1
        elif ch == "," and depth_p == depth_b == depth_c == 0:
            parts.append("".join(buf).strip())
            buf = []; i += 1; continue
        buf.append(ch); i += 1
    if buf:
        parts.append("".join(buf).strip())
    # strip line-continuation backslashes inside each arg
    parts = [re.sub(r"\\\s*\n", " ", p).strip() for p in parts]
    parts = [re.sub(r"\s+", " ", p) for p in parts]
    return parts


# -----------------------------------------------------------------------------
# Template expansion
# -----------------------------------------------------------------------------
def expand(macro: str, args: List[str]) -> Optional[str]:
    # Fill in default clk / rst when omitted
    def with_defaults(expected):
        out = list(args)
        while len(out) < expected:
            if len(out) == expected - 2: out.append(DEFAULT_CLK)
            elif len(out) == expected - 1: out.append(DEFAULT_RST)
            else: break
        return out if len(out) >= expected - 2 else None

    if macro == "ASSERT":
        a = with_defaults(4)
        if not a: return None
        name, prop, clk, rst = (a + [DEFAULT_CLK, DEFAULT_RST])[:4]
        return (f"{name}: assert property (@(posedge {clk}) "
                f"disable iff (({rst}) !== '0) ({prop}));")
    if macro == "ASSERT_NEVER":
        a = with_defaults(4)
        if not a: return None
        name, prop, clk, rst = (a + [DEFAULT_CLK, DEFAULT_RST])[:4]
        return (f"{name}: assert property (@(posedge {clk}) "
                f"disable iff (({rst}) !== '0) not ({prop}));")
    if macro == "ASSERT_KNOWN":
        a = with_defaults(4)
        if not a: return None
        name, sig, clk, rst = (a + [DEFAULT_CLK, DEFAULT_RST])[:4]
        return (f"{name}: assert property (@(posedge {clk}) "
                f"disable iff (({rst}) !== '0) (!$isunknown({sig})));")
    if macro == "ASSUME":
        a = with_defaults(4)
        if not a: return None
        name, prop, clk, rst = (a + [DEFAULT_CLK, DEFAULT_RST])[:4]
        return (f"{name}: assume property (@(posedge {clk}) "
                f"disable iff (({rst}) !== '0) ({prop}));")
    if macro == "COVER":
        a = with_defaults(4)
        if not a: return None
        name, prop, clk, rst = (a + [DEFAULT_CLK, DEFAULT_RST])[:4]
        return (f"{name}: cover property (@(posedge {clk}) "
                f"disable iff (({rst}) !== '0) ({prop}));")
    if macro == "ASSERT_AT_RESET":
        a = args + [DEFAULT_RST]
        if len(a) < 3: return None
        name, prop, rst = a[0], a[1], a[2]
        return (f"{name}: assert property (@(posedge {rst}) "
                f"$isunknown({rst}) || ({prop}));")
    return None


# -----------------------------------------------------------------------------
# File walker
# -----------------------------------------------------------------------------
_MACRO_START_RE = re.compile(r"`(" + "|".join(MACROS) + r")\s*\(")


def scan_file(path: Path) -> List[dict]:
    try:
        text = path.read_text(errors="ignore")
    except Exception:
        return []
    out = []
    for m in _MACRO_START_RE.finditer(text):
        macro = m.group(1)
        open_paren = m.end() - 1
        close_paren = _balanced_close(text, open_paren)
        if close_paren is None:
            continue
        arg_str = text[open_paren + 1:close_paren]
        args = split_args(arg_str)
        if not args or len(args) < 2:
            continue
        sva = expand(macro, args)
        if not sva:
            continue
        # Skip macros whose `prop` is an identifier only (placeholder refs)
        prop = args[1] if len(args) > 1 else ""
        if re.fullmatch(r"[A-Za-z_]\w*", prop):
            continue
        out.append({
            "macro": macro,
            "sva": sva,
            "args": args,
            "file": str(path),
            "line": text.count("\n", 0, m.start()) + 1,
        })
    return out


def walk_repo(repo_dir: Path) -> List[dict]:
    out = []
    for f in repo_dir.rglob("*"):
        if not f.is_file() or f.suffix.lower() not in SV_EXTS:
            continue
        if any(part in ("build", "_out", ".git") for part in f.parts):
            continue
        out.extend(scan_file(f))
    return out


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos", nargs="*",
                    default=["lowRISC__opentitan", "lowRISC__ibex",
                             "openhwgroup__cva6", "openhwgroup__cvw",
                             "openhwgroup__cv32e40p", "openhwgroup__cv32e40x",
                             "pulp-platform__axi", "pulp-platform__common_cells",
                             "pulp-platform__snitch_cluster",
                             "chipsalliance__Caliptra-RTL"],
                    help="subdirectory names under data/raw/github_repos/")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seen = set()
    total = kept = 0
    by_repo = {}
    by_macro = {}
    with open(OUT_JSONL, "w") as out:
        for rname in args.repos:
            rdir = REPOS_DIR / rname
            if not rdir.exists():
                print(f"  [skip] {rname}: not cloned")
                continue
            recs = walk_repo(rdir)
            total += len(recs)
            repo_kept = 0
            for r in recs:
                h = hashlib.sha256(re.sub(r"\s+", " ", r["sva"]).encode()).hexdigest()[:16]
                if h in seen:
                    continue
                seen.add(h)
                r["source_repo"] = rname.replace("__", "/")
                r["body_hash"] = h
                r["file"] = str(Path(r["file"]).relative_to(rdir))
                out.write(json.dumps(r) + "\n")
                by_macro[r["macro"]] = by_macro.get(r["macro"], 0) + 1
                kept += 1
                repo_kept += 1
            by_repo[rname] = repo_kept
            print(f"  [scan] {rname:<40} {len(recs):>5} found, "
                  f"{repo_kept:>5} unique")

    print(f"\n[expand] total raw invocations: {total}")
    print(f"[expand] unique SVAs written:   {kept}")
    print(f"[expand] output: {OUT_JSONL}")
    print(f"\nPer-macro:")
    for m, n in sorted(by_macro.items(), key=lambda x: -x[1]):
        print(f"  {m:<20} {n}")
    print(f"\nPer-repo:")
    for r, n in sorted(by_repo.items(), key=lambda x: -x[1]):
        print(f"  {r:<40} {n}")


if __name__ == "__main__":
    main()
