"""normalize_sva_for_verilator.py
Apply syntactic transforms that turn industrial SVAs into Verilator-
compatible (`--lint-only --assert`) form. Each rule is a no-op on already-
clean SVAs and best-effort on dirty ones.

Rules (ordered):
  R1 strip backtick-macros : `\`FOO` -> `FOO`     (treat as identifier)
  R2 flatten hier paths    : `a.b.c` -> `a_b_c`
  R3 strip package prefix  : `pkg::SYM` -> `SYM`
  R4 strip else $action    : `else $error/$fatal/$warning(...)` -> ""
  R5 bounded liveness      : `s_eventually(X)` -> `##[1:32] (X)`
                            `s_until_with` / `until_with` similar bound
  R6 drop nested assert    : `(assert property (...) until_with X)` -> drop
                            (synthesis bug from method-3 expansion)
  R7 strip end-of-assert text after closing `;`
"""
from __future__ import annotations
import argparse
import json
import multiprocessing as mp
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# R1: backtick macros. Strip the backtick, treat as identifier.
_BTICK_RE = re.compile(r"`([A-Za-z_]\w*)")

# R2: hierarchical path. Match foo.bar (or foo.bar.baz) — but NOT array
# member like `arr[0].x` (Verilator handles `.` after `]` differently).
# Simple version: any contiguous run of `<id>.<id>(\.<id>)*`.
_HIER_RE = re.compile(r"\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*){1,}\b")

# R3: package scope.  pkg::SYM -> SYM
_PKG_RE = re.compile(r"\b[A-Za-z_]\w*::([A-Za-z_]\w*)")

# R4: else $action(...) — match balanced () after $task name.
_ELSE_ACT_RE = re.compile(
    r"\s*else\s*\$\w+\s*\([^()]*(?:\([^()]*\)[^()]*)*\)\s*", re.IGNORECASE)

# R5: liveness rewrites
_S_EVENTUALLY_RE = re.compile(r"\bs_eventually\s*\(", re.IGNORECASE)
_S_ALWAYS_RE = re.compile(r"\bs_always\s*\(", re.IGNORECASE)
_S_UNTIL_WITH_RE = re.compile(r"\bs_until_with\s*\(", re.IGNORECASE)
_S_UNTIL_RE = re.compile(r"\bs_until\s*\(", re.IGNORECASE)
_UNTIL_WITH_RE = re.compile(r"\buntil_with\b", re.IGNORECASE)
_NEXTTIME_RE = re.compile(r"\bnexttime\s*\(", re.IGNORECASE)

# R6: strip the inner `assert property (...) until_with` synthesis bug
# Pattern: `(assert property ( ... ) until_with X )` is malformed; the
# outer assert already wraps. We replace `assert property (` with empty
# inside another assert property body.
_NESTED_ASSERT_RE = re.compile(
    r"\(\s*assert\s+property\s*\(\s*", re.IGNORECASE)

# R8: bare `assert property (X);` with no clocking event → wrap X with
# @(posedge clk). Caliptra formal-property files use external `bind`
# clocks which we can't see in lint mode.
_BARE_ASSERT_RE = re.compile(
    r"(assert\s+property\s*\(\s*)(?!@)([^@()][^()]*?)(\s*\))\s*;",
    re.IGNORECASE | re.DOTALL)

# R9: hierarchical paths with bit-selects in the middle, like
#     gen_PE[0].box_i.s_in, arr[i].field, foo[3:0].bar
# Treat the whole chain as a single identifier by joining with `_`
# and dropping bit-select brackets.
_HIER_WITH_IDX_RE = re.compile(
    r"\b[A-Za-z_]\w*(?:\[[^\]]*\])?(?:\.[A-Za-z_]\w*(?:\[[^\]]*\])?){1,}\b")

# R10: typed cast `WIDTH'(expr)` → `(expr)` — Verilator handles cast in
# regular code but the pattern in SVA props sometimes trips lint.
_CAST_RE = re.compile(r"\b[A-Za-z_]\w*\s*'\s*\(")

# R11: drop in-SVA `//` comments (Verilator lint doesn't always like them
# inside property bodies).
_INLINE_COMMENT_RE = re.compile(r"//[^\n]*")


_AP_OPEN_RE = re.compile(r"\bassert\s+property\s*\(", re.IGNORECASE)
_CLOCK_RE = re.compile(r"@\s*\([^)]*\)")
_DI_RE = re.compile(r"\bdisable\s+iff\s*\([^)]*\)", re.IGNORECASE)

# R17: any of (assert|assume|cover) property ( ... ) <action>;  — strip the
# action block (pass action and/or `else <stmt>`). Otherwise the wrapper
# would have to declare the action's LHS as a writable signal; stripping
# is lossless for a lint check (action is purely a runtime side effect).
_ACTION_AP_OPEN_RE = re.compile(
    r"\b(?:assert|assume|cover)\s+property\s*\(", re.IGNORECASE)


def _wrap_unclocked_assertions(s: str) -> str:
    """Wrap `(assert|assume|cover) property (BODY);` with `@(posedge clk)`
    when BODY doesn't already start with a clocking event. Balanced-paren
    aware so BODY can contain nested parens (e.g. `!(init && next)`).
    Body that begins with `disable iff (X)` is also accepted (clock is
    inserted before the disable iff)."""
    out = []
    i = 0
    pat = re.compile(r"\b(?:assert|assume|cover)\s+property\s*\(",
                     re.IGNORECASE)
    while i < len(s):
        m = pat.search(s, i)
        if not m:
            out.append(s[i:])
            break
        out.append(s[i:m.end()])
        depth = 1
        j = m.end()
        while j < len(s) and depth > 0:
            if s[j] == '(': depth += 1
            elif s[j] == ')': depth -= 1
            j += 1
        if depth != 0:
            out.append(s[m.end():])
            break
        body = s[m.end():j-1]
        if not re.match(r"\s*@\s*\(", body):
            body = "@(posedge clk) " + body.lstrip()
        out.append(body)
        out.append(")")
        i = j
    return "".join(out)


def _strip_action_block(s: str) -> str:
    """R17: strip pass / else action blocks from
    `(assert|assume|cover) property (X) <action>;`.
    Empty action (`(X);`) is preserved. Common targets:
        assert property (X) hits[3] = 1;          -> assert property (X);
        assert property (X) else cnt = cnt + 1;   -> assert property (X);
    Side effect: prevents the free-input wrapper from declaring the
    action's LHS as `input logic` (which causes VCS Error-[VIPCBD])."""
    out = []
    i = 0
    while i < len(s):
        m = _ACTION_AP_OPEN_RE.search(s, i)
        if not m:
            out.append(s[i:])
            break
        out.append(s[i:m.end()])
        depth = 1
        j = m.end()
        while j < len(s) and depth > 0:
            ch = s[j]
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
            j += 1
        if depth != 0:
            out.append(s[m.end():])
            break
        out.append(s[m.end():j])  # property body incl. closing ')'
        # scan whitespace after `)`
        k = j
        while k < len(s) and s[k].isspace():
            k += 1
        if k >= len(s):
            i = k
            continue
        if s[k] == ';':
            out.append(';')
            i = k + 1
            continue
        # Non-empty action — find the next `;` and drop everything between
        semi = s.find(';', k)
        if semi < 0:
            out.append(';')
            i = len(s)
            break
        out.append(';')
        i = semi + 1
    return "".join(out)


def _strip_nested_assert_property(s: str) -> str:
    """Method3 expansion bug: SVAs of the form
        assert property (@(...) disable iff (...) ANTE |-> (assert property
            (@(...) disable iff (...) ANTE) until_with CONS));
    have a redundant inner `assert property (...)`. Detect the second and
    later `assert property (...)` blocks, balanced-paren-match their close,
    strip the inner clocking event + disable iff, and inline the body.
    """
    out_parts = []
    i = 0
    seen_first = False
    while i < len(s):
        m = _AP_OPEN_RE.search(s, i)
        if not m:
            out_parts.append(s[i:])
            break
        if not seen_first:
            seen_first = True
            out_parts.append(s[i:m.end()])
            i = m.end()
            continue
        # Inner / nested occurrence — strip
        out_parts.append(s[i:m.start()])
        depth = 1
        j = m.end()
        while j < len(s) and depth > 0:
            if s[j] == '(':
                depth += 1
            elif s[j] == ')':
                depth -= 1
            j += 1
        if depth != 0:
            out_parts.append(s[m.start():])
            break
        body = s[m.end():j - 1]
        # Drop @(clk) and disable iff from inner body
        body = _CLOCK_RE.sub("", body)
        body = _DI_RE.sub("", body)
        body = body.strip()
        out_parts.append(body)
        i = j
    return "".join(out_parts)


def normalize_sva(sva: str) -> str:
    """Apply all rules."""
    s = sva or ""
    # R11: strip in-property `//` comments first (so they don't break
    # downstream regexes).
    s = _INLINE_COMMENT_RE.sub(" ", s)
    # R1: drop backticks (treat as identifier)
    s = _BTICK_RE.sub(r"\1", s)
    # R2 + R9: flatten hierarchical paths (incl. bit-select-in-middle)
    def _flat(m):
        # join `a.b[0].c` -> `a_b_0_c` (drop brackets)
        x = m.group(0)
        x = re.sub(r"\[[^\]]*\]", "", x)
        return x.replace(".", "_")
    s = _HIER_WITH_IDX_RE.sub(_flat, s)
    # Re-apply simple flatten in case any plain dotted ids remain
    def _flat_simple(m):
        return m.group(0).replace(".", "_")
    s = _HIER_RE.sub(_flat_simple, s)
    # R3: drop package prefix
    s = _PKG_RE.sub(r"\1", s)
    # R4: drop else $action(...) (Verilator generally OK but trims noise)
    s = _ELSE_ACT_RE.sub(" ", s)
    # R5: liveness — Verilator 5.020 does NOT support `##[N:M]` ranged
    # delays or any liveness operator. Replace with fixed `##1` delay (best
    # approximation that lints cleanly).
    s = _S_EVENTUALLY_RE.sub("##1 (", s)
    s = _S_ALWAYS_RE.sub("(", s)               # s_always X -> just X
    s = _S_UNTIL_WITH_RE.sub("(", s)
    s = _S_UNTIL_RE.sub("(", s)
    s = _UNTIL_WITH_RE.sub("##1", s)
    s = _NEXTTIME_RE.sub("##1 (", s)
    # R13: collapse `##[a:b]` ranged delays to `##a` — Verilator doesn't
    # support range delays in sequence expressions.
    s = re.sub(r"##\s*\[\s*(\d+)\s*:\s*(?:\d+|\$)\s*\]", r"##\1", s)
    # R14: collapse `[*a:b]` consecutive-rep ranges to `[*a]`
    s = re.sub(r"\[\s*\*\s*(\d+)\s*:\s*(?:\d+|\$)\s*\]", r"[*\1]", s)
    # R15: drop `[=N]` and `[->N]` non-consecutive repetitions
    s = re.sub(r"\[\s*(?:=|->)\s*\d+(?:\s*:\s*(?:\d+|\$))?\s*\]", "", s)
    # R6: nested assert property bug — balanced-paren-aware deep strip.
    # Replaces the old split-rejoin which left dangling `(@... X)` fragments
    # that Verilator couldn't parse.
    s = _strip_nested_assert_property(s)
    # R17: strip pass/else action blocks (avoids VCS VIPCBD errors when the
    # free-input wrapper would declare an action's LHS as `input`).
    s = _strip_action_block(s)
    # Strip stray `strong(` / `weak(` wrappers
    s = re.sub(r"\bstrong\s*\(", "(", s)
    s = re.sub(r"\bweak\s*\(", "(", s)
    # R10: drop typed cast `IDENT'(...)` → `(...)`
    s = _CAST_RE.sub("(", s)
    # R8: wrap bare `assert property (X)` (no @clk) with @(posedge clk).
    # Balanced-paren aware (replaces previous regex that broke on nested ()).
    s = _wrap_unclocked_assertions(s)
    # R16: drop bare `until` keyword (Verilator can't lint it). Replace
    # `expr1 until expr2` with `expr1 ##1 expr2`. Imperfect but lints.
    s = re.sub(r"\buntil\b(?!\s*_with)", "##1", s, flags=re.IGNORECASE)
    # R12: balance parens — synthesis bugs in method3 expansion left some
    # SVAs with mismatched `(` count. Pad missing `)` before the trailing
    # `;`. Drop excess `)` from the end if the SVA is over-closed.
    n_open = s.count('(')
    n_close = s.count(')')
    if n_open > n_close:
        diff = n_open - n_close
        # Insert ')' before trailing ';' if present
        m_semi = re.search(r";\s*$", s)
        if m_semi:
            s = s[:m_semi.start()] + ")" * diff + s[m_semi.start():]
        else:
            s = s + ")" * diff
    elif n_close > n_open:
        diff = n_close - n_open
        # Strip diff trailing `)` characters from end (before any `;`)
        m_semi = re.search(r";\s*$", s)
        end = m_semi.start() if m_semi else len(s)
        # Walk backwards from `end` removing `)` and whitespace
        removed = 0
        cut = end
        while cut > 0 and removed < diff:
            ch = s[cut - 1]
            if ch == ')':
                removed += 1
                cut -= 1
            elif ch.isspace():
                cut -= 1
            else:
                break
        s = s[:cut] + s[end:]
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _check_one(args):
    """Try original, then normalized; record both outcomes."""
    idx, sva, ref_sva = args
    # Lazy import to avoid top-level overhead
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_g", str(ROOT / "training" / "rlvf" / "run_grpo_pilot.py"))
    g = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(g)

    orig_ok = g.verilator_compile_check(sva, ref_sva)
    norm = normalize_sva(sva)
    norm_ok = g.verilator_compile_check(norm, ref_sva) if norm else False
    return idx, orig_ok, norm_ok, norm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="",
                    help="optional jsonl: each row gets reference_sva replaced "
                         "by the normalized form when it makes Verilator pass")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = []
    with open(args.input) as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
            if args.limit and len(rows) >= args.limit:
                break
    print(f"[input] {args.input}  rows={len(rows)}")

    work = [(i, r.get("reference_sva", ""), r.get("reference_sva", ""))
            for i, r in enumerate(rows)]
    n_orig_ok = n_norm_ok = n_recovered = 0
    t0 = time.time()
    norm_results = [None] * len(rows)
    with mp.Pool(args.workers) as pool:
        for done, (idx, orig_ok, norm_ok, norm) in enumerate(
                pool.imap_unordered(_check_one, work, chunksize=8), 1):
            norm_results[idx] = (orig_ok, norm_ok, norm)
            if orig_ok: n_orig_ok += 1
            if norm_ok: n_norm_ok += 1
            if (not orig_ok) and norm_ok: n_recovered += 1
            if done % 1000 == 0 or done == len(rows):
                rate = done / (time.time() - t0)
                eta = (len(rows) - done) / max(rate, 1e-3)
                print(f"[norm] {done}/{len(rows)}  "
                      f"orig_ok={n_orig_ok} ({100*n_orig_ok/done:.1f}%)  "
                      f"norm_ok={n_norm_ok} ({100*n_norm_ok/done:.1f}%)  "
                      f"recovered={n_recovered}  "
                      f"{rate:.1f}/s  eta={eta:.0f}s")

    print(f"\n=== Summary ===")
    print(f"original verilator pass : {n_orig_ok}/{len(rows)}  ({100*n_orig_ok/len(rows):.1f}%)")
    print(f"after normalize pass    : {n_norm_ok}/{len(rows)}  ({100*n_norm_ok/len(rows):.1f}%)")
    print(f"newly recovered by norm : {n_recovered}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        n_replaced = 0
        with open(args.output, "w") as f:
            for i, r in enumerate(rows):
                orig_ok, norm_ok, norm = norm_results[i]
                new_r = dict(r)
                if (not orig_ok) and norm_ok and norm:
                    new_r["reference_sva_orig"] = r.get("reference_sva", "")
                    new_r["reference_sva"] = norm
                    new_r["_sva_normalized"] = True
                    n_replaced += 1
                f.write(json.dumps(new_r, ensure_ascii=False) + "\n")
        print(f"replaced {n_replaced} rows' reference_sva with normalized form")
        print(f"output: {args.output}")


if __name__ == "__main__":
    main()
