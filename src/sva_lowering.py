"""
sva_lowering.py — translate a SystemVerilog concurrent assertion into an
immediate-form equivalent that yosys-slang can parse without
`--ignore-assertions`, so the lowered SVA can be formally verified by
SymbiYosys.

Supported patterns (2026-04-20):
  1. `A`                                         pure combinational
  2. `A |-> B`                                   same-cycle implication
  3. `A |=> B`                                   next-cycle implication
  4. `A |-> ##N B`                               fixed delay
  5. `A |-> ##[a:b] B`                           ranged delay (bounded)
  6. `$rose(x)`, `$fell(x)`, `$stable(x)`        edge helpers
  7. `disable iff (...)`                         lifted to antecedent gate
  8. `not (A)`                                   negation around whole prop

Bounded-liveness rewrite (opt-in via lower_sva's `liveness_bound` arg):
  - `s_eventually P`  →  `##[0:N] P`    (bounded existence)
  - `nexttime P`      →  `##1 P`        (exact)
  - `s_always P`      →  `P`            (BMC depth N covers the quantifier)

NOT yet supported (returns None → caller falls back to syntax-only reward):
  - Liveness still untouched: `s_until`, `until_with`, `strong(...)`
  - Sequence composites: `throughout`, `within`, `intersect`, `first_match`
  - Goto / consecutive repeat: `[*a:b]`, `[=a:b]`, `[->a:b]`
  - Parameterized delays: `##[0:$]`, `##[$]`

Output is a dict with:
  {
    "ok":        bool,
    "pattern":   "combinational" | "implies_same" | "implies_next" | ...
    "tcl":       int (1..5)
    "lowered":   str — a SV fragment declaring any helper `reg`s, delay-chain
                 always_ffs, plus the immediate assert. Inject at the end of
                 a module, before `endmodule`.
    "notes":    list of str
  }
"""
import re
from typing import Dict, Optional


# -----------------------------------------------------------------------
# Tokenize / strip comments + whitespace
# -----------------------------------------------------------------------
def _clean(sva: str) -> str:
    """Drop comments + collapse whitespace."""
    sva = re.sub(r"//[^\n]*", " ", sva)
    sva = re.sub(r"/\*.*?\*/", " ", sva, flags=re.DOTALL)
    sva = re.sub(r"\s+", " ", sva).strip()
    return sva


# -----------------------------------------------------------------------
# Extract the property body from a full `assert property (@(posedge clk) BODY);`
# -----------------------------------------------------------------------
_ENTRY_RE = re.compile(
    r"(?:[a-zA-Z_]\w*\s*:\s*)?"                          # optional label
    r"(assert|assume|cover)\s+property\s*\((.*)\)\s*;?", re.DOTALL | re.IGNORECASE,
)
_CLK_RE = re.compile(
    r"^\s*@\s*\(\s*(?:posedge|negedge)\s+([A-Za-z_]\w*)[^)]*\)\s*",
    re.IGNORECASE,
)
_DISABLE_HEAD_RE = re.compile(r"^\s*disable\s+iff\s*\(", re.IGNORECASE)


def _consume_paren_balanced(s: str, start: int) -> Optional[int]:
    """Given that `s[start]` is '(', return the index just after the matching
    ')'. Return None if unbalanced."""
    if start >= len(s) or s[start] != "(":
        return None
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "(":
            depth += 1
        elif s[i] == ")":
            depth -= 1
            if depth == 0:
                return i + 1
    return None


def _strip_disable_iff(inner: str) -> tuple:
    """If `inner` starts with `disable iff (...)`, return (disable_expr, rest).
    Otherwise return (None, inner). Paren-balanced — handles `$sampled(x)` etc."""
    m = _DISABLE_HEAD_RE.match(inner)
    if not m:
        return None, inner
    open_at = m.end() - 1                     # position of the '('
    end = _consume_paren_balanced(inner, open_at)
    if end is None:
        return None, inner
    disable_expr = inner[open_at + 1:end - 1].strip()
    rest = inner[end:].lstrip()
    return disable_expr, rest


def _rewrite_sysfunc(s: str, fname: str, transform) -> str:
    """Find `$<fname>(...)` calls (paren-balanced) and replace with
    `transform(arg)`. Used to translate SV system functions that yosys-slang
    doesn't accept into pure-Boolean equivalents."""
    out = []
    i = 0
    pat = re.compile(r"\$" + re.escape(fname) + r"\s*\(")
    while i < len(s):
        m = pat.search(s, i)
        if not m:
            out.append(s[i:])
            break
        out.append(s[i:m.start()])
        open_at = m.end() - 1
        depth = 1
        j = open_at + 1
        while j < len(s) and depth > 0:
            if s[j] == "(":
                depth += 1
            elif s[j] == ")":
                depth -= 1
            j += 1
        if depth != 0:
            # Unbalanced — bail out, leave string as-is from this point
            out.append(s[m.start():])
            break
        arg = s[open_at + 1: j - 1].strip()
        out.append(transform(arg))
        i = j
    return "".join(out)


def _rewrite_unsupported_sysfuncs(s: str) -> str:
    """Translate yosys-slang-unsupported SV system functions into Boolean
    equivalents that BMC can reason about.

      `$onehot0(x)`  → `((x & (x - 1)) == 0)`        (zero or one bit set)
      `$onehot(x)`   → `((x != 0) && ((x & (x - 1)) == 0))`   (exactly one bit)
    """
    s = _rewrite_sysfunc(
        s, "onehot0",
        lambda a: f"(({a}) & (({a}) - 1)) == 0",
    )
    s = _rewrite_sysfunc(
        s, "onehot",
        lambda a: f"(({a}) != 0) && ((({a}) & (({a}) - 1)) == 0)",
    )
    return s


def _extract_body(sva: str) -> Optional[Dict]:
    """Parse an `assert property (@(posedge clk) disable iff (rst) BODY);` string.
    Return {verb, clk, disable, body} or None."""
    s = _clean(sva)
    m = _ENTRY_RE.search(s)
    if not m:
        return None
    verb, inner = m.group(1).lower(), m.group(2).strip()
    # strip outer matched parens: inner may be `(A |-> B)` or `A |-> B`
    while inner.startswith("(") and _paren_balanced(inner):
        inner = inner[1:-1].strip()
    # pull clock
    clk_m = _CLK_RE.match(inner)
    clk = clk_m.group(1) if clk_m else "clk"
    if clk_m:
        inner = inner[clk_m.end():].strip()
    # pull disable iff (paren-balanced — supports `disable iff ($sampled(x))` etc.)
    disable, inner = _strip_disable_iff(inner)
    return {"verb": verb, "clk": clk, "disable": disable, "body": inner}


def _paren_balanced(s: str) -> bool:
    """True iff s begins with '(' and the matching ')' is the last char."""
    if not s.startswith("(") or not s.endswith(")"):
        return False
    depth = 0
    for i, ch in enumerate(s):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i == len(s) - 1
    return False


# -----------------------------------------------------------------------
# Pattern matchers over the body
# -----------------------------------------------------------------------
# Implication  (LHS) |-> (RHS)  or  (LHS) |=> (RHS)
_IMP_SPLIT_RE = re.compile(r"\|(->|=>)")


def _top_split(body: str, op_re: re.Pattern) -> Optional[tuple]:
    """Split `body` at top-level matches of op_re.
    Return (lhs, op, rhs) or None if no top-level match."""
    depth_p = depth_b = depth_c = 0
    i = 0
    while i < len(body):
        ch = body[i]
        if ch == "(": depth_p += 1
        elif ch == ")": depth_p -= 1
        elif ch == "[": depth_b += 1
        elif ch == "]": depth_b -= 1
        elif ch == "{": depth_c += 1
        elif ch == "}": depth_c -= 1
        if depth_p == depth_b == depth_c == 0:
            m = op_re.match(body, i)
            if m:
                lhs = body[:i].strip()
                op = m.group(0)
                rhs = body[m.end():].strip()
                return lhs, op, rhs
        i += 1
    return None


def _strip_outer_parens(s: str) -> str:
    while _paren_balanced(s):
        s = s[1:-1].strip()
    return s


# ##N  or  ##[a:b]
_DELAY_PREFIX_RE = re.compile(
    r"##\s*(?:\[\s*(\d+)\s*:\s*(\d+|\$)\s*\]|(\d+))"
)


def _parse_delay_prefix(rhs: str) -> Optional[tuple]:
    """If RHS starts with `##N X` or `##[a:b] X`, return (lo, hi, X_rest)."""
    m = _DELAY_PREFIX_RE.match(rhs)
    if not m:
        return None
    if m.group(3) is not None:   # ##N
        lo = hi = int(m.group(3))
    else:                         # ##[a:b]
        lo = int(m.group(1))
        if m.group(2) == "$":
            return None           # unbounded — not supported yet
        hi = int(m.group(2))
    tail = rhs[m.end():].strip()
    return lo, hi, tail


# Top-level scan for `##N` / `##[a:b]` sequence operators. Returns list of
# (delay_lo, delay_hi, term_str) such that the original sequence is equivalent
# to "term[0] ##d1 term[1] ##d2 term[2] ..." with the first term carrying
# implicit delay 0. Used to decompose sequence-form LHS like
#   `(a == 0 & b == 1) ##1 (c == 0)`
# into delay-shifted Boolean terms whose AND is the antecedent match.
_DELAY_OP_RE = re.compile(
    r"##\s*(?:\[\s*(\d+)\s*:\s*(\d+)\s*\]|(\d+))"
)


def _split_sequence(seq: str) -> Optional[list]:
    """Split a sequence expression at top-level `##N` / `##[a:b]` boundaries.

    Returns list of (lo, hi, term_str) where the first tuple has lo==hi==0
    (the leading boolean has no preceding delay). Returns None if the body
    contains constructs we can't lower as a flat sequence (unbounded ranges,
    `[*..]`, `throughout`, `within`, `intersect`, etc.).
    """
    # Hard-fail on operators we cannot lower as a flat AND-chain
    for kw in ("throughout", "within", "intersect", "first_match"):
        if re.search(r"\b" + kw + r"\b", seq):
            return None
    if re.search(r"\[\*|\[=|\[->", seq):
        return None
    if re.search(r"##\s*\[\s*\d+\s*:\s*\$\s*\]|##\s*\$", seq):
        return None  # unbounded delay

    parts = []                     # list of (lo, hi, term)
    pending_lo = pending_hi = 0    # delay before the *next* term
    cursor = 0
    depth_p = depth_b = depth_c = 0
    last_emit = 0
    i = 0
    while i < len(seq):
        ch = seq[i]
        if ch == "(": depth_p += 1
        elif ch == ")": depth_p -= 1
        elif ch == "[": depth_b += 1
        elif ch == "]": depth_b -= 1
        elif ch == "{": depth_c += 1
        elif ch == "}": depth_c -= 1
        if depth_p == depth_b == depth_c == 0:
            m = _DELAY_OP_RE.match(seq, i)
            if m:
                term = seq[last_emit:i].strip()
                parts.append((pending_lo, pending_hi, term))
                if m.group(3) is not None:
                    pending_lo = pending_hi = int(m.group(3))
                else:
                    pending_lo = int(m.group(1))
                    pending_hi = int(m.group(2))
                i = m.end()
                last_emit = i
                continue
        i += 1
    tail = seq[last_emit:].strip()
    parts.append((pending_lo, pending_hi, tail))

    # Reject empty terms (e.g. leading `##1 a`) — caller should not call us on those.
    if any(not t for _, _, t in parts):
        return None
    return parts


# -----------------------------------------------------------------------
# Helper-register builders
# -----------------------------------------------------------------------
def _delay_reg_chain(sig_expr: str, depth: int, clk: str, prefix: str):
    """Emit SV code declaring reg prefix_d1..prefix_d{depth} that shifts
    `sig_expr` forward one cycle per register. Returns (decl_lines,
    ff_lines, last_name)."""
    decls, ffs = [], []
    prev = None
    for i in range(1, depth + 1):
        name = f"{prefix}_d{i}"
        decls.append(f"logic {name};")
        src = sig_expr if i == 1 else f"{prefix}_d{i - 1}"
        ffs.append(f"    {name} <= {src};")
    if depth > 0:
        last = f"{prefix}_d{depth}"
    else:
        last = sig_expr
    return decls, ffs, last


def _wrap_always_ff(clk: str, ff_body_lines):
    return (
        f"always_ff @(posedge {clk}) begin\n"
        + "\n".join(ff_body_lines)
        + "\nend"
    )


def _emit_immediate_assert(
    clk: str, disable: Optional[str], cond_expr: str, verb: str = "assert",
):
    """Produce an immediate `verb` (assert / assume / cover) guarded by a
    manually-registered init flag.

    yosys-slang does NOT support $initstate, so we synthesize our own:

        logic svlow_inited = 1'b0;
        always_ff @(posedge clk) svlow_inited <= 1'b1;
        always_ff @(posedge clk)
            if (svlow_inited [&& !disable])
                <verb>(cond);
    """
    guard = "svlow_inited"
    if disable:
        guard += f" && !({disable})"
    return (
        "logic svlow_inited = 1'b0;\n"
        f"always_ff @(posedge {clk}) svlow_inited <= 1'b1;\n"
        f"always_ff @(posedge {clk}) begin\n"
        f"    if ({guard}) {verb} ({cond_expr});\n"
        f"end"
    )


# -----------------------------------------------------------------------
# Public entry points
# -----------------------------------------------------------------------
_S_EVENTUALLY_RE = re.compile(r"\bs_eventually\b")
_NEXTTIME_RE = re.compile(r"\bnexttime\b")
_S_ALWAYS_RE = re.compile(r"\bs_always\b\s*")
# Unbounded delay range `##[n:$]` — real C3 references from engineer-written
# SVA often use this explicit BMC-style "eventually" idiom, usually wrapped
# in `strong(...)`. Substitute `$` with the bound.
_UNBOUNDED_RANGE_RE = re.compile(r"##\s*\[\s*(\d+)\s*:\s*\$\s*\]")
# `strong(seq)` makes a sequence obligation strict. Under bounded BMC the
# sequence must complete within the trace horizon anyway, so the wrapper
# can be dropped symmetrically on both sides.
_STRONG_RE = re.compile(r"\bstrong\s*\(")


def _strip_strong_wrapper(body: str) -> tuple:
    """Remove `strong(…)` wrappers by matching balanced parens. Returns
    (new_body, count)."""
    result = []
    i = 0
    count = 0
    while i < len(body):
        m = _STRONG_RE.match(body, i)
        if not m:
            result.append(body[i])
            i += 1
            continue
        # Found `strong(` — walk paren-balanced to find matching `)`.
        count += 1
        depth = 1
        j = m.end()  # position just after `(`
        while j < len(body) and depth > 0:
            if body[j] == '(':
                depth += 1
            elif body[j] == ')':
                depth -= 1
            j += 1
        # body[m.end():j-1] is the inner sequence; drop the wrapper.
        result.append(body[m.end():j - 1])
        i = j
    return "".join(result), count


def _rewrite_bounded_liveness(body: str, bound: int) -> tuple:
    """Rewrite the supported liveness operators into bounded BMC
    equivalents. Returns (new_body, notes).

    Both sides of a PEC check must call this with the *same* bound so the
    resulting verdict is meaningful — asymmetric rewriting would bias the
    directional implications.

    Rewrites:
      - `s_eventually P`  → `##[0:N] P`
      - `nexttime P`      → `##1 P`
      - `s_always P`      → `P` (BMC at every cycle covers the quantifier)
      - `##[n:$] P`       → `##[n:N] P` (bound the infinite horizon)
      - `strong(seq)`     → `seq` (bounded BMC is strong by construction
                                   within its depth)
    """
    notes = []
    new_body = body
    if _S_EVENTUALLY_RE.search(new_body):
        new_body = _S_EVENTUALLY_RE.sub(f"##[0:{bound}]", new_body)
        notes.append(f"rewrote s_eventually -> ##[0:{bound}]")
    if _NEXTTIME_RE.search(new_body):
        new_body = _NEXTTIME_RE.sub("##1", new_body)
        notes.append("rewrote nexttime -> ##1")
    if _S_ALWAYS_RE.search(new_body):
        new_body = _S_ALWAYS_RE.sub("", new_body)
        notes.append("rewrote s_always -> (stripped; BMC depth covers)")
    if _UNBOUNDED_RANGE_RE.search(new_body):
        new_body = _UNBOUNDED_RANGE_RE.sub(
            lambda m: f"##[{m.group(1)}:{bound}]", new_body)
        notes.append(f"rewrote ##[n:$] -> ##[n:{bound}]")
    if _STRONG_RE.search(new_body):
        new_body, n = _strip_strong_wrapper(new_body)
        if n:
            notes.append(f"stripped {n} strong(...) wrapper(s)")
    return new_body, notes


def lower_sva(sva: str, helper_prefix: str = "svlow",
              liveness_bound: Optional[int] = None) -> Dict:
    """
    Translate a concurrent SVA to immediate form + helper state.

    `liveness_bound`: when not None, rewrites `s_eventually` / `nexttime` /
    `s_always` to bounded BMC equivalents before the unsupported-operator
    check. Must be applied symmetrically to both sides of a PEC comparison.

    Returns dict with keys listed in module docstring. On unsupported pattern
    returns {"ok": False, "pattern": None, ...}.
    """
    parsed = _extract_body(sva)
    if not parsed:
        return {"ok": False, "pattern": None, "tcl": 0, "lowered": "",
                "notes": ["failed to parse assert property (…)"]}

    verb, clk, disable, body = parsed["verb"], parsed["clk"], parsed["disable"], parsed["body"]
    # Rewrite SV system functions yosys-slang doesn't handle (e.g. $onehot0)
    # into Boolean equivalents BEFORE pattern matching, so they survive into
    # the lowered fragment intact.
    body = _rewrite_unsupported_sysfuncs(body)
    if disable:
        disable = _rewrite_unsupported_sysfuncs(disable)
    body = _strip_outer_parens(body)

    # Bounded-liveness rewrite (opt-in). Must run BEFORE the fail-fast
    # UNSUPPORTED check — after the rewrite, the three rewritten operators
    # are gone and the check will not trip on them, but `s_until`,
    # `throughout`, etc. remain and are still rejected.
    liveness_notes: list = []
    if liveness_bound is not None and liveness_bound > 0:
        body, liveness_notes = _rewrite_bounded_liveness(body, liveness_bound)
        # A bare `##[0:N] X` produced by rewriting `s_eventually X` has no
        # top-level implication — the existing "bare sequence outside
        # implication" reject at the bottom of this function would trip.
        # In SVA `assert property P` is semantically equivalent to
        # `assert property (1'b1 |-> P)` since the property is evaluated
        # at every sampling event; wrap the bare sequence so it lands on
        # the supported `A |-> ##[a:b] B` pattern.
        if liveness_notes and body.lstrip().startswith("##"):
            body = f"1'b1 |-> ({body})"
            liveness_notes.append("wrapped bare sequence with 1'b1 |-> antecedent")

    # Liveness / unsupported — fail fast
    for kw in ("s_eventually", "s_until", "s_always", "until_with", "nexttime",
               "throughout", "within", "intersect", "first_match"):
        if re.search(r"\b" + kw + r"\b", body):
            return {"ok": False, "pattern": None, "tcl": 5,
                    "lowered": "",
                    "notes": [f"unsupported operator: {kw}"]}
    if "strong(" in body.replace(" ", ""):
        return {"ok": False, "pattern": None, "tcl": 5, "lowered": "",
                "notes": ["unsupported operator: strong"]}
    if re.search(r"\[\*|\[=|\[->", body):
        return {"ok": False, "pattern": None, "tcl": 5, "lowered": "",
                "notes": ["unsupported: repeat/goto operator"]}

    # (A |-> B) or (A |=> B), possibly with `##N` / `##[a:b]` between
    split = _top_split(body, _IMP_SPLIT_RE)
    if split:
        lhs, op, rhs = split
        lhs = _strip_outer_parens(lhs)
        rhs = _strip_outer_parens(rhs)

        # ---- LHS sequence handling ----
        # If LHS contains top-level `##N` (e.g. "(a) ##1 (b) ##2 (c)"), it is a
        # sequence whose match means: "a was true T_total cycles ago AND b was
        # true T_remaining cycles ago AND ... AND last term is true now."
        # We lower this to a delay-chain per term, ANDed together.
        lhs_seq = _split_sequence(lhs) if "##" in lhs else None
        antecedent_match: Optional[str] = None
        seq_decl: list = []
        seq_ff: list = []
        if lhs_seq is not None and len(lhs_seq) > 1:
            # Reject ranged inter-term delays — would need OR-window logic.
            if any(lo != hi for lo, hi, _ in lhs_seq):
                return {"ok": False, "pattern": None, "tcl": 4,
                        "lowered": "",
                        "notes": ["unsupported: ranged ##[a:b] inside LHS sequence"]}
            # Compute cumulative delay from the END (last term has delay 0,
            # second-last delayed by its successor's delay, etc.)
            cum = 0
            term_exprs = []
            for k in range(len(lhs_seq) - 1, -1, -1):
                _, _, term = lhs_seq[k]
                # The first tuple's delay (lo,hi) refers to delay BEFORE that
                # term, which for k==0 is unused (always 0). For k>=1 the
                # delay-tuple-[k] is the gap between term k-1 and term k.
                # Build delay regs for `(term)` shifted by `cum` cycles.
                if cum == 0:
                    term_exprs.append(f"({term})")
                else:
                    decls_k, ffs_k, last_k = _delay_reg_chain(
                        f"({term})", cum, clk,
                        f"{helper_prefix}_seq{k}")
                    seq_decl.extend(decls_k); seq_ff.extend(ffs_k)
                    term_exprs.append(last_k)
                # delay before THIS term (inherited from lhs_seq[k][:2])
                # accumulates for the term to its left
                if k > 0:
                    cum += lhs_seq[k][0]   # lo == hi at this point
            # term_exprs is back-to-front; reverse for readability
            term_exprs.reverse()
            antecedent_match = "(" + " && ".join(term_exprs) + ")"
        elif lhs_seq is None and "##" in lhs:
            # LHS has `##` but we couldn't decompose it cleanly — fail-safe.
            return {"ok": False, "pattern": None, "tcl": 4,
                    "lowered": "",
                    "notes": ["unsupported: complex sequence in LHS"]}

        # optional ##N / ##[a:b] on the RHS
        dpref = _parse_delay_prefix(rhs)
        extra_delay = 0
        use_range_lo = use_range_hi = None
        if dpref:
            lo, hi, rhs_tail = dpref
            if lo == hi:
                extra_delay = lo
            else:
                use_range_lo, use_range_hi = lo, hi
            rhs = _strip_outer_parens(rhs_tail)

        # Reject sequence-form RHS we can't represent as a single Boolean
        if "##" in rhs:
            return {"ok": False, "pattern": None, "tcl": 4,
                    "lowered": "",
                    "notes": ["unsupported: sequence in RHS after delay strip"]}

        # Total antecedent delay applied to whole sequence-match:
        #   |->   means 0 cycles
        #   |=>   means 1 cycle
        #   plus extra_delay from ##N (between LHS-match and RHS check)
        base_delay = 0 if op == "|->" else 1
        total_delay = base_delay + extra_delay

        decl = list(seq_decl)
        ff = list(seq_ff)
        # Build delay chain on (boolean LHS or pre-built sequence-match expr)
        ant_src = antecedent_match if antecedent_match is not None else f"({lhs})"
        decls, ffs, last_lhs = _delay_reg_chain(
            ant_src, total_delay, clk, f"{helper_prefix}_a")
        decl.extend(decls); ff.extend(ffs)

        if use_range_lo is None:
            # Fixed-cycle obligation: if lhs fired total_delay ago, rhs must hold now
            cond = f"!({last_lhs}) || ({rhs})"
            if antecedent_match is not None:
                pattern = "seq_implies_next" if op == "|=>" else "seq_implies_same"
                tcl = 4
            else:
                pattern = "implies_next" if op == "|=>" else \
                          ("implies_same" if extra_delay == 0 else "implies_fixed_delay")
                tcl = 4 if extra_delay == 0 else 2
        else:
            # Ranged ##[lo:hi]: rhs must hold at least once within cycles lo..hi after lhs.
            # We slide a window: keep register for lhs at (base_delay+lo) through (base_delay+hi).
            # If lhs fired (base_delay+hi) ago and rhs was never seen in any of the
            # matching cycles, the assertion should have already failed; so here we just
            # build the OR of (rhs @ cycle k) for k in [lo, hi], computed via its own
            # delay-chain on rhs (forward-looking is not possible; we BMC the window).
            # For BMC correctness we structure it as:
            #   at cycle t: assert (!(lhs_{t-(base_delay+hi)}) || any rhs in last hi-lo+1 cycles)
            hi_delay = base_delay + use_range_hi
            # lhs shifted by hi_delay:
            decls2, ffs2, last_lhs_hi = _delay_reg_chain(
                f"({lhs})", hi_delay, clk, f"{helper_prefix}_ar")
            decl, ff = decls2, ffs2   # supersede fixed-delay regs
            # Now build a sliding register of rhs: rhs_d0..rhs_d{hi-lo}
            window = use_range_hi - use_range_lo
            rhs_expr_list = [f"({rhs})"]
            for k in range(1, window + 1):
                name = f"{helper_prefix}_rw_d{k}"
                decl.append(f"logic {name};")
                src = f"({rhs})" if k == 1 else f"{helper_prefix}_rw_d{k - 1}"
                ff.append(f"    {name} <= {src};")
                rhs_expr_list.append(name)
            or_rhs = " || ".join(rhs_expr_list)
            cond = f"!({last_lhs_hi}) || ({or_rhs})"
            pattern = "implies_ranged_delay"
            tcl = 3

        # `verb` is "assert" / "assume" / "cover" — preserve original semantics
        body_chunks = []
        if decl:
            body_chunks.append("    " + "\n    ".join(decl))
        if ff:
            body_chunks.append(
                "    " + _wrap_always_ff(clk, ff).replace("\n", "\n    "))
        body_chunks.append(
            "    " + _emit_immediate_assert(
                clk, disable, cond, verb=verb).replace("\n", "\n    "))
        lowered = "\n".join(body_chunks)
        return {"ok": True, "pattern": pattern, "tcl": tcl,
                "lowered": lowered, "notes": []}

    # No |-> / |=>: combinational body, just wrap as immediate <verb>
    if "##" in body:
        return {"ok": False, "pattern": None, "tcl": 2, "lowered": "",
                "notes": ["unsupported: bare sequence outside implication"]}
    return {"ok": True, "pattern": "combinational", "tcl": 1,
            "lowered": "    " + _emit_immediate_assert(
                clk, disable, body, verb=verb).replace("\n", "\n    "),
            "notes": []}


# -----------------------------------------------------------------------
# Inject lowered SVA into an existing RTL module — strip any prior
# concurrent-form asserts (keep functional logic intact) and append the new
# immediate form at the end, before `endmodule`.
# -----------------------------------------------------------------------
_CONCURRENT_ASSERT_RE = re.compile(
    r"(?:[A-Za-z_]\w*\s*:\s*)?(?:assert|assume|cover)\s+property\s*\("
    r"[^;]*\)\s*;\s*",
    re.DOTALL,
)


def strip_concurrent_asserts(rtl: str) -> str:
    """Remove every `assert/assume/cover property (...)` statement from RTL."""
    # Simple regex is not paren-aware; we do a balanced-paren sweep instead.
    out = []
    i = 0
    while i < len(rtl):
        m = re.search(r"(?:[A-Za-z_]\w*\s*:\s*)?(?:assert|assume|cover)\s+property\s*\(",
                      rtl[i:])
        if not m:
            out.append(rtl[i:])
            break
        out.append(rtl[i:i + m.start()])
        open_at = i + m.end() - 1
        depth = 1
        j = open_at + 1
        while j < len(rtl) and depth > 0:
            if rtl[j] == "(": depth += 1
            elif rtl[j] == ")": depth -= 1
            j += 1
        # swallow trailing `;`
        while j < len(rtl) and rtl[j] != ";":
            j += 1
        i = j + 1   # skip past ';'
    return "".join(out)


_CLK_ALIASES = ["clk_i", "clock", "aclk", "S_AXI_ACLK", "core_clk", "clk_core"]


def _detect_module_clock(rtl: str) -> Optional[str]:
    """Find the first plausible clock signal declared in the module header.

    Returns the signal name or None.  Priority: exact `clk` → `clk_i` →
    any `input ... c(l)?k.*` token.
    """
    # Look only inside module header (first `(...)` block)
    m = re.search(r"\bmodule\s+[A-Za-z_]\w*\s*(?:#\s*\([^)]*\))?\s*\((.*?)\)",
                  rtl, re.DOTALL)
    header = m.group(1) if m else rtl[:2000]
    # Collect all port names
    for prio in ["clk", "clk_i", "clock", "aclk", "S_AXI_ACLK",
                 "CLK", "Clk", "clk_core"]:
        if re.search(rf"\b{re.escape(prio)}\b", header):
            return prio
    # Fallback: any identifier containing "clk" or "clock"
    cand = re.findall(r"\b([A-Za-z_]\w*)\b", header)
    for c in cand:
        if re.search(r"c(l)?k|clock", c, re.IGNORECASE):
            return c
    return None


def inject_lowered(
    rtl: str, lowered: str,
    clk_in_sva: str = "clk",
) -> Optional[str]:
    """Replace all concurrent asserts in `rtl` with `lowered`.

    If the module has no signal named `clk_in_sva` but does have another
    clock-like signal, insert a `wire <clk_in_sva> = <module_clk>;` alias
    so the lowered fragment can reference `clk_in_sva`.

    Return None if no `endmodule` found.
    """
    stripped = strip_concurrent_asserts(rtl)
    end_idx = stripped.rfind("endmodule")
    if end_idx < 0:
        return None

    alias_line = ""
    if not re.search(rf"\b{re.escape(clk_in_sva)}\b", stripped[:end_idx]):
        module_clk = _detect_module_clock(stripped)
        if module_clk and module_clk != clk_in_sva:
            alias_line = f"    wire {clk_in_sva} = {module_clk};\n"
        # else: neither available; leave blank and let compile error happen

    return (stripped[:end_idx].rstrip()
            + "\n\n" + alias_line + lowered + "\n"
            + stripped[end_idx:])


# -----------------------------------------------------------------------
# Self-tests
# -----------------------------------------------------------------------
def _demo():
    tests = [
        "assert property (@(posedge clk) req |=> gnt);",
        "assert property (@(posedge clk) req |-> gnt);",
        "assert property (@(posedge clk) disable iff (!rst_n) req |-> ##3 ack);",
        "assert property (@(posedge clk) req |-> ##[1:3] ack);",
        "assert property (@(posedge clk) valid);",
        "a1: assert property (@(posedge clk) s_eventually done);",
        "assert property (@(posedge clk) $rose(req) |-> $stable(x));",
    ]
    for sva in tests:
        r = lower_sva(sva)
        print(f"\nSVA:     {sva}")
        print(f"  ok={r['ok']}  pattern={r['pattern']}  tcl={r['tcl']}")
        if r["ok"]:
            print("  LOWERED:")
            print(r["lowered"])
        else:
            print("  notes:", r["notes"])


if __name__ == "__main__":
    _demo()
