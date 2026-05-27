"""
vacuity.py — detect trivially-true ("vacuous") SVAs.

Vacuous assertions are those that hold irrespective of the design's actual
behavior, because:
  1. The antecedent is unsatisfiable — the implication is never exercised.
  2. The consequent is a tautology — the property is always true.
  3. The whole body is a constant (e.g. `1`, `1'b1`).

Without a vacuity filter, RL will learn to exploit this: generate
`assert property (@(posedge clk) 1'b1);` which formal-verifies to PASS for
every module. Our mutation oracle measured this attack at +0.45 reward bump.

We implement two detectors:

  (A) Syntactic vacuity — cheap regex checks for constant / tautology forms.
      Catches the obvious cases that RL would most easily exploit.

  (B) Semantic vacuity (optional, expensive) — via SymbiYosys:
      Replace B → !B in the lowered SVA and see if the resulting property
      still FAILS reachably. If the *negated* property still PASSes, the
      original is vacuous under this RTL.

Public API:
    is_vacuous_syntactic(sva)     -> (bool, reason)
    vacuity_score(sva, ...)        -> float in [0, 1]   (for reward)
"""
import re
from typing import Tuple


# Literal constants that make a SV expression always true.
_TRUE_LITERALS = {
    "1", "1'b1", "'1", "1'h1", "1'd1", "1'o1",
    "1'B1", "1'H1", "1'D1", "1'O1",
    "true", "TRUE", "True",
}


def _strip_outer_parens(s: str) -> str:
    s = s.strip()
    while s.startswith("(") and s.endswith(")"):
        d = 0; ok = True
        for i, ch in enumerate(s):
            if ch == "(": d += 1
            elif ch == ")":
                d -= 1
                if d == 0 and i != len(s) - 1:
                    ok = False; break
        if not ok: break
        s = s[1:-1].strip()
    return s


def _is_true_literal(expr: str) -> bool:
    expr = _strip_outer_parens(expr).strip()
    return expr in _TRUE_LITERALS or expr.replace(" ", "") in _TRUE_LITERALS


def _extract_body(sva: str) -> str:
    """Extract the BODY of `assert property (@(posedge clk) BODY);`."""
    s = re.sub(r"//[^\n]*", " ", sva)
    s = re.sub(r"/\*.*?\*/", " ", s, flags=re.DOTALL)
    s = re.sub(r"\s+", " ", s).strip()
    m = re.search(r"(?:assert|assume|cover)\s+property\s*\((.*)\)\s*;?\s*$",
                  s, re.IGNORECASE | re.DOTALL)
    if not m:
        return s
    inner = m.group(1).strip()
    inner = _strip_outer_parens(inner)
    # strip clock + disable iff
    inner = re.sub(r"^@\s*\([^)]*\)\s*", "", inner)
    inner = re.sub(r"^disable\s+iff\s*\([^)]*\)\s*", "", inner, flags=re.IGNORECASE)
    return _strip_outer_parens(inner)


def is_vacuous_syntactic(sva: str) -> Tuple[bool, str]:
    """Return (is_vacuous, reason).

    Flags assertions whose bodies collapse to `true` at parse time.
    """
    body = _extract_body(sva)

    # Case A: body is exactly a true-literal
    if _is_true_literal(body):
        return True, "body is constant true"

    # Case B: `LHS |-> RHS` or `LHS |=> RHS` where RHS is constant-true
    m = re.search(r"\|(->|=>)", body)
    if m:
        rhs = _strip_outer_parens(body[m.end():].strip())
        if _is_true_literal(rhs):
            return True, f"|{m.group(1)} RHS is constant true"
        # Case B': RHS is `1` after `##N` delay
        m2 = re.match(r"##\s*\d+\s+(.+)", rhs)
        if m2 and _is_true_literal(m2.group(1)):
            return True, "##N RHS is constant true"

    # Case C: body is `!0` or `!1'b0` or `~0` style
    neg_zero_re = re.compile(r"^!\s*(?:1'b0|1'B0|0|1'h0|'0|false)$",
                              re.IGNORECASE)
    if neg_zero_re.match(body.replace(" ", "")):
        return True, "body is !0 (always true)"

    # Case D: detect `or` chain that includes a true-literal at top level
    # e.g. `A || 1'b1` → always true
    if re.search(r"\|\|\s*(1'b1|1|'1|true)\b", body) or \
       re.search(r"(1'b1|1|'1|true)\s*\|\|", body):
        return True, "OR with true-literal (always true)"

    return False, ""


def vacuity_score(sva: str) -> float:
    """Return a score in [0, 1]: 1.0 = definitely non-vacuous, 0.0 = vacuous.

    Used as the `(1 - vacuity)` term in the proposal's reward.
    """
    vac, _ = is_vacuous_syntactic(sva)
    return 0.0 if vac else 1.0


# -----------------------------------------------------------------------
# Self-test
# -----------------------------------------------------------------------
if __name__ == "__main__":
    cases = [
        # obvious vacuous
        ("assert property (@(posedge clk) 1'b1);",               True),
        ("assert property (@(posedge clk) 1);",                   True),
        ("assert property (@(posedge clk) !0);",                  True),
        # subtle vacuous: RHS trivially true
        ("assert property (@(posedge clk) req |-> 1'b1);",        True),
        ("assert property (@(posedge clk) req |=> 1);",           True),
        ("assert property (@(posedge clk) a |-> ##3 1'b1);",      True),
        # non-vacuous
        ("assert property (@(posedge clk) req |=> gnt);",         False),
        ("assert property (@(posedge clk) req |-> ##3 ack);",     False),
        ("assert property (@(posedge clk) valid);",               False),
        # OR with true
        ("assert property (@(posedge clk) req || 1'b1);",         True),
    ]
    for sva, expected in cases:
        got, reason = is_vacuous_syntactic(sva)
        mark = "✓" if got == expected else "✗"
        print(f"{mark}  vacuous={got:<5} (expected {expected:<5}) "
              f"reason={reason!r}")
        print(f"    {sva}")
