"""
TCL (Temporal Complexity Level) Classifier for SystemVerilog Assertions.

TCL Levels:
  1 — Combinational: no temporal operators; only $rose, $fell, $stable, plain logic
  2 — Single fixed delay: ##N (constant N), [*N], [=N]
  3 — Variable/multi delay: ##[a:b], [*a:b], [=a:b]
  4 — Sequence operators: |->, |=>, throughout, within, intersect, first_match
  5 — Liveness: s_eventually, s_until, s_always, until_with, strong(...)

Rule: TCL(SVA) = max level over all detected temporal operators in the SVA string.
"""

import re
from typing import Tuple

# --- Operator patterns (ordered most-specific first) ---

# Level 5: liveness operators
_L5_PATTERNS = [
    r'\bs_eventually\b',
    r'\bs_until\b',
    r'\bs_always\b',
    r'\buntil_with\b',
    r'\bstrong\s*\(',
]

# Level 4: sequence implication / complex ops
_L4_PATTERNS = [
    r'\|->',            # |->  (overlap implication)
    r'\|=>',            # |=>  (non-overlap implication)
    r'\bthroughout\b',
    r'\bwithin\b',
    r'\bintersect\b',
    r'\bfirst_match\b',
]

# Level 3: variable / ranged delays  ##[a:b]  [*a:b]  [=a:b]
_L3_PATTERNS = [
    r'##\s*\[\s*\d+\s*:\s*(?:\d+|\$)\s*\]',   # ##[a:b] or ##[a:$]
    r'\[\s*\*\s*\d+\s*:\s*(?:\d+|\$)\s*\]',    # [*a:b]
    r'\[\s*=\s*\d+\s*:\s*(?:\d+|\$)\s*\]',     # [=a:b]
    r'\[\s*-\s*>\s*\d+\s*:\s*(?:\d+|\$)\s*\]', # [->a:b] (goto repetition)
]

# Level 2: single fixed delay  ##N  [*N]  [=N]
_L2_PATTERNS = [
    r'##\s*\d+',          # ##N
    r'\[\s*\*\s*\d+\s*\]',  # [*N]
    r'\[\s*=\s*\d+\s*\]',   # [=N]
    r'\[\s*-\s*>\s*\d+\s*\]', # [->N]
]

# Level 1 markers: $rose $fell $stable $past — still combinational (no advance)
_L1_MARKERS = [
    r'\$rose\b',
    r'\$fell\b',
    r'\$stable\b',
    r'\$past\b',
    r'\$changed\b',
]

_COMPILED = {
    5: [re.compile(p, re.IGNORECASE) for p in _L5_PATTERNS],
    4: [re.compile(p, re.IGNORECASE) for p in _L4_PATTERNS],
    3: [re.compile(p, re.IGNORECASE) for p in _L3_PATTERNS],
    2: [re.compile(p, re.IGNORECASE) for p in _L2_PATTERNS],
    1: [re.compile(p, re.IGNORECASE) for p in _L1_MARKERS],
}


def classify_tcl(sva: str) -> Tuple[int, str]:
    """
    Classify a single SVA string into TCL level 1-5.

    Returns:
        (level, reason)  where level in {1,2,3,4,5}
    """
    # Strip comments (// ... and /* ... */)
    sva_clean = re.sub(r'//[^\n]*', ' ', sva)
    sva_clean = re.sub(r'/\*.*?\*/', ' ', sva_clean, flags=re.DOTALL)

    for level in [5, 4, 3, 2, 1]:
        for pat in _COMPILED[level]:
            m = pat.search(sva_clean)
            if m:
                return level, f"matched '{m.group()}' at level {level}"

    # Nothing matched — purely combinational
    return 1, "no temporal operators found (combinational)"


def classify_batch(svas: list) -> list:
    """Classify a list of SVA strings. Returns list of (level, reason)."""
    return [classify_tcl(s) for s in svas]


# ---------------------------------------------------------------------------
# Unit tests (≥20)
# ---------------------------------------------------------------------------
UNIT_TESTS = [
    # (sva_string, expected_level, description)

    # --- TCL 1: Combinational ---
    ("assert property (@(posedge clk) a && b);", 1, "L1: plain logic"),
    ("assert property (@(posedge clk) $rose(req) |-> gnt);", 4,
     "L4: $rose with |-> (|-> elevates to L4)"),
    ("assert property (@(posedge clk) $rose(req));", 1, "L1: only $rose"),
    ("assert property (@(posedge clk) $fell(ack));", 1, "L1: only $fell"),
    ("assert property (@(posedge clk) $stable(data));", 1, "L1: only $stable"),
    ("assert property (@(posedge clk) valid && !ready);", 1, "L1: no temporal ops"),
    ("assert property (@(posedge clk) a == b);", 1, "L1: pure equality"),

    # --- TCL 2: Single fixed delay ---
    ("assert property (@(posedge clk) req |-> ##1 gnt);", 4,
     "L4 wins: |-> is L4 even with ##1"),
    ("assert property (@(posedge clk) ##1 valid);", 2, "L2: ##1 only"),
    ("assert property (@(posedge clk) ##3 data_out == expected);", 2, "L2: ##3"),
    ("assert property (@(posedge clk) a [*3]);", 2, "L2: [*3]"),
    ("assert property (@(posedge clk) a [=2]);", 2, "L2: [=2]"),
    ("assert property (@(posedge clk) a [->1]);", 2, "L2: [->1]"),

    # --- TCL 3: Variable/multi delay ---
    ("assert property (@(posedge clk) ##[1:4] valid);", 3, "L3: ##[1:4]"),
    ("assert property (@(posedge clk) ##[2:$] ack);", 3, "L3: ##[2:$]"),
    ("assert property (@(posedge clk) a [*1:5]);", 3, "L3: [*1:5]"),
    ("assert property (@(posedge clk) a [=1:3]);", 3, "L3: [=1:3]"),
    ("assert property (@(posedge clk) a [->1:4]);", 3, "L3: [->1:4]"),

    # --- TCL 4: Sequence operators ---
    ("assert property (@(posedge clk) req |-> gnt);", 4, "L4: |->"),
    ("assert property (@(posedge clk) req |=> ##1 gnt);", 4, "L4: |=>"),
    ("assert property (@(posedge clk) a throughout b ##1 c);", 4, "L4: throughout"),
    ("assert property (@(posedge clk) a within b);", 4, "L4: within"),
    ("assert property (@(posedge clk) (a ##1 b) intersect (c ##1 d));", 4,
     "L4: intersect"),
    ("assert property (@(posedge clk) first_match(a ##[0:3] b));", 4,
     "L4: first_match"),

    # --- TCL 5: Liveness ---
    ("assert property (@(posedge clk) req |-> s_eventually gnt);", 5,
     "L5: s_eventually"),
    ("assert property (@(posedge clk) s_always valid);", 5, "L5: s_always"),
    ("assert property (@(posedge clk) a s_until b);", 5, "L5: s_until"),
    ("assert property (@(posedge clk) a until_with b);", 5, "L5: until_with"),
    ("assert property (@(posedge clk) strong(a ##[1:$] b));", 5, "L5: strong(...)"),

    # --- Edge cases ---
    ("// assert property (@(posedge clk) s_eventually gnt);\nassert property (valid);",
     1, "L1: s_eventually in comment only"),
    ("assert property (@(posedge clk) /* ##1 */ a && b);", 1,
     "L1: ##1 inside block comment"),
    ("assert property (@(posedge clk) ##[0:0] a);", 3, "L3: ##[0:0] is still ranged"),
]


def run_unit_tests(verbose: bool = True) -> dict:
    """Run all unit tests, return {passed, failed, total}."""
    passed = 0
    failed = 0
    failures = []

    for sva, expected_level, desc in UNIT_TESTS:
        level, reason = classify_tcl(sva)
        if level == expected_level:
            passed += 1
            if verbose:
                print(f"  PASS  [{desc}]  → L{level}")
        else:
            failed += 1
            failures.append((desc, expected_level, level, reason))
            if verbose:
                print(f"  FAIL  [{desc}]  expected L{expected_level}, got L{level} ({reason})")

    print(f"\nUnit tests: {passed}/{passed+failed} passed")
    return {"passed": passed, "failed": failed, "total": passed + failed,
            "failures": failures}


if __name__ == "__main__":
    run_unit_tests(verbose=True)
