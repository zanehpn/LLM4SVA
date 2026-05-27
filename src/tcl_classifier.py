"""
TCL Classifier — public API wrapper around src/tcl.py.

Temporal Complexity Level (TCL) classification for SystemVerilog Assertions.

TCL Levels (from TemporalSVA paper):
  Level 1: Combinational — no temporal ops; only $rose, $fell, $stable
  Level 2: Single fixed delay — ##N (constant), [*N], [=N]
  Level 3: Variable/multi delay — ##[a:b], [*a:b], [=a:b]
  Level 4: Sequence operators — |->, |=>, throughout, within, intersect, first_match
  Level 5: Liveness — s_eventually, s_until, s_always, until_with, strong(...)

Usage:
    from src.tcl_classifier import TCLClassifier

    clf = TCLClassifier()
    level, reason = clf.classify("assert property (@(posedge clk) req |-> ##[1:3] gnt);")
    print(level, reason)  # → 4, "matched '|->' at level 4"

    results = clf.classify_batch(sva_list)
    report  = clf.report(results)
"""

import sys
import os

# Ensure src/ is on path when called from scripts/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
from typing import List, Tuple, Dict, Any, Optional

# ---------------------------------------------------------------------------
# Operator pattern definitions (mirrors tcl.py but consolidated here so this
# module is self-contained and satisfies the required file name)
# ---------------------------------------------------------------------------

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
    r'\|->',              # |->
    r'\|=>',              # |=>  (was missing — original \|->>? did not match this)
    r'\bthroughout\b',
    r'\bwithin\b',
    r'\bintersect\b',
    r'\bfirst_match\b',
]

# Level 3: variable / ranged delays  ##[a:b]  [*a:b]  [=a:b]  [->a:b]
_L3_PATTERNS = [
    r'##\s*\[\s*\d+\s*:\s*(?:\d+|\$)\s*\]',    # ##[a:b] or ##[a:$]
    r'\[\s*\*\s*\d+\s*:\s*(?:\d+|\$)\s*\]',     # [*a:b]
    r'\[\s*=\s*\d+\s*:\s*(?:\d+|\$)\s*\]',      # [=a:b]
    r'\[\s*-\s*>\s*\d+\s*:\s*(?:\d+|\$)\s*\]',  # [->a:b]
]

# Level 2: single fixed delay  ##N  [*N]  [=N]  [->N]
_L2_PATTERNS = [
    r'##\s*\d+',               # ##N  (matches ##0, ##1, ##3 …)
    r'\[\s*\*\s*\d+\s*\]',    # [*N]
    r'\[\s*=\s*\d+\s*\]',     # [=N]
    r'\[\s*-\s*>\s*\d+\s*\]', # [->N]
]

# Level 1 markers (still combinational; no clock advancement)
_L1_MARKERS = [
    r'\$rose\b',
    r'\$fell\b',
    r'\$stable\b',
    r'\$past\b',
    r'\$changed\b',
]

_TCL_LEVEL_NAMES = {
    1: "Combinational",
    2: "SingleFixedDelay",
    3: "VariableDelay",
    4: "SequenceOperator",
    5: "Liveness",
}


# ---------------------------------------------------------------------------
# Three-class taxonomy (paper-facing). Coarser, robust, and unambiguous:
#   C1 Combinational : no clock-advancing operator
#   C2 TemporalDelay : any bounded temporal operator (delays, sequence, impl.)
#   C3 Liveness      : any unbounded fairness / eventuality operator
#
# 5-way -> 3-way mapping: L1->C1, L2/L3/L4->C2, L5->C3.
# ---------------------------------------------------------------------------

_C3_NAMES = {1: "Combinational", 2: "TemporalDelay", 3: "Liveness"}


def classify_3way(sva: str) -> Tuple[int, str]:
    """Three-class TCL classification (1=combinational, 2=temporal, 3=liveness).

    Robust to the 5-way classifier's edge cases (|=> matching, ##0 ambiguity)
    by collapsing all bounded temporal operators into a single class.
    """
    level, reason = classify_tcl(sva)
    c3 = 1 if level == 1 else (3 if level == 5 else 2)
    return c3, f"5-way L{level} -> 3-way C{c3} ({_C3_NAMES[c3]}); {reason}"


class TCLClassifier:
    """
    Classifies SystemVerilog Assertion strings into Temporal Complexity Levels 1-5.

    Algorithm:
        1. Strip // and /* */ comments.
        2. Scan for level-5 patterns → if any match, return 5.
        3. Scan level-4 patterns → if any match, return 4.
        4. Scan level-3 patterns → if any match, return 3.
        5. Scan level-2 patterns → if any match, return 2.
        6. Otherwise return 1 (combinational / no temporal ops).

    Note: Level-1 markers ($rose, $fell, $stable) are detected but do NOT
    override the max level — they only confirm "at least L1".
    """

    def __init__(self):
        self._compiled: Dict[int, List[re.Pattern]] = {
            5: [re.compile(p, re.IGNORECASE) for p in _L5_PATTERNS],
            4: [re.compile(p, re.IGNORECASE) for p in _L4_PATTERNS],
            3: [re.compile(p, re.IGNORECASE) for p in _L3_PATTERNS],
            2: [re.compile(p, re.IGNORECASE) for p in _L2_PATTERNS],
            1: [re.compile(p, re.IGNORECASE) for p in _L1_MARKERS],
        }

    @staticmethod
    def _strip_comments(sva: str) -> str:
        """Remove // line comments and /* block comments */ from SVA text."""
        sva = re.sub(r'//[^\n]*', ' ', sva)
        sva = re.sub(r'/\*.*?\*/', ' ', sva, flags=re.DOTALL)
        return sva

    def classify(self, sva: str) -> Tuple[int, str]:
        """
        Classify a single SVA string.

        Args:
            sva: raw SystemVerilog assertion text (may include property/assert wrappers)

        Returns:
            (level, reason) — level in {1,2,3,4,5}, reason is a human-readable string
        """
        clean = self._strip_comments(sva)

        for level in [5, 4, 3, 2]:
            for pat in self._compiled[level]:
                m = pat.search(clean)
                if m:
                    return level, f"matched '{m.group().strip()}' → level {level} ({_TCL_LEVEL_NAMES[level]})"

        # Default: combinational (level 1)
        l1_markers_found = [
            p.pattern for p in self._compiled[1] if p.search(clean)
        ]
        if l1_markers_found:
            return 1, f"combinational with markers: {l1_markers_found}"
        return 1, "no temporal operators found (pure combinational)"

    def classify_batch(self, svas: List[str]) -> List[Dict[str, Any]]:
        """
        Classify a list of SVA strings.

        Returns:
            list of dicts: {index, sva_snippet, level, level_name, reason}
        """
        results = []
        for i, sva in enumerate(svas):
            level, reason = self.classify(sva)
            results.append({
                "index": i,
                "sva_snippet": sva[:80].replace("\n", " "),
                "level": level,
                "level_name": _TCL_LEVEL_NAMES[level],
                "reason": reason,
            })
        return results

    def report(
        self,
        batch_results: List[Dict[str, Any]],
        hand_labels: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Produce a classification report.

        Args:
            batch_results:  output of classify_batch()
            hand_labels:    optional list of ground-truth TCL levels (1-5)

        Returns:
            dict with per-level counts, accuracy (if hand_labels given), and confusion
        """
        from collections import Counter

        level_counts: Counter = Counter(r["level"] for r in batch_results)
        total = len(batch_results)
        dist = {
            f"L{lv}_{_TCL_LEVEL_NAMES[lv]}": level_counts.get(lv, 0)
            for lv in range(1, 6)
        }

        report: Dict[str, Any] = {
            "total": total,
            "level_distribution": dist,
        }

        if hand_labels is not None:
            assert len(hand_labels) == total, "hand_labels length must match batch_results"
            predicted = [r["level"] for r in batch_results]
            correct = sum(p == g for p, g in zip(predicted, hand_labels))
            overall_acc = correct / total if total > 0 else 0.0
            report["overall_accuracy"] = overall_acc
            report["correct"] = correct
            report["incorrect"] = total - correct

            # Per-level accuracy
            per_level_acc: Dict[str, Any] = {}
            for lv in range(1, 6):
                idxs = [i for i, g in enumerate(hand_labels) if g == lv]
                if not idxs:
                    continue
                lv_correct = sum(1 for i in idxs if predicted[i] == lv)
                per_level_acc[f"L{lv}"] = {
                    "count": len(idxs),
                    "correct": lv_correct,
                    "accuracy": lv_correct / len(idxs),
                }
            report["per_level_accuracy"] = per_level_acc

            # Confusion list (wrong predictions)
            confusion = [
                {
                    "index": batch_results[i]["index"],
                    "sva_snippet": batch_results[i]["sva_snippet"],
                    "expected": hand_labels[i],
                    "predicted": predicted[i],
                    "reason": batch_results[i]["reason"],
                }
                for i in range(total)
                if predicted[i] != hand_labels[i]
            ]
            report["confusion"] = confusion

        return report


# ---------------------------------------------------------------------------
# Convenience top-level functions (backwards-compatible with tcl.py)
# ---------------------------------------------------------------------------

_DEFAULT_CLASSIFIER = None


def classify_tcl(sva: str) -> Tuple[int, str]:
    """Module-level convenience: classify a single SVA string."""
    global _DEFAULT_CLASSIFIER
    if _DEFAULT_CLASSIFIER is None:
        _DEFAULT_CLASSIFIER = TCLClassifier()
    return _DEFAULT_CLASSIFIER.classify(sva)


def classify_batch(svas: List[str]) -> List[Tuple[int, str]]:
    """Module-level convenience: classify a list of SVA strings."""
    clf = TCLClassifier()
    return [clf.classify(s) for s in svas]


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

def _self_test():
    clf = TCLClassifier()
    tests = [
        ("assert property (@(posedge clk) a && b);", 1),
        ("assert property (@(posedge clk) $rose(req));", 1),
        ("assert property (@(posedge clk) ##3 data_out == expected);", 2),
        ("assert property (@(posedge clk) a [*3]);", 2),
        ("assert property (@(posedge clk) ##[1:4] valid);", 3),
        ("assert property (@(posedge clk) a [*1:5]);", 3),
        ("assert property (@(posedge clk) req |-> gnt);", 4),
        ("assert property (@(posedge clk) a throughout b ##1 c);", 4),
        ("assert property (@(posedge clk) req |-> s_eventually gnt);", 5),
        ("assert property (@(posedge clk) s_always valid);", 5),
        # Comment stripping
        ("// s_eventually\nassert property (@(posedge clk) valid);", 1),
        ("assert property (@(posedge clk) /* ##1 */ a);", 1),
    ]
    passed = failed = 0
    for sva, expected in tests:
        level, reason = clf.classify(sva)
        status = "PASS" if level == expected else "FAIL"
        if level != expected:
            failed += 1
            print(f"  {status}  expected L{expected} got L{level}: {sva[:60]}")
        else:
            passed += 1
            print(f"  {status}  L{level}: {sva[:60]}")

    print(f"\nSelf-test: {passed}/{passed+failed} passed")
    return failed == 0


if __name__ == "__main__":
    ok = _self_test()
    sys.exit(0 if ok else 1)
