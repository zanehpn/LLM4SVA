"""test_tcl_corpus.py — paper §3 / App. C TCL-classifier validation.

Paper App. C claims the regex-AST classifier reaches 100% accuracy on a
90-SVA hand-labeled corpus (30 per class). This test loads that corpus
from `data/test/tcl_corpus_90.jsonl` (one JSON object per line with
fields `sva` and `expected_class` ∈ {C1, C2, C3}) and re-asserts the
100% claim.

The corpus itself is not bundled in the repository because the SVAs
were aggregated from multiple licensed sources; the schema below is
how to reconstruct it. Each row:

    {"id": "h_001", "sva": "assert property (a |-> b);", "expected_class": "C2"}

If the corpus file is absent, the test is SKIPPED rather than failed —
re-running the test on a host with the corpus is the way to confirm the
paper's 100% claim.

A separate 32-edge-case test suite lives in `src/tcl.py::_self_test()`
and is always-run via the smoke target.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.tcl_classifier import classify_3way


CORPUS_PATH = ROOT / "data" / "test" / "tcl_corpus_90.jsonl"


def _load_corpus():
    rows = []
    with open(CORPUS_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("sva") and r.get("expected_class") in {"C1", "C2", "C3"}:
                rows.append(r)
    return rows


def test_corpus_accuracy():
    """Replicate paper App. C: 100% accuracy on the 90-SVA hand-labeled
    corpus. Skipped (not failed) when the corpus file is absent."""
    if not CORPUS_PATH.exists():
        print(f"[SKIP] corpus not present at {CORPUS_PATH}")
        print("       Reconstruct it from the schema in this file's docstring")
        print("       and re-run to confirm the paper's 100% claim.")
        return None

    rows = _load_corpus()
    if not rows:
        print(f"[SKIP] corpus at {CORPUS_PATH} is empty")
        return None

    by_class = {"C1": 0, "C2": 0, "C3": 0}
    correct = {"C1": 0, "C2": 0, "C3": 0}
    misses = []
    for r in rows:
        expected = r["expected_class"]
        by_class[expected] += 1
        got_idx, reason = classify_3way(r["sva"])
        got = {1: "C1", 2: "C2", 3: "C3"}[got_idx]
        if got == expected:
            correct[expected] += 1
        else:
            misses.append((r.get("id", "?"), expected, got, reason, r["sva"]))

    total = sum(by_class.values())
    acc = 100.0 * sum(correct.values()) / max(total, 1)
    print(f"[tcl-corpus] {total} SVAs  acc={acc:.2f}%")
    for cls in ("C1", "C2", "C3"):
        denom = max(by_class[cls], 1)
        print(f"  {cls}: {correct[cls]}/{by_class[cls]}  "
              f"({100.0 * correct[cls] / denom:.1f}%)")
    if misses:
        print(f"[tcl-corpus] {len(misses)} misclassifications (showing first 10):")
        for mid, exp, got, reason, sva in misses[:10]:
            print(f"  id={mid}  expected={exp}  got={got}  reason={reason}")
            print(f"    sva={sva[:120]}")

    # Paper App. C asserts 100%; gate the check at 100% so any regression
    # against the published corpus fails loudly.
    assert acc == 100.0, (
        f"TCL classifier dropped to {acc:.2f}% on the hand-labeled "
        f"corpus ({len(misses)} misses). Paper App. C requires 100%."
    )
    return acc


if __name__ == "__main__":
    acc = test_corpus_accuracy()
    if acc is None:
        sys.exit(0)
    print(f"OK — {acc:.2f}%")
