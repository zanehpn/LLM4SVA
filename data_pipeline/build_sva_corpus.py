#!/usr/bin/env python3
"""
build_sva_corpus.py — Validate and summarize the hand-authored SVA corpus.

This script:
  1. Loads data/sample_svas.json
  2. Validates structural integrity (50 SVAs, 10 per TCL level)
  3. Runs TCLClassifier on each SVA and compares to hand-labeled TCL
  4. Reports any mismatches
  5. Saves a validation report to results/

Usage:
    cd outputs/SVA4DAC/experiments
    python scripts/build_sva_corpus.py
"""

import json
import sys
import os
from datetime import datetime
from collections import Counter

# Add parent dir to path so we can import src/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tcl_classifier import TCLClassifier


DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sample_svas.json")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def main():
    print("=" * 60)
    print("SVA Corpus Validation")
    print("=" * 60)

    # Load corpus
    with open(DATA_PATH) as f:
        corpus = json.load(f)

    assertions = corpus["assertions"]
    print(f"\nLoaded {len(assertions)} assertions from corpus.")

    # Structural check
    level_counts = Counter(a["tcl"] for a in assertions)
    print(f"\nLevel distribution in hand labels:")
    for lv in range(1, 6):
        count = level_counts.get(lv, 0)
        status = "OK" if count == 10 else f"WARNING: expected 10, got {count}"
        print(f"  L{lv}: {count:2d} assertions  [{status}]")

    assert len(assertions) == 50, f"Expected 50 assertions, got {len(assertions)}"
    for lv in range(1, 6):
        assert level_counts[lv] == 10, f"Expected 10 L{lv} assertions, got {level_counts[lv]}"

    # Run TCL classifier on each SVA
    clf = TCLClassifier()
    mismatches = []
    per_level_results = {lv: {"correct": 0, "total": 0} for lv in range(1, 6)}

    print("\nClassifier results per assertion:")
    print(f"  {'ID':<10} {'Hand':^6} {'Pred':^6} {'Status':<8} Reason")
    print("  " + "-" * 70)

    for entry in assertions:
        aid = entry["id"]
        hand_label = entry["tcl"]
        sva = entry["sva"]

        pred_level, reason = clf.classify(sva)
        per_level_results[hand_label]["total"] += 1

        match = pred_level == hand_label
        status = "PASS" if match else "FAIL"
        if match:
            per_level_results[hand_label]["correct"] += 1
        else:
            mismatches.append({
                "id": aid,
                "hand_label": hand_label,
                "predicted": pred_level,
                "sva": sva,
                "reason": reason,
            })

        print(f"  {aid:<10} L{hand_label:^5} L{pred_level:^5} {status:<8} {reason[:50]}")

    # Summary
    total_correct = sum(v["correct"] for v in per_level_results.values())
    total = len(assertions)
    overall_acc = total_correct / total

    print("\n" + "=" * 60)
    print("Per-Level Accuracy:")
    for lv in range(1, 6):
        r = per_level_results[lv]
        acc = r["correct"] / r["total"] if r["total"] > 0 else 0
        bar = "#" * r["correct"] + "-" * (r["total"] - r["correct"])
        print(f"  L{lv}: {r['correct']}/{r['total']}  ({acc*100:.0f}%)  [{bar}]")

    print(f"\nOverall accuracy: {total_correct}/{total} = {overall_acc*100:.1f}%")

    if mismatches:
        print(f"\nMismatches ({len(mismatches)}):")
        for m in mismatches:
            print(f"  {m['id']}: hand=L{m['hand_label']}, predicted=L{m['predicted']}")
            print(f"    SVA: {m['sva'][:80]}")
            print(f"    Reason: {m['reason']}")
    else:
        print("\nNo mismatches — classifier perfectly matches hand labels!")

    # Save report
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(RESULTS_DIR, f"corpus_validation_{timestamp}.json")

    report = {
        "timestamp": timestamp,
        "total_assertions": total,
        "total_correct": total_correct,
        "overall_accuracy": overall_acc,
        "per_level": {
            f"L{lv}": {
                "correct": per_level_results[lv]["correct"],
                "total": per_level_results[lv]["total"],
                "accuracy": per_level_results[lv]["correct"] / per_level_results[lv]["total"],
            }
            for lv in range(1, 6)
        },
        "mismatches": mismatches,
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nReport saved to: {report_path}")
    print("=" * 60)

    # Exit code: 0 if accuracy >= 80%, else 1
    if overall_acc >= 0.80:
        print(f"SUCCESS: Accuracy {overall_acc*100:.1f}% >= 80% target")
        return 0
    else:
        print(f"WARNING: Accuracy {overall_acc*100:.1f}% < 80% target")
        return 1


if __name__ == "__main__":
    sys.exit(main())
