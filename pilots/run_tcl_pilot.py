"""
Run TCL classifier on the dataset and report per-level distribution.

Input:  data/sva_examples.json
Output: results/tcl_distribution.json
        Prints accuracy and per-level stats
"""

import sys
import os
import json
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.tcl import classify_tcl, run_unit_tests

EXPERIMENTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Use the hand-labeled 50-SVA corpus (sample_svas.json)
DATA_FILE = os.path.join(EXPERIMENTS_DIR, "data", "sample_svas.json")
RESULTS_DIR = os.path.join(EXPERIMENTS_DIR, "results")
OUT_FILE = os.path.join(RESULTS_DIR, "tcl_distribution.json")


def run_pilot():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- Unit tests first ---
    print("=== Running TCL unit tests ===")
    unit_result = run_unit_tests(verbose=True)
    print()

    # --- Load dataset ---
    if not os.path.exists(DATA_FILE):
        print(f"ERROR: {DATA_FILE} not found.")
        sys.exit(1)

    with open(DATA_FILE) as f:
        raw = json.load(f)

    # Support both flat list format and the sample_svas.json format
    if isinstance(raw, list):
        dataset = raw
    else:
        # sample_svas.json format: {"assertions": [{id, tcl, nl, sva}, ...]}
        dataset = [
            {"sva": a["sva"], "expected_level": a["tcl"], "nl": a.get("nl",""), "id": a["id"]}
            for a in raw["assertions"]
        ]

    print(f"=== TCL Classifier Pilot ({len(dataset)} examples) ===")

    # --- Classify all ---
    results = []
    for ex in dataset:
        level, reason = classify_tcl(ex["sva"])
        ex_result = dict(ex)
        ex_result["classified_level"] = level
        ex_result["classified_reason"] = reason
        results.append(ex_result)

    # --- Stats ---
    total = len(results)

    # Accuracy against expected_level (what was intended)
    correct_vs_expected = sum(
        1 for r in results if r["classified_level"] == r["expected_level"]
    )
    acc_expected = correct_vs_expected / total if total > 0 else 0.0

    # TCL distribution of classified levels
    classified_dist = Counter(r["classified_level"] for r in results)
    expected_dist   = Counter(r["expected_level"] for r in results)

    # Per-level accuracy
    per_level_correct = defaultdict(int)
    per_level_total   = defaultdict(int)
    for r in results:
        lv = r["expected_level"]
        per_level_total[lv] += 1
        if r["classified_level"] == r["expected_level"]:
            per_level_correct[lv] += 1

    per_level_acc = {
        lv: per_level_correct[lv] / per_level_total[lv]
        for lv in per_level_total
    }

    # Confusion: where misclassifications go
    confusion = defaultdict(Counter)
    for r in results:
        if r["classified_level"] != r["expected_level"]:
            confusion[r["expected_level"]][r["classified_level"]] += 1

    print(f"\nOverall accuracy (classified vs intended): {correct_vs_expected}/{total} = {acc_expected:.1%}")
    print("\nPer-level accuracy:")
    for lv in sorted(per_level_acc):
        n_correct = per_level_correct[lv]
        n_total   = per_level_total[lv]
        acc = per_level_acc[lv]
        print(f"  L{lv}: {n_correct}/{n_total} = {acc:.1%}")

    print("\nClassified TCL distribution (what classifier outputs):")
    for lv in sorted(classified_dist):
        print(f"  L{lv}: {classified_dist[lv]}")

    print("\nExpected TCL distribution (what was requested/intended):")
    for lv in sorted(expected_dist):
        print(f"  L{lv}: {expected_dist[lv]}")

    if confusion:
        print("\nMisclassification details (expected → classified):")
        for exp_lv in sorted(confusion):
            for clf_lv, cnt in sorted(confusion[exp_lv].items()):
                print(f"  L{exp_lv} → L{clf_lv}: {cnt} cases")

    # --- Save results ---
    out = {
        "total_examples": total,
        "accuracy_vs_intended": round(acc_expected, 4),
        "correct_vs_intended": correct_vs_expected,
        "unit_tests": unit_result,
        "classified_distribution": {str(k): v for k, v in classified_dist.items()},
        "expected_distribution":   {str(k): v for k, v in expected_dist.items()},
        "per_level_accuracy": {str(k): round(v, 4) for k, v in per_level_acc.items()},
        "confusion": {
            str(exp): {str(clf): cnt for clf, cnt in clf_dist.items()}
            for exp, clf_dist in confusion.items()
        },
        "details": results,
    }

    with open(OUT_FILE, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {OUT_FILE}")

    return out


if __name__ == "__main__":
    run_pilot()
