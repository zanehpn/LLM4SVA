"""
WaveformLens pilot on synthetic waveform pairs.

Creates 10 (expected, actual) waveform pairs with known violations,
runs extract_temporal_constraints, verifies correctness.

Output: results/waveform_pilot.json
"""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.waveform_lens import (
    extract_temporal_constraints,
    format_constraints_for_prompt,
    find_transitions,
)

EXPERIMENTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(EXPERIMENTS_DIR, "data", "synthetic_waveforms")
RESULTS_DIR = os.path.join(EXPERIMENTS_DIR, "results")
OUT_FILE = os.path.join(RESULTS_DIR, "waveform_pilot.json")


# ---------------------------------------------------------------------------
# 10 synthetic waveform pairs with known ground truth
# ---------------------------------------------------------------------------
WAVEFORM_PAIRS = [
    {
        # Persistent signal (stays high after rise): only rising edge counted
        "id": 0,
        "name": "req_gnt_2cycle_delay",
        "description": "gnt rising 2 cycles late (persistent: stays high)",
        "signals": ["gnt"],
        "expected": {
            "gnt": [0, 0, 1, 1, 1, 1],   # gnt rises at cycle 2, stays
        },
        "actual": {
            "gnt": [0, 0, 0, 0, 1, 1],   # gnt rises at cycle 4 (2 cycles late)
        },
        "ground_truth": [
            {"signal": "gnt", "violation": "delayed", "expected_cycle": 2,
             "actual_cycle": 4, "delta": 2},
        ],
    },
    {
        "id": 1,
        "name": "valid_missing_entirely",
        "description": "valid signal never asserted (no rise at all)",
        "signals": ["valid"],
        "expected": {"valid": [0, 1, 1, 1, 1]},  # stays high
        "actual":   {"valid": [0, 0, 0, 0, 0]},
        "ground_truth": [
            {"signal": "valid", "violation": "missing", "expected_cycle": 1},
        ],
    },
    {
        "id": 2,
        "name": "done_early_by_1",
        "description": "done asserts 1 cycle early (persistent)",
        "signals": ["done"],
        "expected": {"done": [0, 0, 0, 1, 1]},   # rises at cycle 3, stays
        "actual":   {"done": [0, 0, 1, 1, 1]},   # rises at cycle 2 (1 cycle early)
        "ground_truth": [
            {"signal": "done", "violation": "early", "expected_cycle": 3,
             "actual_cycle": 2, "delta": -1},
        ],
    },
    {
        "id": 3,
        "name": "ack_3_cycles_late",
        "description": "ack response delayed by 3 cycles (persistent)",
        "signals": ["ack"],
        "expected": {"ack": [0, 0, 1, 1, 1, 1, 1]},  # rises at cycle 2, stays
        "actual":   {"ack": [0, 0, 0, 0, 0, 1, 1]},  # rises at cycle 5 (3 cycles late)
        "ground_truth": [
            {"signal": "ack", "violation": "delayed", "expected_cycle": 2,
             "actual_cycle": 5, "delta": 3},
        ],
    },
    {
        "id": 4,
        "name": "multi_signal_perfect_match",
        "description": "no violations — all transitions match exactly",
        "signals": ["a", "b"],
        "expected": {
            "a": [0, 1, 1, 1],
            "b": [0, 0, 1, 1],
        },
        "actual": {
            "a": [0, 1, 1, 1],
            "b": [0, 0, 1, 1],
        },
        "ground_truth": [],  # no violations
    },
    {
        "id": 5,
        "name": "two_signals_both_delayed",
        "description": "both req and gnt delayed by 1 (persistent)",
        "signals": ["req", "gnt"],
        "expected": {
            "req": [0, 1, 1, 1, 1],
            "gnt": [0, 0, 1, 1, 1],
        },
        "actual": {
            "req": [0, 0, 1, 1, 1],   # req delayed by 1
            "gnt": [0, 0, 0, 1, 1],   # gnt delayed by 1
        },
        "ground_truth": [
            {"signal": "req", "violation": "delayed", "expected_cycle": 1,
             "actual_cycle": 2, "delta": 1},
            {"signal": "gnt", "violation": "delayed", "expected_cycle": 2,
             "actual_cycle": 3, "delta": 1},
        ],
    },
    {
        "id": 6,
        "name": "falling_edge_late",
        "description": "falling edge of enable is 2 cycles late (no rising violation)",
        "signals": ["en"],
        "expected": {"en": [1, 1, 0, 0, 0]},
        "actual":   {"en": [1, 1, 1, 1, 0]},
        "ground_truth": [
            {"signal": "en", "violation": "delayed", "expected_cycle": 2,
             "actual_cycle": 4, "delta": 2},
        ],
    },
    {
        "id": 7,
        "name": "irq_ack_missing",
        "description": "irq_ack never arrives (persistent expected, nothing actual)",
        "signals": ["irq_ack"],
        "expected": {"irq_ack": [0, 0, 0, 1, 1]},  # rises at cycle 3
        "actual":   {"irq_ack": [0, 0, 0, 0, 0]},  # never asserted
        "ground_truth": [
            {"signal": "irq_ack", "violation": "missing", "expected_cycle": 3},
        ],
    },
    {
        "id": 8,
        "name": "grant_early_by_2",
        "description": "grant appears 2 cycles too early (persistent)",
        "signals": ["grant"],
        "expected": {"grant": [0, 0, 0, 0, 1, 1]},  # rises at cycle 4
        "actual":   {"grant": [0, 0, 1, 1, 1, 1]},  # rises at cycle 2 (early)
        "ground_truth": [
            {"signal": "grant", "violation": "early", "expected_cycle": 4,
             "actual_cycle": 2, "delta": -2},
        ],
    },
    {
        "id": 9,
        "name": "three_signal_mixed",
        "description": "req correct, gnt delayed 1, done missing",
        "signals": ["req", "gnt", "done"],
        "expected": {
            "req":  [0, 1, 1, 1, 1, 1],
            "gnt":  [0, 0, 1, 1, 1, 1],
            "done": [0, 0, 0, 0, 1, 1],
        },
        "actual": {
            "req":  [0, 1, 1, 1, 1, 1],
            "gnt":  [0, 0, 0, 1, 1, 1],  # delayed by 1
            "done": [0, 0, 0, 0, 0, 0],  # missing
        },
        "ground_truth": [
            {"signal": "gnt",  "violation": "delayed", "expected_cycle": 2,
             "actual_cycle": 3, "delta": 1},
            {"signal": "done", "violation": "missing", "expected_cycle": 4},
        ],
    },
]


def check_result(pair: dict, constraints: list) -> dict:
    """
    Verify WaveformLens output against ground truth.
    Returns {correct, total, details}
    """
    gt = pair["ground_truth"]
    # Build easy-lookup sets
    gt_set = set()
    for g in gt:
        key = (g["signal"], g["violation"], g.get("expected_cycle"))
        gt_set.add(key)

    found_set = set()
    for c in constraints:
        key = (c["signal"], c["violation"], c.get("expected_cycle"))
        found_set.add(key)

    tp = gt_set & found_set
    fp = found_set - gt_set
    fn = gt_set - found_set

    return {
        "gt_count": len(gt_set),
        "found_count": len(found_set),
        "tp": len(tp),
        "fp": len(fp),
        "fn": len(fn),
        "precision": len(tp) / len(found_set) if found_set else (1.0 if not gt_set else 0.0),
        "recall":    len(tp) / len(gt_set) if gt_set else (1.0 if not found_set else 0.0),
        "perfect": (len(fp) == 0 and len(fn) == 0),
    }


def run_waveform_pilot():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=== WaveformLens Pilot ===")
    print(f"Testing {len(WAVEFORM_PAIRS)} waveform pairs\n")

    # Save synthetic waveforms
    for pair in WAVEFORM_PAIRS:
        fname = os.path.join(DATA_DIR, f"pair_{pair['id']:02d}_{pair['name']}.json")
        with open(fname, "w") as f:
            json.dump(pair, f, indent=2)

    all_results = []
    perfect_count = 0
    total_tp = total_fp = total_fn = 0

    for pair in WAVEFORM_PAIRS:
        constraints = extract_temporal_constraints(
            pair["expected"], pair["actual"], pair["signals"]
        )
        check = check_result(pair, constraints)
        perfect_count += int(check["perfect"])
        total_tp += check["tp"]
        total_fp += check["fp"]
        total_fn += check["fn"]

        result = {
            "id": pair["id"],
            "name": pair["name"],
            "description": pair["description"],
            "constraints_found": constraints,
            "ground_truth": pair["ground_truth"],
            "check": check,
            "prompt_excerpt": format_constraints_for_prompt(constraints)[:300],
        }
        all_results.append(result)

        status = "PERFECT" if check["perfect"] else f"TP={check['tp']} FP={check['fp']} FN={check['fn']}"
        print(f"  [{pair['id']}] {pair['name']}: {status}")
        print(f"       {format_constraints_for_prompt(constraints)[:200]}")
        print()

    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 1.0
    overall_recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 1.0

    print(f"Summary: {perfect_count}/{len(WAVEFORM_PAIRS)} perfect matches")
    print(f"Overall Precision: {overall_precision:.1%}, Recall: {overall_recall:.1%}")

    out = {
        "total_pairs": len(WAVEFORM_PAIRS),
        "perfect_matches": perfect_count,
        "overall_precision": round(overall_precision, 4),
        "overall_recall":    round(overall_recall, 4),
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "pairs": all_results,
    }

    with open(OUT_FILE, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {OUT_FILE}")

    return out


if __name__ == "__main__":
    run_waveform_pilot()
