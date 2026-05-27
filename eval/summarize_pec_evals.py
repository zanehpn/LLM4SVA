#!/usr/bin/env python3
"""
summarize_pec_evals.py — table of all SFT_replay50 + GRPO variant Func@1 scores.
Used to compare GRPO Phase 1 / Phase 2 / Phase 2.5 against the SFT baseline.
"""
import json
import re
import sys
from pathlib import Path

RESULTS_DIR = Path("${REPO_ROOT}/results/")


def variant_key(name: str):
    """Sort key: SFT-only first, then by GRPO version, then by step."""
    m_v = re.search(r"grpo[_a-z]*([0-9]+|phase2[_a-z0-9]*)", name)
    v = m_v.group(1) if m_v else ""
    m_step = re.search(r"step([0-9]+)|checkpoint-([0-9]+)|phase2_([0-9]+)", name)
    step = int(m_step.group(1) or m_step.group(2) or m_step.group(3)) if m_step else 0
    sft_only = ("grpo" not in name)
    return (not sft_only, v, step)


def main():
    files = sorted(RESULTS_DIR.glob("pec_eval_*.json"), key=lambda p: variant_key(p.name))
    print(f"{'variant':<70}  {'n':>4}  {'Func@1':>7}  {'Relaxed':>7}  "
          f"{'L1':>5}  {'L4':>5}  {'L5':>5}")
    print("-" * 110)
    for f in files:
        try:
            d = json.load(open(f))
        except Exception as e:
            print(f"{f.name[:70]:<70}  ERROR: {e}")
            continue
        n = d.get("evaluable", d.get("total", "?"))
        strict = d.get("func1_full_pct", d.get("func_at_1_pct", "?"))
        relax = d.get("func1_relaxed_pct", "?")
        per_tcl = d.get("per_tcl", {})
        l1 = per_tcl.get("1", {}).get("func1_full_pct", "?")
        l4 = per_tcl.get("4", {}).get("func1_full_pct", "?")
        l5 = per_tcl.get("5", {}).get("func1_full_pct", "?")
        # Strip the date suffix for shorter display
        name = re.sub(r"_\d{8}_\d{6}\.json$", "", f.name)
        print(f"{name[:70]:<70}  {n:>4}  {strict:>7}  {relax:>7}  "
              f"{l1:>5}  {l4:>5}  {l5:>5}")


if __name__ == "__main__":
    main()
