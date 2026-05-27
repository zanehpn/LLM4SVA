#!/usr/bin/env python3
"""
run_nl2sva_pilot.py — NL→SVA generation with TCL-stratified evaluation.

Uses gpt-4o-mini to generate SVAs from 10 NL descriptions (nl2sva_tasks.json).
Classifies each generated SVA by TCL level and checks syntactic validity.

Metrics:
  - Per-TCL syntactic validity rate (does output look like a valid SVA?)
  - TCL level match rate (did model generate the right complexity?)
  - Cost estimate

Usage:
    cd outputs/SVA4DAC/experiments
    python scripts/run_nl2sva_pilot.py

Requires: OPENAI_API_KEY environment variable
Budget: ~$0.01-0.05 for 10 tasks at gpt-4o-mini rates
"""

import json
import sys
import os
import time
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tcl import classify_tcl
from src.mock_verifier import MockVerifier, syntax_check
from src.baselines import generate_sva_gpt4o_mini

EXPERIMENTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASKS_FILE = os.path.join(EXPERIMENTS_DIR, "data", "nl2sva_tasks.json")
RESULTS_DIR = os.path.join(EXPERIMENTS_DIR, "results")

# Per-TCL syntactic validity check using regex
def is_syntactically_valid_sva(sva_text: str) -> bool:
    """
    Check if generated text looks like a valid SVA.
    Uses the mock verifier's syntax check + extra heuristics.
    """
    # Strip markdown code fences
    cleaned = sva_text.strip()
    for fence in ["```systemverilog", "```sv", "```verilog", "```"]:
        if cleaned.startswith(fence):
            cleaned = cleaned[len(fence):]
    cleaned = cleaned.strip().rstrip("`").strip()
    
    result = syntax_check(cleaned)
    return result["ok"]


def extract_sva_from_response(response_text: str) -> str:
    """
    Extract clean SVA code from GPT response (may include markdown fences).
    """
    text = response_text.strip()
    # Try to extract from code fence
    import re
    fence_match = re.search(r'```(?:systemverilog|sv|verilog|)?(.+?)```', text, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()
    return text


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set.")
        print("       Export it and re-run: export OPENAI_API_KEY=sk-...")
        sys.exit(1)

    print("=" * 60)
    print("NL2SVA Pilot — gpt-4o-mini TCL-stratified evaluation")
    print(f"Timestamp: {timestamp}")
    print("=" * 60)

    # Load tasks
    with open(TASKS_FILE) as f:
        task_data = json.load(f)
    tasks = task_data["tasks"]
    print(f"\nLoaded {len(tasks)} NL→SVA tasks")

    verifier = MockVerifier()
    results = []
    per_tcl_stats = defaultdict(lambda: {"total": 0, "syntax_ok": 0, "tcl_match": 0})
    total_tokens = 0

    print("\nGenerating SVAs...")
    print("-" * 60)

    for task in tasks:
        task_id = task["id"]
        nl = task["nl"]
        ref_sva = task["reference_sva"]
        expected_tcl = task["expected_tcl"]

        print(f"\n[{task_id}] (expected TCL-{expected_tcl})")
        print(f"  NL: {nl[:80]}")

        # Generate
        try:
            gen_result = generate_sva_gpt4o_mini(
                nl_spec=nl,
                temperature=0.1,
                max_tokens=200,
            )
            generated_raw = gen_result["generated_sva"]
            usage = gen_result["usage"]
            total_tokens += usage["total_tokens"]
        except Exception as e:
            print(f"  ERROR: API call failed: {e}")
            results.append({
                "id": task_id,
                "nl": nl,
                "expected_tcl": expected_tcl,
                "generated_raw": "",
                "generated_sva": "",
                "syntax_ok": False,
                "tcl_match": False,
                "gen_tcl": None,
                "error": str(e),
            })
            per_tcl_stats[expected_tcl]["total"] += 1
            continue

        # Extract and classify
        generated_sva = extract_sva_from_response(generated_raw)
        syntax_ok = is_syntactically_valid_sva(generated_sva)
        gen_tcl, gen_reason = classify_tcl(generated_sva)
        tcl_match = (gen_tcl == expected_tcl)

        # Reference classification
        ref_tcl, _ = classify_tcl(ref_sva)

        per_tcl_stats[expected_tcl]["total"] += 1
        if syntax_ok:
            per_tcl_stats[expected_tcl]["syntax_ok"] += 1
        if tcl_match:
            per_tcl_stats[expected_tcl]["tcl_match"] += 1

        status_sym = "✓" if syntax_ok else "✗"
        tcl_sym    = "=" if tcl_match else "≠"
        print(f"  Gen: {generated_sva[:80]}")
        print(f"  Syntax: {status_sym}  TCL: gen=L{gen_tcl} {tcl_sym} expected=L{expected_tcl}  tokens={usage['total_tokens']}")

        results.append({
            "id": task_id,
            "nl": nl,
            "expected_tcl": expected_tcl,
            "reference_sva": ref_sva,
            "reference_tcl": ref_tcl,
            "generated_raw": generated_raw,
            "generated_sva": generated_sva,
            "syntax_ok": syntax_ok,
            "tcl_match": tcl_match,
            "gen_tcl": gen_tcl,
            "gen_reason": gen_reason,
            "usage": usage,
        })

        time.sleep(0.3)  # gentle rate limiting

    # Summary
    total = len(results)
    syntax_ok_total = sum(1 for r in results if r.get("syntax_ok", False))
    tcl_match_total = sum(1 for r in results if r.get("tcl_match", False))
    syntax_rate = syntax_ok_total / total if total > 0 else 0
    tcl_match_rate = tcl_match_total / total if total > 0 else 0

    # Estimated cost (gpt-4o-mini: $0.15/1M input, $0.60/1M output tokens)
    estimated_cost = (total_tokens / 1_000_000) * 0.60  # rough upper bound

    print("\n" + "=" * 60)
    print("NL2SVA PILOT RESULTS")
    print("=" * 60)
    print(f"\nOverall:")
    print(f"  Tasks completed:       {total}/10")
    print(f"  Syntactically valid:   {syntax_ok_total}/{total} ({syntax_rate*100:.0f}%)")
    print(f"  TCL level match:       {tcl_match_total}/{total} ({tcl_match_rate*100:.0f}%)")
    print(f"  Total tokens used:     {total_tokens}")
    print(f"  Estimated API cost:    ${estimated_cost:.4f}")

    print(f"\nPer-TCL-Level breakdown:")
    print(f"  {'Level':<10} {'Syntax OK':<12} {'TCL Match':<12}")
    print(f"  {'-----':<10} {'---------':<12} {'---------':<12}")
    for lv in range(1, 6):
        stats = per_tcl_stats[lv]
        if stats["total"] == 0:
            continue
        syn_str = f"{stats['syntax_ok']}/{stats['total']}"
        tcl_str = f"{stats['tcl_match']}/{stats['total']}"
        print(f"  L{lv:<9} {syn_str:<12} {tcl_str:<12}")

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"nl2sva_pilot_{timestamp}.json")
    output = {
        "timestamp": timestamp,
        "model": "gpt-4o-mini",
        "total_tasks": total,
        "syntax_valid_count": syntax_ok_total,
        "syntax_valid_rate": syntax_rate,
        "tcl_match_count": tcl_match_total,
        "tcl_match_rate": tcl_match_rate,
        "total_tokens": total_tokens,
        "estimated_cost_usd": estimated_cost,
        "per_tcl_stats": {
            f"L{lv}": dict(per_tcl_stats[lv]) for lv in range(1, 6)
        },
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {out_path}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
