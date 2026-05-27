"""
SVA generation pilot using GPT-4o-mini as baseline.

- Reads 20 (NL spec, reference SVA) pairs from data/nl_to_sva.json
- Asks GPT-4o-mini to generate SVA for each NL spec (20 API calls)
- Classifies both generated and reference SVAs by TCL level
- Compares TCL distributions and reports findings

Output: results/sva_gen_pilot.json
"""

import sys
import os
import json
import time
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.tcl import classify_tcl
from src.baselines import generate_sva_gpt4o_mini
from src.mock_verifier import MockVerifier

EXPERIMENTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE    = os.path.join(EXPERIMENTS_DIR, "data", "nl_to_sva.json")
RESULTS_DIR  = os.path.join(EXPERIMENTS_DIR, "results")
OUT_FILE     = os.path.join(RESULTS_DIR, "sva_gen_pilot.json")


def run_gen_pilot():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(DATA_FILE) as f:
        pairs = json.load(f)

    verifier = MockVerifier()
    results = []
    api_calls = 0

    print(f"=== SVA Generation Pilot ({len(pairs)} pairs) ===\n")

    for pair in pairs:
        print(f"[{pair['id']:02d}] {pair['nl_spec'][:70]}...")

        # Generate
        gen_result = generate_sva_gpt4o_mini(pair["nl_spec"])
        api_calls += 1
        generated_sva = gen_result["generated_sva"]

        # Classify
        gen_level, gen_reason = classify_tcl(generated_sva)
        ref_level, ref_reason = classify_tcl(pair["reference_sva"])

        # Syntax check via mock verifier
        ver_result = verifier.verify(generated_sva)

        tcl_match = (gen_level == ref_level)
        level_diff = gen_level - ref_level

        print(f"  Generated: {generated_sva[:100]}")
        print(f"  Reference: {pair['reference_sva'][:100]}")
        print(f"  TCL: gen=L{gen_level}, ref=L{ref_level}, match={tcl_match}")
        print(f"  Syntax: {ver_result.status}")
        print()

        results.append({
            "id": pair["id"],
            "nl_spec": pair["nl_spec"],
            "reference_sva": pair["reference_sva"],
            "reference_tcl": ref_level,
            "reference_tcl_reason": ref_reason,
            "generated_sva": generated_sva,
            "generated_tcl": gen_level,
            "generated_tcl_reason": gen_reason,
            "tcl_match": tcl_match,
            "level_diff": level_diff,  # positive = gen is higher complexity
            "syntax_ok": ver_result.syntax_ok,
            "syntax_status": ver_result.status,
            "gpt_usage": gen_result["usage"],
        })

        time.sleep(0.2)  # rate limit

    # --- Analysis ---
    gen_dist  = Counter(r["generated_tcl"] for r in results)
    ref_dist  = Counter(r["reference_tcl"]  for r in results)
    tcl_match_rate = sum(r["tcl_match"] for r in results) / len(results)
    syntax_ok_rate = sum(r["syntax_ok"] for r in results) / len(results)
    total_tokens = sum(r["gpt_usage"]["total_tokens"] for r in results)

    # Bias analysis: does GPT-4o-mini under-generate high TCL?
    avg_gen = sum(r["generated_tcl"] for r in results) / len(results)
    avg_ref = sum(r["reference_tcl"]  for r in results) / len(results)
    level_diff_avg = avg_gen - avg_ref

    print("=== Results Summary ===")
    print(f"API calls used: {api_calls}")
    print(f"Total tokens: {total_tokens}")
    print(f"TCL match rate (gen == ref): {tcl_match_rate:.1%}")
    print(f"Syntax OK rate: {syntax_ok_rate:.1%}")
    print(f"Avg generated TCL: {avg_gen:.2f}, Avg reference TCL: {avg_ref:.2f}")
    print(f"Level diff (gen-ref): {level_diff_avg:+.2f} ({'SIMPLER' if level_diff_avg < 0 else 'COMPLEX' if level_diff_avg > 0 else 'SAME'})")
    print()
    print("Generated TCL distribution:")
    for lv in sorted(gen_dist):
        bar = "#" * gen_dist[lv]
        print(f"  L{lv}: {gen_dist[lv]:3d}  {bar}")
    print("Reference TCL distribution:")
    for lv in sorted(ref_dist):
        bar = "#" * ref_dist[lv]
        print(f"  L{lv}: {ref_dist[lv]:3d}  {bar}")

    out = {
        "total_pairs": len(pairs),
        "api_calls": api_calls,
        "total_tokens": total_tokens,
        "tcl_match_rate": round(tcl_match_rate, 4),
        "syntax_ok_rate": round(syntax_ok_rate, 4),
        "avg_generated_tcl": round(avg_gen, 4),
        "avg_reference_tcl": round(avg_ref, 4),
        "level_diff_avg": round(level_diff_avg, 4),
        "bias_direction": "simpler" if level_diff_avg < -0.1 else "complex" if level_diff_avg > 0.1 else "neutral",
        "generated_tcl_distribution": {str(k): v for k, v in gen_dist.items()},
        "reference_tcl_distribution": {str(k): v for k, v in ref_dist.items()},
        "details": results,
    }

    with open(OUT_FILE, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {OUT_FILE}")

    return out


if __name__ == "__main__":
    run_gen_pilot()
