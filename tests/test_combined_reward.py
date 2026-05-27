#!/usr/bin/env python3
"""
test_combined_reward.py — evaluate the full 4-component reward's
differentiating power (proposal's formula).

Reward = 0.15·syntax + 0.40·formal + 0.20·(1-vacuity) + 0.25·AST_sim

Re-uses scripts/test_formal_mutation.py's golden + 3 mutations + vacuous
construction, but scores each rollout with the combined reward.
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import random
import re
import sys
from collections import Counter
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXPERIMENTS_DIR))
from src.formal_verify import formal_verify
from src.mock_verifier import syntax_check
from src.vacuity import is_vacuous_syntactic

# Pull mutate_sva/make_vacuous from the sibling test_formal_mutation file.
sys.path.insert(0, str(EXPERIMENTS_DIR / "tests"))
from test_formal_mutation import (
    mutate_sva, make_vacuous,
)

TIER1_POOL = EXPERIMENTS_DIR / "data" / "train" / "grpo" / "verifiable_parseable.jsonl"

W_SYN = 0.15
W_FV = 0.40
W_VAC = 0.20
W_SIM = 0.25


# ---- AST similarity (same as run_grpo_pilot.py) --------------------------
def _tokenize(sva: str):
    s = re.sub(r"\s+", " ", sva).strip()
    return re.findall(r"[A-Za-z_]\w*|##\[[^\]]*\]|##\d+|\|->|\|=>|[()\[\];,.]|\S", s)


def _lev(a, b):
    n, m = len(a), len(b)
    if n == 0: return m
    if m == 0: return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            tmp = dp[j]
            dp[j] = prev if a[i - 1] == b[j - 1] else 1 + min(dp[j], dp[j - 1], prev)
            prev = tmp
    return dp[m]


def ast_sim(g: str, r: str) -> float:
    if not g or not r: return 0.0
    a, b = _tokenize(g), _tokenize(r)
    if not a or not b: return 0.0
    return max(0.0, 1.0 - _lev(a, b) / max(len(a), len(b)))


def combined_reward(sva: str, rtl: str, ref_sva: str) -> dict:
    syn = 1.0 if syntax_check(sva)["ok"] else 0.0
    sim = ast_sim(sva, ref_sva)
    fv = formal_verify(sva, rtl, timeout=12, depth=8)
    # vacuity already baked into formal (VACUOUS status → reward 0.2)
    # but compute (1-vacuity) as its own term here
    vac_bool, _ = is_vacuous_syntactic(sva)
    non_vacuous = 0.0 if vac_bool else 1.0
    # for formal reward in combined: map {PASS→1, FAIL→0.5, TIMEOUT→0.3, VACUOUS→0, PARSE_ERROR→0}
    fv_map = {"PASS": 1.0, "FAIL": 0.5, "TIMEOUT": 0.3,
              "VACUOUS": 0.0, "PARSE_ERROR": 0.0, "EXTRACT_ERROR": 0.0,
              "UNSUPPORTED": 0.5}
    fv_score = fv_map.get(fv.status, 0.0)
    R = W_SYN * syn + W_FV * fv_score + W_VAC * non_vacuous + W_SIM * sim
    return {
        "R": round(R, 3),
        "syn": round(syn, 3),
        "formal": round(fv_score, 3),
        "formal_status": fv.status,
        "non_vacuous": non_vacuous,
        "sim": round(sim, 3),
    }


def _work(args):
    tag, sva, rtl, ref = args
    r = combined_reward(sva, rtl, ref)
    return tag, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out",
                    default=str(EXPERIMENTS_DIR / "results" / "combined_reward_oracle.json"))
    args = ap.parse_args()
    random.seed(args.seed)

    samples = []
    with open(TIER1_POOL) as f:
        for line in f:
            r = json.loads(line)
            if r.get("expected_tcl", r.get("tcl", 0)) in (1, 2, 4):
                samples.append(r)
    random.shuffle(samples)
    samples = samples[:args.n]
    print(f"[combined] using {len(samples)} Tier-1 samples")

    tasks = []
    for i, s in enumerate(samples):
        # Golden (reference = itself, AST_sim = 1 definitionally)
        tasks.append((f"golden_{i}", s["sva"], s["rtl_module"], s["sva"]))
        for k in ("swap", "flip", "delay"):
            tasks.append((f"mut_{k}_{i}", mutate_sva(s["sva"], k),
                           s["rtl_module"], s["sva"]))
        tasks.append((f"vacuous_{i}", make_vacuous(),
                       s["rtl_module"], s["sva"]))
    print(f"[combined] verifier runs: {len(tasks)}")

    res = {}
    done = 0
    with mp.Pool(args.workers) as pool:
        for tag, r in pool.imap_unordered(_work, tasks, chunksize=2):
            res[tag] = r
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(tasks)}")

    def bucket(prefix):
        return [v["R"] for k, v in res.items() if k.startswith(prefix)]
    def component(prefix, field):
        return [v[field] for k, v in res.items() if k.startswith(prefix)]

    def avg(x):
        return sum(x) / max(len(x), 1)

    print("\n" + "=" * 60)
    print("4-COMPONENT REWARD DIFFERENTIATION")
    print("=" * 60)
    print(f"{'type':<15} {'R':>6}  {'syn':>5} {'formal':>6} {'1-vac':>5} {'sim':>5}")
    for kind in ["golden_", "vacuous_", "mut_swap_", "mut_flip_", "mut_delay_"]:
        name = kind.rstrip("_")
        print(f"  {name:<13} {avg(bucket(kind)):>6.3f}  "
              f"{avg(component(kind,'syn')):>5.2f} "
              f"{avg(component(kind,'formal')):>6.2f} "
              f"{avg(component(kind,'non_vacuous')):>5.2f} "
              f"{avg(component(kind,'sim')):>5.2f}")
    dvac = avg(bucket("golden_")) - avg(bucket("vacuous_"))
    dmut = avg(bucket("golden_")) - (avg(bucket("mut_swap_"))
                                      + avg(bucket("mut_flip_"))
                                      + avg(bucket("mut_delay_"))) / 3
    print()
    print(f"golden − vacuous  = {dvac:+.3f}")
    print(f"golden − mut(avg) = {dmut:+.3f}")
    ok = dvac > 0.1 and dmut > 0.1
    print(f"\n{'✓ USABLE as GRPO reward' if ok else '⚠ still weak'}")

    rep = {
        "mean_R": {k.rstrip("_"): round(avg(bucket(k)), 3)
                   for k in ["golden_", "vacuous_", "mut_swap_",
                             "mut_flip_", "mut_delay_"]},
        "delta_golden_vacuous": round(dvac, 3),
        "delta_golden_mutation": round(dmut, 3),
    }
    json.dump(rep, open(args.out, "w"), indent=2)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
