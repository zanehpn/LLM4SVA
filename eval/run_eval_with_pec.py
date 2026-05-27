#!/usr/bin/env python3
"""
run_eval_with_pec.py — re-score an existing NL2SVA-Human eval result with
PEC functional equivalence (Yosys-based) instead of TCL category match.

Reads a JSON produced by `run_eval_nl2sva_human.py` (which has the
`per_sample` array with `generated_sva` for each task), then for each task
runs `prop_equivalence(generated_sva, reference_sva, rtl_context)` to get
the PEC verdict. Aggregates Func@1 (= % EQUIVALENT) overall and per TCL.

Tasks where the PEC oracle cannot evaluate (UNSUPPORTED / PARSE_ERROR /
TIMEOUT on the reference itself) are excluded from the denominator and
reported separately.

Usage:
  source ${OSS_CAD_SUITE}/environment
  PYTHONPATH=. python3 scripts/run_eval_with_pec.py \\
      --eval-result results/eval_nl2sva_human_<model>_<ts>.json \\
      --workers 8 --depth 15 --timeout 30
"""
import argparse
import json
import multiprocessing as mp
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXPERIMENTS_DIR))
from src.pec_yosys import prop_equivalence

TEST_PATH = EXPERIMENTS_DIR / "data" / "test" / "nl2sva_human.jsonl"
COVERAGE_PATH = EXPERIMENTS_DIR / "results" / "pec_coverage_check.json"
RESULTS_DIR = EXPERIMENTS_DIR / "results"


def load_oracle_coverage() -> dict:
    """Map task_id → reference-vs-reference verdict.
    A task is 'evaluable' iff its self-equivalence verdict is EQUIVALENT."""
    if not COVERAGE_PATH.exists():
        raise SystemExit(
            f"missing {COVERAGE_PATH}. Run scripts/pec_coverage_check.py first.")
    cov = json.load(open(COVERAGE_PATH))
    return {r["id"]: r["verdict"] for r in cov["rows"]}


def _work(args):
    idx, gen_sva, ref_sva, rtl, depth, timeout = args
    if not gen_sva.strip():
        return idx, "EMPTY", "", "", 0.0
    r = prop_equivalence(gen_sva, ref_sva, rtl,
                         depth=depth, timeout=timeout)
    return idx, r.verdict, r.fwd_status, r.bwd_status, round(r.wallclock_s, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-result", required=True,
                    help="path to JSON produced by run_eval_nl2sva_human.py")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--depth", type=int, default=15)
    ap.add_argument("--timeout", type=int, default=30)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    eval_path = Path(args.eval_result)
    if not eval_path.exists():
        raise SystemExit(f"eval result not found: {eval_path}")
    data = json.load(open(eval_path))
    model_name = data.get("model", eval_path.stem)
    samples = data.get("per_sample", [])
    if args.limit:
        samples = samples[:args.limit]

    # Build id → reference / rtl lookup from the test set
    test_lookup = {}
    with open(TEST_PATH) as f:
        for line in f:
            r = json.loads(line)
            test_lookup[r["id"]] = (
                r.get("reference_sva", ""),
                r.get("rtl_context", ""),
                int(r.get("expected_tcl", 0)),
            )

    # Load oracle coverage map
    coverage = load_oracle_coverage()

    work = []
    expected_tcls = []
    ids = []
    evaluable_mask = []
    for i, s in enumerate(samples):
        sid = s.get("id")
        gen = s.get("generated_sva", "")
        ref, rtl, tcl = test_lookup.get(sid, ("", "", 0))
        ids.append(sid); expected_tcls.append(tcl)
        if coverage.get(sid) != "EQUIVALENT":
            evaluable_mask.append(False)
            continue
        evaluable_mask.append(True)
        work.append((i, gen, ref, rtl, args.depth, args.timeout))

    print(f"[pec-eval] model={model_name}")
    print(f"[pec-eval] samples={len(samples)}  "
          f"evaluable={sum(evaluable_mask)}  "
          f"oracle-blocked={sum(1 for m in evaluable_mask if not m)}")
    print()

    verdicts = [None] * len(samples)
    fwd = [""] * len(samples)
    bwd = [""] * len(samples)
    seconds = [0.0] * len(samples)
    by_v = Counter()
    done = 0
    with mp.Pool(args.workers) as pool:
        for idx, v, f_, b_, dt in pool.imap_unordered(_work, work, chunksize=2):
            verdicts[idx] = v; fwd[idx] = f_; bwd[idx] = b_; seconds[idx] = dt
            by_v[v] += 1
            done += 1
            if done % 10 == 0 or done == len(work):
                print(f"  {done}/{len(work)}  "
                      f"verdict counts: {dict(by_v)}")

    # Aggregate Func@1 over evaluable subset only
    n_eval = sum(evaluable_mask)
    n_equiv = sum(1 for v in verdicts if v == "EQUIVALENT")
    n_implies_fwd = sum(1 for v in verdicts if v == "IMPLIES_REF_TO_LM")
    n_implies_bwd = sum(1 for v in verdicts if v == "IMPLIES_LM_TO_REF")
    n_func_relaxed = n_equiv + n_implies_fwd + n_implies_bwd
    func1_pct = 100 * n_equiv / max(n_eval, 1)
    func1_relaxed_pct = 100 * n_func_relaxed / max(n_eval, 1)

    # Per-TCL Func@1
    per_tcl = defaultdict(lambda: {"eval": 0, "equiv": 0, "relaxed": 0})
    for i, v in enumerate(verdicts):
        if not evaluable_mask[i]:
            continue
        t = expected_tcls[i]
        per_tcl[t]["eval"] += 1
        if v == "EQUIVALENT":
            per_tcl[t]["equiv"] += 1
            per_tcl[t]["relaxed"] += 1
        elif v in ("IMPLIES_REF_TO_LM", "IMPLIES_LM_TO_REF"):
            per_tcl[t]["relaxed"] += 1

    print()
    print("=" * 60)
    print(f"PEC FUNCTIONAL EQUIVALENCE — model={model_name}")
    print("=" * 60)
    print(f"Evaluable subset: {n_eval}/{len(samples)} "
          f"(oracle covers {n_eval} tasks)")
    print(f"Func@1 (full equivalence):    "
          f"{n_equiv}/{n_eval} = {func1_pct:.1f}%")
    print(f"Func@1 (either direction):    "
          f"{n_func_relaxed}/{n_eval} = {func1_relaxed_pct:.1f}%")
    print()
    print(f"Per-TCL Func@1:")
    print(f"  TCL   eval   equiv   relaxed   pct(equiv)   pct(relaxed)")
    for t in sorted(per_tcl):
        d = per_tcl[t]
        e_pct = 100 * d["equiv"] / max(d["eval"], 1)
        r_pct = 100 * d["relaxed"] / max(d["eval"], 1)
        print(f"  L{t}    {d['eval']:>4d}   {d['equiv']:>5d}   {d['relaxed']:>7d}   "
              f"{e_pct:>10.1f}%   {r_pct:>11.1f}%")
    print()
    print(f"Verdict distribution: {dict(by_v)}")

    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"pec_eval_{model_name}_{ts}.json"
    json.dump({
        "model": model_name,
        "source_eval": str(eval_path),
        "evaluable": n_eval,
        "total": len(samples),
        "func1_full_pct": round(func1_pct, 2),
        "func1_relaxed_pct": round(func1_relaxed_pct, 2),
        "per_tcl": {str(k): v for k, v in per_tcl.items()},
        "verdict_counts": dict(by_v),
        "rows": [{
            "id": ids[i], "tcl": expected_tcls[i],
            "evaluable": evaluable_mask[i],
            "verdict": verdicts[i], "fwd": fwd[i], "bwd": bwd[i],
            "seconds": seconds[i],
        } for i in range(len(samples))],
    }, open(out_path, "w"), indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
