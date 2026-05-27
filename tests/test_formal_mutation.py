#!/usr/bin/env python3
"""
test_formal_mutation.py — measure the **differentiating power** of our
formal-verify reward, which is what GRPO actually needs.

For each Tier-1 sample, generate 3 kinds of input:
  golden:   the original SVA from the dataset
  mutated:  syntactically valid but semantically wrong mutations of golden
  vacuous:  `assert property (@(posedge clk) 1'b1);`

We measure per-type PASS/FAIL/ERROR distributions, then report the key metric:

  reward(golden) - reward(mutated) > 0  ← this is all GRPO needs.
  reward(golden) - reward(vacuous) > 0  ← vacuity defense.

If these deltas are > 0.1 on average, the reward is usable as a group-relative
signal in GRPO (even if the absolute PASS rate on golden is <60%).
"""
import argparse
import json
import multiprocessing as mp
import random
import re
import sys
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXPERIMENTS_DIR))
from src.formal_verify import formal_verify

TIER1_POOL = EXPERIMENTS_DIR / "data" / "train" / "grpo" / "verifiable_parseable.jsonl"


# --- mutation primitives ------------------------------------------------

def _split_top_implication(sva: str):
    """Find the FIRST top-level `|->` / `|=>` operator inside the SVA body.

    Walks the string with a paren/bracket/brace depth counter, ignoring
    `|->`/`|=>` that appear inside nested parentheses (which are sub-clauses
    bound tighter than the outermost implication). Returns
    `(prefix, lhs, op, rhs, suffix)` where the original equals
    `prefix + lhs + op + rhs + suffix` modulo whitespace, OR None if no
    top-level implication is found.
    """
    # We scan only inside the body of `assert/assume/cover property (...)`.
    m = re.search(
        r"((?:[A-Za-z_]\w*\s*:\s*)?(?:assert|assume|cover)\s+property\s*\()",
        sva,
    )
    if not m:
        return None
    open_at = m.end() - 1   # index of the '('
    # Find matching close
    depth = 0
    body_end = None
    for i in range(open_at, len(sva)):
        if sva[i] == "(":
            depth += 1
        elif sva[i] == ")":
            depth -= 1
            if depth == 0:
                body_end = i
                break
    if body_end is None:
        return None
    body_start = open_at + 1
    body = sva[body_start:body_end]
    # Skip leading clocking event @(posedge clk) and any `disable iff (...)`
    cursor = 0
    # @(...)
    cm = re.match(r"\s*@\s*\(", body)
    if cm:
        d = 0
        j = cm.end() - 1
        while j < len(body):
            if body[j] == "(": d += 1
            elif body[j] == ")":
                d -= 1
                if d == 0:
                    j += 1; break
            j += 1
        cursor = j
    # disable iff (...)
    dm = re.match(r"\s*disable\s+iff\s*\(", body[cursor:])
    if dm:
        d = 0
        j = cursor + dm.end() - 1
        while j < len(body):
            if body[j] == "(": d += 1
            elif body[j] == ")":
                d -= 1
                if d == 0:
                    j += 1; break
            j += 1
        cursor = j

    # Now scan for a top-level |-> or |=> at depth 0 in the remaining body
    depth_p = depth_b = depth_c = 0
    op_at = None
    op_kind = None
    i = cursor
    while i < len(body) - 1:
        ch = body[i]
        if ch == "(": depth_p += 1
        elif ch == ")": depth_p -= 1
        elif ch == "[": depth_b += 1
        elif ch == "]": depth_b -= 1
        elif ch == "{": depth_c += 1
        elif ch == "}": depth_c -= 1
        if depth_p == depth_b == depth_c == 0:
            if ch == "|" and i + 2 < len(body):
                if body[i + 1] == "-" and body[i + 2] == ">":
                    op_at = i; op_kind = "|->"; break
                if body[i + 1] == "=" and body[i + 2] == ">":
                    op_at = i; op_kind = "|=>"; break
        i += 1
    if op_at is None:
        return None

    lhs = body[cursor:op_at].strip()
    rhs = body[op_at + 3:].strip()
    if not lhs or not rhs:
        return None

    prefix = sva[:body_start] + body[:cursor]
    suffix = sva[body_end:]
    return prefix, lhs, op_kind, rhs, suffix


def mutate_swap_implication(sva: str) -> str:
    """Swap LHS and RHS of the top-level `|->` / `|=>` operator. Returns the
    original string if no top-level implication exists (caller treats that as
    a no-op mutation)."""
    s = _split_top_implication(sva)
    if s is None:
        return sva
    prefix, lhs, op, rhs, suffix = s
    # Strip ##N / ##[a:b] off the RHS so the swapped operand is the *Boolean*
    # the antecedent should react to, not a delay-fronted sequence which
    # would be parser-illegal as an LHS.
    rhs_no_delay = re.sub(r"^##\s*(?:\[\s*\d+\s*:\s*(?:\d+|\$)\s*\]|\d+)\s*",
                           "", rhs)
    sep = "" if prefix.endswith(" ") else " "
    return f"{prefix}{sep}{rhs_no_delay} {op} {lhs}{suffix}"


def mutate_flip_condition(sva: str) -> str:
    """Negate one `==` to `!=` or vice versa (break the condition).
    If neither exists, fall back to flipping the first `&&` to `||`."""
    if "==" in sva:
        return sva.replace("==", "!=", 1)
    if "!=" in sva:
        return sva.replace("!=", "==", 1)
    if "&&" in sva:
        return sva.replace("&&", "||", 1)
    return sva


def mutate_change_delay(sva: str) -> str:
    """Change `|->` to `|=>`, or vice versa (off-by-one cycle).
    If neither exists, try shifting a fixed `##N` to `##(N+1)`."""
    if "|->" in sva:
        return sva.replace("|->", "|=>", 1)
    if "|=>" in sva:
        return sva.replace("|=>", "|->", 1)
    m = re.search(r"##\s*(\d+)", sva)
    if m:
        n = int(m.group(1))
        return sva.replace(m.group(0), f"##{n + 1}", 1)
    return sva


def mutate_sva(sva: str, kind: str) -> str:
    """Apply one mutation. Return original if mutation didn't apply."""
    if kind == "swap":
        return mutate_swap_implication(sva)
    if kind == "flip":
        return mutate_flip_condition(sva)
    if kind == "delay":
        return mutate_change_delay(sva)
    return sva


def make_vacuous() -> str:
    return "assert property (@(posedge clk) 1'b1);"


# --- run one verify ------------------------------------------------------

def _work(args):
    tag, sva, rtl = args
    r = formal_verify(sva, rtl, timeout=12, depth=8)
    return tag, r.status, r.reward if r.reward is not None else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out",
                    default=str(EXPERIMENTS_DIR / "results" / "mutation_oracle.json"))
    args = ap.parse_args()
    random.seed(args.seed)

    samples = []
    with open(TIER1_POOL) as f:
        for line in f:
            r = json.loads(line)
            if r.get("expected_tcl", r.get("tcl", 0)) in (1, 2, 4):    # only patterns we actually lower
                samples.append(r)
    random.shuffle(samples)
    samples = samples[:args.n]
    print(f"[mutation] using {len(samples)} Tier-1 samples")

    tasks = []
    noop_counts = {"swap": 0, "flip": 0, "delay": 0}
    for i, s in enumerate(samples):
        tasks.append((f"golden_{i}", s["sva"], s["rtl_module"]))
        for k in ("swap", "flip", "delay"):
            mutated = mutate_sva(s["sva"], k)
            if re.sub(r"\s+", " ", mutated).strip() \
               == re.sub(r"\s+", " ", s["sva"]).strip():
                noop_counts[k] += 1
            tasks.append((f"mut_{k}_{i}", mutated, s["rtl_module"]))
        tasks.append((f"vacuous_{i}", make_vacuous(), s["rtl_module"]))
    print(f"[mutation] total verifier runs: {len(tasks)}")
    print(f"[mutation] mutation no-op counts (regex didn't change SVA): "
          f"{noop_counts}  (out of {len(samples)} per kind)")

    out = {}
    done = 0
    with mp.Pool(args.workers) as pool:
        for tag, status, reward in pool.imap_unordered(_work, tasks, chunksize=2):
            out[tag] = {"status": status, "reward": reward}
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(tasks)}")

    # Aggregate
    def bucket(prefix):
        return [v["reward"] for k, v in out.items() if k.startswith(prefix)]
    def bucket_status(prefix):
        from collections import Counter
        return dict(Counter(v["status"] for k, v in out.items()
                            if k.startswith(prefix)))

    golden = bucket("golden_")
    vacuous = bucket("vacuous_")
    mut_swap = bucket("mut_swap_")
    mut_flip = bucket("mut_flip_")
    mut_delay = bucket("mut_delay_")

    def avg(x):
        return sum(x) / max(len(x), 1)

    print("\n" + "=" * 60)
    print("DIFFERENTIATION STUDY")
    print("=" * 60)
    print(f"                mean_reward   status")
    print(f"  golden         {avg(golden):>5.3f}    "
          f"{bucket_status('golden_')}")
    print(f"  vacuous        {avg(vacuous):>5.3f}    "
          f"{bucket_status('vacuous_')}")
    print(f"  mut:swap       {avg(mut_swap):>5.3f}    "
          f"{bucket_status('mut_swap_')}")
    print(f"  mut:flip       {avg(mut_flip):>5.3f}    "
          f"{bucket_status('mut_flip_')}")
    print(f"  mut:delay      {avg(mut_delay):>5.3f}    "
          f"{bucket_status('mut_delay_')}")

    delta_vac = avg(golden) - avg(vacuous)
    delta_mut = avg(golden) - (avg(mut_swap) + avg(mut_flip) + avg(mut_delay)) / 3
    print(f"\nKEY DELTAS:")
    print(f"  golden − vacuous  = {delta_vac:+.3f}   "
          f"(target > 0: golden beats trivially-true)")
    print(f"  golden − mut(avg) = {delta_mut:+.3f}   "
          f"(target > 0: golden beats semantically-wrong)")

    if delta_vac > 0.05 and delta_mut > 0.05:
        print("\n✓ Reward is usable as a GRPO signal.")
    else:
        print("\n⚠ Reward has insufficient differentiating power.")

    report = {
        "n_samples": len(samples),
        "n_verifier_runs": len(tasks),
        "mutation_noop_counts": noop_counts,
        "mean_reward": {
            "golden": round(avg(golden), 3),
            "vacuous": round(avg(vacuous), 3),
            "mut_swap": round(avg(mut_swap), 3),
            "mut_flip": round(avg(mut_flip), 3),
            "mut_delay": round(avg(mut_delay), 3),
        },
        "delta_golden_minus_vacuous": round(delta_vac, 3),
        "delta_golden_minus_mutation": round(delta_mut, 3),
        "status_counts": {
            "golden": bucket_status("golden_"),
            "vacuous": bucket_status("vacuous_"),
            "mut_swap": bucket_status("mut_swap_"),
            "mut_flip": bucket_status("mut_flip_"),
            "mut_delay": bucket_status("mut_delay_"),
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
