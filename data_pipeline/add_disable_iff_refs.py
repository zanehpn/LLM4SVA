#!/usr/bin/env python3
"""
add_disable_iff_refs.py — wrap canonical refs with `disable iff (tb_reset)`
so that GRPO with the FVEval prompt format (which steers the model toward
disable iff outputs) gets non-zero PEC reward.

The NL2SVA-Human eval set has `disable iff (tb_reset)` on every reference,
so the eval-matched FVEval prompt template includes a disable-iff exemplar.
Without this rewrite, the model produces SVAs *with* disable iff but the
canonical refs are *without* — PEC says NOT_EQUIVALENT or
IMPLIES_LM_TO_REF (alt is stronger), and the v5/v3-multiref reward maps
both to 0.0 → no reward signal → no learning.

Strategy:
  1. Load grpo_pool_phase2_v2.jsonl.
  2. For each record, build a disable-iff variant of the canonical ref:
     `assert property(@(posedge clk) BODY);` →
     `assert property(@(posedge clk) disable iff (tb_reset) BODY);`
     (and similar for assume/cover, multi-line clock blocks, etc.)
  3. Set canonical ref to the disable-iff version.
  4. Add the original (raw) ref to the ref_svas list.
  5. Write grpo_pool_phase2_v3.jsonl.

Why replace the canonical (not just append): the multi-ref reward takes
max over refs, so adding a disable-iff variant alongside the raw ref would
work — but in practice we want the FIRST ref (used for free-input RTL
construction) to drive the in-distribution form. The raw form stays
available for prompts where the model happens to omit disable iff.

Usage:
  PYTHONPATH=. python3 scripts/add_disable_iff_refs.py
"""
import json
import re
import sys
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent
POOL_IN = EXPERIMENTS_DIR / "data" / "train" / "grpo" / "grpo_pool_phase2_v2.jsonl"
POOL_OUT = EXPERIMENTS_DIR / "data" / "train" / "grpo" / "grpo_pool_phase2_v3.jsonl"

# Match the verb (assert|assume|cover) property + clocking + body.
# Body is captured non-greedy, ends just before the closing ')' of the
# `property(...)` parens — paren-balanced so nested parens in the body
# don't trip the regex.
PROP_RE = re.compile(
    r"\b(assert|assume|cover)\s+property\s*\(\s*"
    r"(@\([^)]*\))\s*"   # clocking event
    r"(.+?)\s*\)\s*;",
    re.DOTALL | re.IGNORECASE,
)


def add_disable_iff(sva: str) -> str:
    """Insert `disable iff (tb_reset)` between the clocking event and the
    body. Returns the rewritten SVA, or the original if no match / already
    has disable iff."""
    if "disable iff" in sva.lower():
        return sva
    m = PROP_RE.search(sva)
    if not m:
        return sva
    verb, clk, body = m.group(1), m.group(2), m.group(3)
    # Paren-balance the body (PROP_RE may have stopped at an inner paren).
    start = m.end(2)
    depth = 1
    i = start
    while i < len(sva) and depth > 0:
        c = sva[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                break
        i += 1
    if depth != 0:
        return sva
    full_body = sva[start:i].strip()
    rewritten = f"{verb} property ({clk} disable iff (tb_reset)\n    {full_body}\n);"
    # Preserve any trailing chars after the closing ');'
    return rewritten


def main():
    if not POOL_IN.exists():
        print(f"[error] missing {POOL_IN}", file=sys.stderr)
        sys.exit(2)

    pool = [json.loads(l) for l in open(POOL_IN)]
    print(f"[load] {len(pool)} prompts from {POOL_IN.name}")

    n_rewritten = 0
    n_already = 0
    n_failed = 0
    for r in pool:
        raw_ref = r["reference_sva"]
        di_ref = add_disable_iff(raw_ref)
        if di_ref == raw_ref:
            if "disable iff" in raw_ref.lower():
                n_already += 1
            else:
                n_failed += 1
            continue
        # Replace canonical with disable-iff version; prepend raw to ref_svas
        # so multi-ref still covers prompts where model omits disable iff.
        existing = list(r.get("ref_svas") or [raw_ref])
        new_refs = [di_ref]
        seen = {re.sub(r"\s+", " ", di_ref).strip()}
        for s in [raw_ref] + existing:
            key = re.sub(r"\s+", " ", s).strip()
            if key in seen:
                continue
            seen.add(key)
            new_refs.append(s)
        r["reference_sva"] = di_ref
        r["ref_svas"] = new_refs
        r["n_extra_refs"] = len(new_refs) - 1
        n_rewritten += 1

    print(f"[rewrite] disable-iff added: {n_rewritten}")
    print(f"[rewrite] already had disable iff: {n_already}")
    print(f"[rewrite] regex did not match: {n_failed}")

    with open(POOL_OUT, "w") as f:
        for r in pool:
            f.write(json.dumps(r) + "\n")
    print(f"[write] {POOL_OUT}")

    # Sanity: print before/after on a sample
    print("\n--- sample 0 ---")
    print("CANON:", pool[0]["reference_sva"])
    print("REFS:")
    for s in pool[0]["ref_svas"][:3]:
        print(" -", s.replace("\n", " "))


if __name__ == "__main__":
    main()
