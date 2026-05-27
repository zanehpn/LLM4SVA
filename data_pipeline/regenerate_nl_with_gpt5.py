#!/usr/bin/env python3
"""
regenerate_nl_with_gpt5.py — replace the `nl` field of every row in
master_train.jsonl by an LLM-generated description that mirrors the
nl2sva_human style ("Create a SVA assertion that checks: ...").

Inputs:
  - reference_sva  : the assertion the NL should describe
  - rtl_context    : a (possibly long) RTL blob; we use only signal-decl
                     hints, not the full file, to keep prompts short
  - few-shot       : a fixed bundle of 3-4 nl2sva_human examples for style

Output: a sibling jsonl with a new `nl` field plus `nl_origin = "gpt5_*"`.
The original `nl` is preserved as `nl_original` for audit.

Run with --limit for a pilot. Resumable via --resume which skips ids
already present in the output file.

Usage:
    # pilot 50 rows
    python scripts/regenerate_nl_with_gpt5.py --input data/master/master_train.jsonl \
        --output data/master/master_train.nl_regen.jsonl \
        --limit 50 --workers 8

    # full run
    python scripts/regenerate_nl_with_gpt5.py --input data/master/master_train.jsonl \
        --output data/master/master_train.nl_regen.jsonl \
        --workers 16 --resume
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

# Reach api_function.py
sys.path.insert(0, "${SVA_CORPUS_ROOT}/Local_agent")
from api_function import gpt5


# ---------------------------------------------------------------------------
# Prompt building — keep RTL context bounded so we don't waste tokens.
# ---------------------------------------------------------------------------
SVA_TOKEN_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_$]*)\b")
SVA_KW = {
    "assert", "property", "posedge", "negedge", "disable", "iff", "if", "else",
    "begin", "end", "always", "always_ff", "always_comb", "logic", "wire", "reg",
    "module", "endmodule", "input", "output", "and", "or", "not",
    "throughout", "within", "intersect", "first_match", "until", "until_with",
    "s_until", "s_eventually", "s_always", "nexttime", "strong", "weak",
    "rose", "fell", "stable", "past", "changed", "sampled",
    "onehot", "onehot0", "countones", "isunknown", "isknown",
}


def extract_sva_signals(sva: str) -> list:
    seen = []
    excluded = SVA_KW | {"clk", "tb_reset", "rst", "reset", "rst_n", "reset_n"}
    cleaned = re.sub(r"\d+'[bdho][0-9a-fA-FxXzZ_]+|\d+", " ", sva)
    cleaned = re.sub(r"\.[A-Za-z_]\w*", "", cleaned)  # drop hierarchical
    for tok in SVA_TOKEN_RE.findall(cleaned):
        if tok.lower() in excluded:
            continue
        if tok not in seen:
            seen.append(tok)
    return seen


def trim_rtl(rtl: str, max_chars: int = 600) -> str:
    """Pull just the module declaration head (port list) so the prompt
    stays small. If it's already small we keep it whole."""
    if not rtl: return ""
    if len(rtl) <= max_chars: return rtl
    # First module's signature
    m = re.search(r"module\s+\w+[^;]*;", rtl)
    if m:
        head = rtl[m.start(): min(m.end() + 200, len(rtl))]
        return head[:max_chars]
    return rtl[:max_chars]


FEW_SHOT = """\
Examples (target style):

SVA: assert property (@(posedge clk) disable iff (tb_reset) (fifo_full && wr_push) !== 1'b1);
NL: Create a SVA assertion that checks: that the FIFO does not overflow, assuming no bypass. Use the signals 'wr_push' and 'fifo_full'.

SVA: assert property (@(posedge clk) disable iff (tb_reset) !fifo_empty |-> strong(##[0:$] rd_pop));
NL: Create a SVA assertion that checks: that when response is pending, data is eventually popped from the FIFO. Use the signals 'rd_pop' and 'fifo_empty'.

SVA: assert property (@(posedge clk) disable iff (tb_reset) (req && !ack) |=> ##3 ack);
NL: Create a SVA assertion that checks: that whenever a request is asserted without an acknowledgement, the acknowledgement must be asserted exactly three cycles later. Use the signals 'req' and 'ack'.
"""


def build_prompt(sva: str, rtl: str) -> str:
    sigs = extract_sva_signals(sva)
    sig_hint = ", ".join(f"'{s}'" for s in sigs[:8]) if sigs else "(none extracted)"
    rtl_trim = trim_rtl(rtl, max_chars=500)
    rtl_block = f"\n\nRTL signal context (truncated):\n{rtl_trim}\n" if rtl_trim else ""
    return f"""\
Generate a one-sentence natural-language description of what the following \
SystemVerilog assertion checks. The description must:

(a) Start exactly with: "Create a SVA assertion that checks: "
(b) Describe the *intent* of the property in plain engineering English \
(no SVA syntax tokens like |->, ##, $rose, etc.).
(c) End with "Use the signal(s) '<name>' [and '<name>']." listing the \
non-clock, non-reset identifiers from the SVA.
(d) Be 25-200 characters total.

{FEW_SHOT}
SVA to describe:
{sva}
{rtl_block}
Signal hint: {sig_hint}

Output ONLY the NL sentence. No markdown, no explanation, no quotes.
"""


# ---------------------------------------------------------------------------
# Generation worker
# ---------------------------------------------------------------------------
SVA_TOKEN_LEAK_RE = re.compile(
    r"\|->|\|=>|##\s*\[|##\s*\d|s_eventually|s_until|s_always|"
    r"\bassert\s+property\b|disable\s+iff|posedge|negedge",
    re.I,
)


def validate_nl(nl: str) -> tuple[bool, str]:
    nl = (nl or "").strip().strip('"').strip("'")
    if not nl: return False, "empty"
    if not nl.lower().startswith("create a sva assertion"):
        return False, "missing_prefix"
    if len(nl) < 25 or len(nl) > 400:
        return False, f"length={len(nl)}"
    if SVA_TOKEN_LEAK_RE.search(nl):
        return False, "sva_token_leak"
    return True, "ok"


def regen_one(row: dict, retries: int = 2) -> dict:
    sva = row.get("reference_sva", "") or ""
    rtl = row.get("rtl_context", "") or ""
    prompt = build_prompt(sva, rtl)
    last_err = ""
    for attempt in range(retries + 1):
        try:
            raw = gpt5(prompt) or ""
            raw = raw.strip()
            ok, why = validate_nl(raw)
            if ok:
                return {
                    "id": row.get("id"),
                    "nl_new": raw,
                    "nl_original": row.get("nl", ""),
                    "status": "ok",
                    "attempt": attempt,
                }
            last_err = why
        except Exception as e:
            last_err = f"exception:{type(e).__name__}:{str(e)[:80]}"
        time.sleep(0.5 * (attempt + 1))
    return {
        "id": row.get("id"),
        "nl_new": "",
        "nl_original": row.get("nl", ""),
        "status": f"failed:{last_err}",
        "attempt": retries,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=0,
                    help="0 = no limit (process all); else cap")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--resume", action="store_true",
                    help="skip ids already present in output file")
    args = ap.parse_args()

    src = Path(args.input)
    dst = Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)

    done_ids = set()
    if args.resume and dst.exists():
        with open(dst) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["id"])
                except Exception:
                    pass
        print(f"[resume] skipping {len(done_ids)} already-done ids")

    rows = []
    with open(src) as f:
        for line in f:
            r = json.loads(line)
            if r["id"] in done_ids:
                continue
            rows.append(r)
            if args.limit and len(rows) >= args.limit:
                break

    print(f"[plan] {len(rows)} rows to regenerate, workers={args.workers}")
    print(f"[plan] output: {dst}")

    out_lock = Lock()
    out_f = open(dst, "a")
    n_done = n_ok = n_fail = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(regen_one, r): r for r in rows}
        for fut in as_completed(futs):
            res = fut.result()
            with out_lock:
                out_f.write(json.dumps(res, ensure_ascii=False) + "\n")
                out_f.flush()
            n_done += 1
            if res["status"] == "ok": n_ok += 1
            else: n_fail += 1
            if n_done % 25 == 0 or n_done == len(rows):
                elapsed = time.time() - t0
                rate = n_done / elapsed
                print(f"[gen] {n_done}/{len(rows)}  ok={n_ok} fail={n_fail}  "
                      f"{rate:.1f}/s  eta={(len(rows)-n_done)/max(rate,1e-3):.0f}s")
    out_f.close()
    print(f"\n[done] ok={n_ok}  fail={n_fail}  total={n_done}  time={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
