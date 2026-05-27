"""normalize_master_disable_iff.py
Rewrite SVA + RTL pair so that any `disable iff (E)` becomes
`disable iff (tb_reset)` plus a `wire tb_reset; assign tb_reset = E;` aliased
into the RTL just before the last `endmodule`. Semantics-preserving.

Goal: align master_train.jsonl with the nl2sva_human canonical form
(`disable iff (tb_reset)` 100%) so SFT does not punish the model into
emitting non-canonical reset names.

Skips rows whose SVA has no `disable iff` clause (those align with
nl2sva_machine, which has 0% disable iff — leaving them alone preserves
that distribution).

Usage:
    python scripts/normalize_master_disable_iff.py \
        --input data/master/master_train.jsonl \
        --output data/master/master_train_norm.jsonl
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def find_balanced_paren(s: str, open_idx: int) -> int:
    """Given s and the index of an opening '(', return the index of the
    matching ')'. Returns -1 if unbalanced."""
    depth = 0
    for i in range(open_idx, len(s)):
        ch = s[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i
    return -1


# Catches "disable iff" with optional whitespace before the opening paren.
DI_KW_RE = re.compile(r"\bdisable\s+iff\s*\(", re.IGNORECASE)


def rewrite_sva_and_extract(sva: str):
    """Replace every `disable iff (E)` in `sva` with `disable iff (tb_reset)`
    and return (new_sva, list_of_E_strings_in_order). E is stripped of
    leading/trailing whitespace.

    If the original SVA already uses tb_reset literally, that occurrence is
    not extracted (no rewrite needed)."""
    out_parts = []
    cursor = 0
    extracted = []
    for m in DI_KW_RE.finditer(sva):
        kw_start = m.start()
        open_paren = m.end() - 1   # the '(' is the last char of the match
        close_paren = find_balanced_paren(sva, open_paren)
        if close_paren < 0:
            # malformed — bail, return original
            return sva, []
        out_parts.append(sva[cursor:kw_start])
        expr = sva[open_paren + 1: close_paren].strip()
        if expr == "tb_reset":
            # already canonical
            out_parts.append(sva[kw_start: close_paren + 1])
        else:
            out_parts.append("disable iff (tb_reset)")
            extracted.append(expr)
        cursor = close_paren + 1
    out_parts.append(sva[cursor:])
    new_sva = "".join(out_parts)
    return new_sva, extracted


def patch_rtl_with_tb_reset(rtl: str, reset_expr: str) -> str:
    """Insert a tb_reset alias defining `tb_reset = reset_expr` just before
    the last `endmodule`. Idempotent: if rtl already declares tb_reset,
    leave it untouched."""
    if "tb_reset" in rtl:
        return rtl
    idx = rtl.rfind("endmodule")
    if idx < 0:
        return rtl
    prefix = rtl[:idx].rstrip()
    suffix = rtl[idx:]
    inject = (f"\n    wire tb_reset;\n"
              f"    assign tb_reset = ({reset_expr});\n")
    return prefix + inject + suffix


def normalize_row(row: dict) -> tuple[dict, str]:
    """Returns (new_row, status). Status is one of:
        - 'rewritten'        : SVA + RTL got patched
        - 'already_canonical': SVA already used disable iff (tb_reset)
        - 'no_disable_iff'   : SVA has no disable iff clause; left alone
        - 'malformed'        : SVA had unbalanced parens; left alone
        - 'multi_distinct'   : SVA had multiple disable iff with different
                                 exprs; rewrote SVA but only first expr
                                 used for tb_reset alias (rare in practice)
    """
    sva = row.get("reference_sva") or ""
    rtl = row.get("rtl_context") or ""

    if not sva.strip():
        return row, "no_disable_iff"

    # Quick path: no disable iff at all
    if not DI_KW_RE.search(sva):
        return row, "no_disable_iff"

    new_sva, extracted = rewrite_sva_and_extract(sva)
    if not extracted and new_sva == sva:
        # already canonical (every disable iff was already tb_reset)
        return row, "already_canonical"
    if not extracted and new_sva != sva:
        # shouldn't happen
        return row, "malformed"
    if new_sva == sva:
        # rewrite_sva returned original due to unbalanced paren
        return row, "malformed"

    distinct = set(extracted)
    status = "multi_distinct" if len(distinct) > 1 else "rewritten"
    # use the first extracted expression for the alias; if multi_distinct,
    # this loses the tail — caller can filter.
    reset_expr = extracted[0]
    new_rtl = patch_rtl_with_tb_reset(rtl, reset_expr)

    new_row = dict(row)
    new_row["reference_sva"] = new_sva
    new_row["rtl_context"] = new_rtl
    new_row["_normalize"] = {
        "original_reset_exprs": extracted,
        "tb_reset_expr": reset_expr,
    }
    return new_row, status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(ROOT / "data" / "master" / "master_train.jsonl"))
    ap.add_argument("--output", default=str(ROOT / "data" / "master" / "master_train_norm.jsonl"))
    ap.add_argument("--drop-multi-distinct", action="store_true",
                    help="drop rows whose SVA has multiple distinct "
                         "disable iff expressions (rare)")
    ap.add_argument("--drop-no-disable-iff", action="store_true",
                    help="also drop rows that never had `disable iff` "
                         "(useful if you want a strictly-human-aligned "
                         "subset)")
    ap.add_argument("--print-samples", type=int, default=3,
                    help="print N before/after rewrite samples per status")
    args = ap.parse_args()

    src = Path(args.input)
    dst = Path(args.output)
    if not src.exists():
        sys.exit(f"missing input: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)

    status_counts = Counter()
    extracted_counter = Counter()
    samples: dict[str, list] = {}
    n_in = n_out = 0
    with open(src) as fin, open(dst, "w") as fout:
        for line in fin:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            n_in += 1
            row = json.loads(line)
            new_row, status = normalize_row(row)
            status_counts[status] += 1
            if status in ("rewritten", "multi_distinct"):
                for e in new_row["_normalize"]["original_reset_exprs"]:
                    extracted_counter[e] += 1
            samples.setdefault(status, []).append((row, new_row))

            drop = False
            if status == "malformed":
                drop = True
            if args.drop_multi_distinct and status == "multi_distinct":
                drop = True
            if args.drop_no_disable_iff and status == "no_disable_iff":
                drop = True
            if not drop:
                fout.write(json.dumps(new_row, ensure_ascii=False) + "\n")
                n_out += 1

    print(f"\n[normalize_master_disable_iff]")
    print(f"  input : {src}")
    print(f"  output: {dst}")
    print(f"  rows in : {n_in}")
    print(f"  rows out: {n_out}")
    print(f"  status:")
    for s, c in status_counts.most_common():
        pct = 100 * c / max(n_in, 1)
        print(f"    {s:<22} {c:>6}  ({pct:5.1f}%)")
    print(f"  top 15 original reset expressions (rewritten -> tb_reset):")
    for expr, c in extracted_counter.most_common(15):
        print(f"    {c:>5}  {expr!r}")

    if args.print_samples > 0:
        for s in ("rewritten", "multi_distinct", "malformed", "already_canonical"):
            if s not in samples:
                continue
            print(f"\n--- samples [{s}] (showing up to {args.print_samples}) ---")
            for orig, new in samples[s][:args.print_samples]:
                print(f"\n  ID: {orig.get('id')}")
                print(f"  ORIG SVA: {(orig.get('reference_sva') or '')[:240]}")
                print(f"  NEW  SVA: {(new.get('reference_sva') or '')[:240]}")
                if s in ("rewritten", "multi_distinct"):
                    rtl_orig = orig.get("rtl_context") or ""
                    rtl_new = new.get("rtl_context") or ""
                    if "tb_reset" in rtl_new and "tb_reset" not in rtl_orig:
                        # Show the injected fragment
                        idx = rtl_new.rfind("endmodule")
                        if idx > 0:
                            inj_start = max(0, idx - 200)
                            print(f"  NEW RTL  (tail before endmodule):")
                            print("    " + rtl_new[inj_start:idx].replace("\n", "\n    "))


if __name__ == "__main__":
    main()
