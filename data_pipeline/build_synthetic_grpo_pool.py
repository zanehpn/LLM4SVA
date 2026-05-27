#!/usr/bin/env python3
"""
build_synthetic_grpo_pool.py — reshape master_train_synthetic.jsonl into
the codev_grpo_unified.jsonl schema, synthesizing a minimal valid RTL
context for every row so PEC / yosys-slang can elaborate them.

Output: data/master/master_train_synthetic_grpo.jsonl

Schema matches codev_grpo_unified.jsonl:
    id, source, nl, reference_sva, rtl_context, expected_tcl,
    cex, split, hash, source_name,
    clk, reset, reset_polarity, signals, tb_for_validity,
    temporal_class

RTL synthesis algorithm:
  1. Extract clk / reset signals from the SVA's `@(posedge X)` and
     `disable iff (Y)` clauses. Default clk='clk', reset='tb_reset'.
  2. Extract every other free identifier from the SVA body (filtering
     SystemVerilog keywords + literals).
  3. Emit:

         module synth_top (
             input logic clk,
             input logic reset,
             input logic <signal>,
             ...
         );
             wire tb_reset;
             assign tb_reset = (reset == 1'b1);    // active-high default
         endmodule

     Width-widening: any identifier the SVA compares to a wide literal
     (e.g. `sig == 8'd159`) gets `[31:0]` instead of `logic`.

  4. If an existing rtl_context already has `module` + `endmodule` +
     `tb_reset`, keep it untouched.

The output is a drop-in replacement that PEC + funcatk can elaborate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "data" / "master" / "master_train_synthetic.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "master" / "master_train_synthetic_grpo.jsonl"

# SystemVerilog keywords / built-ins / temporal operators to exclude from
# signal-extraction. Anything in here is NOT declared as an input port.
SV_KEYWORDS = {
    # property / assertion frame
    "assert", "assume", "cover", "property", "endproperty", "sequence",
    "endsequence", "always", "always_ff", "always_comb", "always_latch",
    "if", "else", "begin", "end", "case", "endcase", "default",
    "module", "endmodule", "input", "output", "inout", "wire", "reg",
    "logic", "bit", "byte", "shortint", "int", "longint", "integer",
    "posedge", "negedge", "edge", "disable", "iff", "for", "while",
    # operators / temporal
    "and", "or", "not", "throughout", "within", "intersect", "first_match",
    "until", "until_with", "s_until", "s_eventually", "s_always",
    "nexttime", "strong", "weak", "implies",
    # system functions / methods
    "rose", "fell", "stable", "past", "changed", "sampled",
    "onehot", "onehot0", "countones", "isunknown", "isknown",
    "asserton", "assertkill", "assertoff", "asserton",
    # canonical signals — we'll handle these specially
    "clk", "tb_reset", "rst", "reset", "rst_n", "reset_n", "rst_ni",
    "reset_ni", "clock",
}

# Recognize identifiers (excluding numeric literals)
_ID_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_$]*)\b")
_LITERAL_RE = re.compile(r"\d+'[bdho][0-9a-fA-FxXzZ_]+|\d+")
_CLOCK_RE = re.compile(r"@\s*\(\s*(?:pos|neg)edge\s+([A-Za-z_]\w*)\s*\)", re.I)
_DISABLE_RE = re.compile(r"\bdisable\s+iff\s*\(\s*([^()]+?)\s*\)", re.I)
# Width detection: signal compared to a wide literal (e.g. `sig == 8'd5`)
_WIDTH_HINT_RE = re.compile(
    r"([A-Za-z_]\w*)\s*(?:==|!=|===|!==|>=|<=|>|<)\s*(\d+)'[bdho]"
)
# Bit-select / range hint: `sig[3]` / `sig[5:0]`
_BITSEL_RE = re.compile(r"([A-Za-z_]\w*)\s*\[\s*\d+\s*(?::\s*\d+\s*)?\]")


def extract_clk(sva: str) -> str:
    m = _CLOCK_RE.search(sva)
    return m.group(1) if m else "clk"


def extract_reset_expr(sva: str) -> tuple[str, bool]:
    """Conservative: always default to ('reset', True) and let the
    synthesized RTL build a `wire tb_reset; assign tb_reset = (reset == 1)`
    alias. Reasoning: real SVAs use diverse reset expressions (hierarchical
    refs, OR'd compound conditions) that don't decompose into a single
    port name. Trying to be clever here mis-extracts e.g.
    `cov_assert_if.rst_ni` as `cov_assert_if`. The synthesized module is
    a best-effort wrapper; PEC's reset_expr canonicalization handles the
    actual reset matching.
    """
    return ("reset", True)


def extract_signals(sva: str, clk: str, reset: str) -> dict[str, dict]:
    """Return {signal_name: {'width': int, 'is_bitvec': bool}} for every
    free identifier in the SVA except clk / reset / keywords / literals.

    Width is inferred from comparison literals; bit-select usage forces
    [31:0] declaration.

    Hierarchical refs (`a.b.c`): we declare only the top-level prefix
    (`a`) once. yosys-slang will fail to elaborate `a.b.c` against a flat
    `logic a` declaration, but at least the SVA parser doesn't choke on
    an undeclared top-level identifier.
    """
    # Strip literals first so they don't interfere with identifier scan
    cleaned = _LITERAL_RE.sub(" ", sva)
    # Strip hierarchical suffixes — keep only the leftmost identifier of
    # each dotted reference. So `cov_assert_if.rst_ni` becomes
    # `cov_assert_if` (and the `.rst_ni` part falls out).
    cleaned = re.sub(r"\.[A-Za-z_]\w*", "", cleaned)
    sigs: dict[str, dict] = {}
    excluded = SV_KEYWORDS | {clk, reset, "clk", "tb_reset"}
    for m in _ID_RE.finditer(cleaned):
        name = m.group(1)
        if name.lower() in excluded:
            continue
        if name not in sigs:
            sigs[name] = {"width": 1, "is_bitvec": False}
    # Width hints from `sig == N'b...`
    for m in _WIDTH_HINT_RE.finditer(sva):
        name, width = m.group(1), int(m.group(2))
        if name in sigs and width > sigs[name]["width"]:
            sigs[name]["width"] = width
            sigs[name]["is_bitvec"] = True
    # Bit-select usage forces [31:0]
    for m in _BITSEL_RE.finditer(sva):
        name = m.group(1)
        if name in sigs:
            sigs[name]["is_bitvec"] = True
            if sigs[name]["width"] < 32:
                sigs[name]["width"] = 32
    return sigs


def synthesize_rtl(sva: str, signals: dict[str, dict],
                   clk: str, reset: str, polarity_active_high: bool) -> str:
    """Build a minimal module: clk + reset ports + every free signal as
    free input + the canonical `tb_reset = (reset == 1'b1)` alias."""
    decls = [
        f"    input logic {clk}",
        f"    input logic {reset}",
    ]
    for name, info in sorted(signals.items()):
        if info["is_bitvec"] and info["width"] > 1:
            decls.append(f"    input logic [{info['width']-1}:0] {name}")
        elif info["is_bitvec"]:
            decls.append(f"    input logic [31:0] {name}")
        else:
            decls.append(f"    input logic {name}")
    polarity_op = "1'b1" if polarity_active_high else "1'b0"
    return (
        "module synth_top (\n"
        + ",\n".join(decls)
        + "\n);\n"
        + "    wire tb_reset;\n"
        + f"    assign tb_reset = ({reset} == {polarity_op});\n"
        + "endmodule\n"
    )


def needs_synth(rtl: str) -> bool:
    if not rtl: return True
    if "module" not in rtl: return True
    if "endmodule" not in rtl: return True
    if "tb_reset" not in rtl: return True
    return False


def reshape_row(row: dict) -> dict:
    sva = row.get("reference_sva", "") or ""
    nl = row.get("nl", "") or ""
    rtl = row.get("rtl_context", "") or ""
    tcl = row.get("expected_tcl")
    rid = row.get("id", "")
    h = row.get("hash_canon") or hashlib.md5(sva.encode()).hexdigest()[:16]
    cls = row.get("temporal_class")

    clk = extract_clk(sva)
    reset, polarity = extract_reset_expr(sva)
    signals = extract_signals(sva, clk, reset)

    if needs_synth(rtl):
        rtl = synthesize_rtl(sva, signals, clk, reset, polarity)

    # source_name: derive from id if it's of form "synthetic_l3_..._<hash>"
    src_name = rid
    return {
        "id": rid,
        "source": "synthetic_from_l4",
        "nl": nl,
        "reference_sva": sva,
        "rtl_context": rtl,
        "expected_tcl": tcl,
        "cex": None,
        "split": "train",
        "hash": h,
        "source_name": src_name,
        "clk": clk,
        "reset": reset,
        "reset_polarity": polarity,
        "signals": sorted(signals.keys()),
        "tb_for_validity": "",
        "temporal_class": cls,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(DEFAULT_INPUT))
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = ap.parse_args()

    src = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    n = n_synthd = n_kept = 0
    with open(src) as fin, open(out, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line: continue
            row = json.loads(line)
            n += 1
            had_good_rtl = not needs_synth(row.get("rtl_context", "") or "")
            new_row = reshape_row(row)
            if had_good_rtl:
                n_kept += 1
            else:
                n_synthd += 1
            fout.write(json.dumps(new_row, ensure_ascii=False) + "\n")

    print(f"[done] reshaped {n} rows")
    print(f"  kept original rtl_context: {n_kept} ({100*n_kept/n:.1f}%)")
    print(f"  synthesized new rtl_context: {n_synthd} ({100*n_synthd/n:.1f}%)")
    print(f"  output: {out}")


if __name__ == "__main__":
    main()
