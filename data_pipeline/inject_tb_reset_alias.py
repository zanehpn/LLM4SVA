#!/usr/bin/env python3
"""
inject_tb_reset_alias.py — codev_grpo_unified / codev_sft_unified store
SVAs that gate on `tb_reset` but their rtl_context only declares the raw
`reset` port. yosys-slang fails to elaborate, PEC returns PARSE_ERROR
on every candidate. Fix: inject the canonical alias

    wire tb_reset;
    assign tb_reset = (reset == 1'b1);   // when reset_polarity == True
    assign tb_reset = (reset == 1'b0);   // when reset_polarity == False

just before `endmodule`. Same convention as nl2sva_machine.

Idempotent: rows whose rtl_context already declares `tb_reset` are left
alone.

Usage:
    python scripts/inject_tb_reset_alias.py
    # or targeted:
    python scripts/inject_tb_reset_alias.py --inputs <path> ...
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_INPUTS = [
    ROOT / "data" / "CodeV-SVA-datasets" / "grpo" / "codev_grpo_unified.jsonl",
    ROOT / "data" / "CodeV-SVA-datasets" / "grpo" / "codev_grpo_unified_C1.jsonl",
    ROOT / "data" / "CodeV-SVA-datasets" / "grpo" / "codev_grpo_unified_C2.jsonl",
    ROOT / "data" / "CodeV-SVA-datasets" / "grpo" / "codev_grpo_unified_C3.jsonl",
    ROOT / "data" / "CodeV-SVA-datasets" / "sft" / "codev_sft_unified.jsonl",
    ROOT / "data" / "CodeV-SVA-datasets" / "sft" / "codev_sft_unified_C1.jsonl",
    ROOT / "data" / "CodeV-SVA-datasets" / "sft" / "codev_sft_unified_C2.jsonl",
    ROOT / "data" / "CodeV-SVA-datasets" / "sft" / "codev_sft_unified_C3.jsonl",
]


def make_alias(reset_signal: str, reset_polarity: bool) -> str:
    # reset_polarity True = active-high reset (assertion fires when reset==1)
    # reset_polarity False = active-low (assertion fires when reset==0)
    if reset_polarity:
        return (f"\n    wire tb_reset;\n"
                f"    assign tb_reset = ({reset_signal} == 1'b1);\n")
    return (f"\n    wire tb_reset;\n"
            f"    assign tb_reset = ({reset_signal} == 1'b0);\n")


def patch_rtl(rtl: str, reset_signal: str, reset_polarity: bool) -> str:
    """Insert a tb_reset alias just before the LAST `endmodule`."""
    if "tb_reset" in rtl:
        return rtl  # idempotent
    idx = rtl.rfind("endmodule")
    if idx < 0:
        return rtl  # malformed, leave alone
    prefix = rtl[:idx].rstrip()
    suffix = rtl[idx:]
    return prefix + make_alias(reset_signal, reset_polarity) + suffix


def annotate_in_place(src: Path):
    tmp = src.with_suffix(src.suffix + ".tmp")
    n_rows = n_patched = n_already = n_no_endmodule = 0
    with open(src) as fin, open(tmp, "w") as fout:
        for line in fin:
            line = line.rstrip("\n")
            if not line.strip():
                fout.write(line + "\n")
                continue
            row = json.loads(line)
            n_rows += 1
            rtl = row.get("rtl_context", "") or ""
            reset_sig = row.get("reset", "reset")
            polarity = bool(row.get("reset_polarity", True))
            if "tb_reset" in rtl:
                n_already += 1
            elif "endmodule" not in rtl:
                n_no_endmodule += 1
            else:
                row["rtl_context"] = patch_rtl(rtl, reset_sig, polarity)
                n_patched += 1
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, src)
    return {
        "n_rows": n_rows,
        "n_patched": n_patched,
        "n_already_had_tb_reset": n_already,
        "n_no_endmodule": n_no_endmodule,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="*", default=[str(p) for p in DEFAULT_INPUTS])
    args = ap.parse_args()

    for path in args.inputs:
        src = Path(path)
        if not src.exists():
            print(f"[skip] missing: {src}")
            continue
        print(f"\n[patch] {src.name}")
        stats = annotate_in_place(src)
        for k, v in stats.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
