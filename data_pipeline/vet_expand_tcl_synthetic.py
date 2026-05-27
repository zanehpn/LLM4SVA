#!/usr/bin/env python3
"""
vet_expand_tcl_synthetic.py — validate method3 synthetic TCL data with
syntax/TCL/lowering/RTL-context checks.

Outputs under experiments/data/expand_tcl:
  - method3_synthetic_l3_l5_annotated.jsonl
  - method3_synthetic_l3_l5_relaxed.jsonl
  - method3_synthetic_l3_l5_with_rtl.jsonl
  - method3_synthetic_l3_l5_rtl_grounded_strict.jsonl
  - method3_synthetic_l3_l5_vetting_manifest.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.mock_verifier import syntax_check
from src.tcl import classify_tcl
from src.sva_lowering import lower_sva, inject_lowered


DEFAULT_IN = ROOT / "data" / "expand_tcl" / "method3_synthetic_l3_l5.jsonl"
DEFAULT_OUT_DIR = ROOT / "data" / "expand_tcl"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def has_module_context(rtl: str) -> bool:
    rtl = (rtl or "").strip()
    return bool(rtl) and "module" in rtl and "endmodule" in rtl


def summarize(rows: list[dict]) -> dict[str, int]:
    c = Counter(int(r.get("expected_tcl", 0)) for r in rows)
    return {f"L{i}": c.get(i, 0) for i in (1, 2, 3, 4, 5)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(DEFAULT_IN))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    annotated: list[dict] = []
    relaxed: list[dict] = []
    with_rtl: list[dict] = []
    strict: list[dict] = []

    counts = Counter()
    by_transform = Counter()

    with open(in_path) as f:
        for line in f:
            rec = json.loads(line)
            sva = rec.get("reference_sva", "")
            rtl = rec.get("rtl_context", "")
            expected = int(rec.get("expected_tcl", 0))

            syn = syntax_check(sva)
            syn_ok = bool(syn["ok"])

            got_tcl, tcl_reason = classify_tcl(sva)
            tcl_ok = int(got_tcl) == expected

            lo = lower_sva(sva)
            lowering_ok = bool(lo.get("ok"))

            rtl_nonempty = bool((rtl or "").strip())
            rtl_has_module = has_module_context(rtl)

            inject_ok = False
            if lowering_ok and rtl_has_module:
                injected = inject_lowered(rtl, lo["lowered"])
                inject_ok = bool(injected and "endmodule" in injected)

            counts["total"] += 1
            if syn_ok:
                counts["syntax_ok"] += 1
            if tcl_ok:
                counts["tcl_ok"] += 1
            if lowering_ok:
                counts["lowering_ok"] += 1
            if rtl_nonempty:
                counts["rtl_nonempty"] += 1
            if rtl_has_module:
                counts["rtl_has_module"] += 1
            if inject_ok:
                counts["inject_ok"] += 1

            verdict_relaxed = syn_ok and tcl_ok
            verdict_with_rtl = verdict_relaxed and rtl_nonempty
            verdict_strict = verdict_relaxed and rtl_nonempty and rtl_has_module and lowering_ok and inject_ok

            if verdict_relaxed:
                counts["relaxed_kept"] += 1
            if verdict_with_rtl:
                counts["with_rtl_kept"] += 1
            if verdict_strict:
                counts["strict_kept"] += 1

            transform = rec.get("transform", "")
            if verdict_strict:
                by_transform[transform] += 1

            rec_out = dict(rec)
            rec_out.update({
                "syntax_ok": syn_ok,
                "syntax_issues": syn.get("issues", []),
                "syntax_warnings": syn.get("warnings", []),
                "classified_tcl": int(got_tcl),
                "tcl_ok": tcl_ok,
                "tcl_reason": tcl_reason,
                "lowering_ok": lowering_ok,
                "lowering_pattern": lo.get("pattern"),
                "lowering_notes": lo.get("notes", []),
                "rtl_nonempty": rtl_nonempty,
                "rtl_has_module": rtl_has_module,
                "inject_ok": inject_ok,
                "vet_relaxed": verdict_relaxed,
                "vet_with_rtl": verdict_with_rtl,
                "vet_strict": verdict_strict,
            })
            annotated.append(rec_out)
            if verdict_relaxed:
                relaxed.append(rec_out)
            if verdict_with_rtl:
                with_rtl.append(rec_out)
            if verdict_strict:
                strict.append(rec_out)

    write_jsonl(out_dir / "method3_synthetic_l3_l5_annotated.jsonl", annotated)
    write_jsonl(out_dir / "method3_synthetic_l3_l5_relaxed.jsonl", relaxed)
    write_jsonl(out_dir / "method3_synthetic_l3_l5_with_rtl.jsonl", with_rtl)
    write_jsonl(out_dir / "method3_synthetic_l3_l5_rtl_grounded_strict.jsonl", strict)

    manifest = {
        "input": str(in_path),
        "policy": {
            "relaxed": "syntax_ok && tcl_ok",
            "with_rtl": "relaxed && rtl_context is non-empty",
            "strict": "relaxed && rtl_context non-empty && rtl_has_module && lowering_ok && inject_ok",
        },
        "counts": dict(counts),
        "relaxed_tcl": summarize(relaxed),
        "with_rtl_tcl": summarize(with_rtl),
        "strict_tcl": summarize(strict),
        "strict_by_transform": dict(by_transform.most_common()),
    }
    with open(out_dir / "method3_synthetic_l3_l5_vetting_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
