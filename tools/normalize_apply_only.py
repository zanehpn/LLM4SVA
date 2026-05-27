"""normalize_apply_only.py
Apply `normalize_sva` from normalize_sva_for_verilator.py unconditionally
to every row's reference_sva and write a new jsonl. No verifier check —
faster than normalize_sva_for_verilator.py for the case where we just
want a normalized pool to feed downstream (VCS bench, etc.)."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data_pipeline"))
from normalize_sva_for_verilator import normalize_sva


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    n = 0; n_changed = 0
    with open(args.input) as fi, open(args.output, "w") as fo:
        for ln in fi:
            ln = ln.strip()
            if not ln: continue
            r = json.loads(ln)
            orig = r.get("reference_sva", "")
            norm = normalize_sva(orig)
            if norm != orig:
                r["reference_sva_orig"] = orig
                r["reference_sva"] = norm
                r["_sva_normalized"] = True
                n_changed += 1
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    print(f"[normalize-apply] rows={n}  changed={n_changed} ({100*n_changed/max(n,1):.1f}%)")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
