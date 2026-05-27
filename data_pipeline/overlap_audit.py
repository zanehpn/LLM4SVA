"""overlap_audit.py — five-key train/test overlap audit (paper App. G).

For every (train_jsonl, eval_jsonl) pair, reports the overlap count under
five normalization-invariant keys:

  1. Row hash           — the released row hash field (or sha256 of the
                          whole JSON line as a fallback).
  2. Reference-SVA body — whitespace/comment-normalized SVA text.
  3. RTL body           — whitespace/comment-normalized RTL module text.
  4. Verilog module name — every `module <name>` extracted from RTL body.
  5. Exact normalized NL — whitespace-collapsed NL text.

The output is a JSON manifest that can be cited directly in App. G's
overlap table. We expect zero overlap on every key; any non-zero count
means the released split has changed.

Usage:
  python data_pipeline/overlap_audit.py \\
      --train data/train/codev_sft_unified.jsonl \\
      --eval-human data/test/nl2sva_human.jsonl \\
      --eval-machine data/test/nl2sva_machine.jsonl \\
      --output results/overlap_audit.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


_WS = re.compile(r"\s+")
_LINE_COMMENT = re.compile(r"//[^\n]*")
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_MODULE = re.compile(r"\bmodule\s+([A-Za-z_]\w*)")


def _strip_comments(s: str) -> str:
    s = _BLOCK_COMMENT.sub(" ", s or "")
    s = _LINE_COMMENT.sub(" ", s)
    return s


def normalize_text(s: str) -> str:
    """Whitespace + comment normalize. Idempotent."""
    s = _strip_comments(s)
    return _WS.sub(" ", s).strip()


def hash_text(s: str) -> str:
    """sha256 over the normalized body — first 16 hex chars."""
    n = normalize_text(s)
    return hashlib.sha256(n.encode("utf-8")).hexdigest()[:16] if n else ""


def row_hash(row: dict) -> str:
    """Released row hash if present; else sha256 over (nl, sva, rtl)."""
    for k in ("hash", "row_hash", "id"):
        v = row.get(k)
        if isinstance(v, str) and v:
            return v
    payload = json.dumps(
        {"nl": row.get("nl", ""),
         "sva": row.get("reference_sva") or row.get("sva", ""),
         "rtl": row.get("rtl_context") or row.get("rtl", "")},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def extract_module_names(rtl: str) -> set:
    return set(_MODULE.findall(_strip_comments(rtl or "")))


def load_keys(path: Path) -> dict:
    """Return five sets of keys from a JSONL file."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    row_hashes  = set()
    sva_hashes  = set()
    rtl_hashes  = set()
    module_set  = set()
    nl_set      = set()
    for r in rows:
        row_hashes.add(row_hash(r))
        sva = r.get("reference_sva") or r.get("sva") or ""
        if sva:
            sva_hashes.add(hash_text(sva))
        rtl = r.get("rtl_context") or r.get("rtl") or ""
        if rtl:
            rtl_hashes.add(hash_text(rtl))
            module_set.update(extract_module_names(rtl))
        nl = r.get("nl") or ""
        if nl:
            nl_set.add(normalize_text(nl))
    return {
        "n_rows":   len(rows),
        "row_hash": row_hashes,
        "sva_body": sva_hashes,
        "rtl_body": rtl_hashes,
        "module":   module_set,
        "nl_exact": nl_set,
    }


def audit_pair(train_keys: dict, eval_keys: dict) -> dict:
    """Return one row of the overlap table for a (train, eval) pair."""
    return {
        "eval_rows":     eval_keys["n_rows"],
        "row_hash":      len(train_keys["row_hash"]  & eval_keys["row_hash"]),
        "sva_body":      len(train_keys["sva_body"]  & eval_keys["sva_body"]),
        "rtl_body":      len(train_keys["rtl_body"]  & eval_keys["rtl_body"]),
        "module_name":   len(train_keys["module"]    & eval_keys["module"]),
        "nl_exact":      len(train_keys["nl_exact"]  & eval_keys["nl_exact"]),
    }


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, type=Path,
                    help="training JSONL pool (e.g. CodeV-SVA 81,640-row split)")
    ap.add_argument("--eval-human",   type=Path,
                    default=ROOT / "data" / "test" / "nl2sva_human.jsonl")
    ap.add_argument("--eval-machine", type=Path,
                    default=ROOT / "data" / "test" / "nl2sva_machine.jsonl")
    ap.add_argument("--output", type=Path, required=True,
                    help="audit JSON path; gets `train_rows`, per-benchmark "
                         "5-key overlap counts, and a `clean=true|false` flag.")
    args = ap.parse_args(argv)

    print(f"[overlap-audit] loading train: {args.train}")
    train_keys = load_keys(args.train)

    results = {"train": str(args.train),
               "train_rows": train_keys["n_rows"],
               "benchmarks": {}}

    for label, path in (("nl2sva_human",   args.eval_human),
                        ("nl2sva_machine", args.eval_machine)):
        if not path.exists():
            print(f"[overlap-audit] skip {label}: {path} not found")
            continue
        print(f"[overlap-audit] loading eval: {path}")
        ek = load_keys(path)
        row = audit_pair(train_keys, ek)
        print(f"  {label}: rows={row['eval_rows']:>4} "
              f"row_hash={row['row_hash']:>3} sva={row['sva_body']:>3} "
              f"rtl={row['rtl_body']:>3} module={row['module_name']:>3} "
              f"nl={row['nl_exact']:>3}")
        results["benchmarks"][label] = row

    overlap_total = sum(
        b["row_hash"] + b["sva_body"] + b["rtl_body"]
        + b["module_name"] + b["nl_exact"]
        for b in results["benchmarks"].values()
    )
    results["clean"] = (overlap_total == 0)
    results["overlap_total"] = overlap_total

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print(f"[overlap-audit] wrote {args.output}  clean={results['clean']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
