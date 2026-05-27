#!/usr/bin/env python3
"""
build_codev_sft_grpo.py — convert CodeV-SVA 83K structured data into the
repo's unified schema, then split it into body-hash-disjoint SFT / GRPO pools.

Input (default):
  experiments/data/CodeV-SVA-datasets/CodeV-SVA-dataset-83K.jsonl

Outputs:
  experiments/data/CodeV-SVA-datasets/sft/codev_sft_unified.jsonl
  experiments/data/CodeV-SVA-datasets/sft/codev_sft_unified_L{1..5}.jsonl
  experiments/data/CodeV-SVA-datasets/grpo/codev_grpo_unified.jsonl
  experiments/data/CodeV-SVA-datasets/grpo/codev_grpo_unified_L{1..5}.jsonl
  experiments/data/CodeV-SVA-datasets/codev_split_manifest.json

Schema per record:
  {
    "id": "...",
    "source": "codev_sva_83k",
    "nl": "...",
    "reference_sva": "...",
    "rtl_context": "...",
    "expected_tcl": 1..5,
    "cex": null,
    "split": "train",
    "hash": "...",
    "source_name": "...",
    "clk": "...",
    "reset": "...",
    "reset_polarity": true/false,
    "signals": [...],
    "tb_for_validity": "..."
  }
"""

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXPERIMENTS_DIR))
DATASET_DIR = EXPERIMENTS_DIR / "data" / "CodeV-SVA-datasets"
DEFAULT_IN = DATASET_DIR / "CodeV-SVA-dataset-83K.jsonl"
DEFAULT_SFT_DIR = DATASET_DIR / "sft"
DEFAULT_GRPO_DIR = DATASET_DIR / "grpo"
DEFAULT_MANIFEST = DATASET_DIR / "codev_split_manifest.json"
TEST_DIR = EXPERIMENTS_DIR / "data" / "test"


def body_hash(sva: str) -> str:
    norm = re.sub(r"\s+", " ", (sva or "")).strip()
    return hashlib.sha256(norm.encode()).hexdigest()[:16]


def stable_bucket(hash16: str, modulo: int = 1000) -> int:
    return int(hash16, 16) % modulo


def normalize_sva(sva: str) -> str:
    return re.sub(r"\s+", " ", (sva or "")).strip()


def is_usable_nl(nl: str) -> bool:
    nl = (nl or "").strip()
    return len(nl) >= 8


def load_test_hashes() -> set[str]:
    out = set()
    for fname in ("nl2sva_human.jsonl", "assertionbench.jsonl"):
        path = TEST_DIR / fname
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                out.add(body_hash(rec.get("reference_sva", "")))
    return out


def classify_tcl(sva: str) -> int:
    from src.tcl_classifier import TCLClassifier
    clf = TCLClassifier()
    level, _ = clf.classify(sva)
    return int(level)


def convert_record(rec: dict) -> dict | None:
    nl = (rec.get("specification") or "").strip()
    sva = normalize_sva(rec.get("sva") or "")
    rtl = (rec.get("rtl_code") or "").strip()
    if not sva or not is_usable_nl(nl):
        return None
    bh = body_hash(sva)
    out = {
        "id": f"codev83k_{rec.get('name', bh)}",
        "source": "codev_sva_83k",
        "nl": nl,
        "reference_sva": sva,
        "rtl_context": rtl,
        "expected_tcl": classify_tcl(sva),
        "cex": None,
        "split": "train",
        "hash": bh,
        "source_name": rec.get("name", ""),
        "clk": rec.get("clk", ""),
        "reset": rec.get("reset", ""),
        "reset_polarity": rec.get("reset_polarity"),
        "signals": rec.get("signals", []),
        "tb_for_validity": rec.get("tb_for_validity", ""),
    }
    return out


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def write_per_tcl(out_dir: Path, prefix: str, rows: list[dict]) -> dict[str, int]:
    per_tcl = Counter(int(r.get("expected_tcl", 0)) for r in rows)
    for lvl in (1, 2, 3, 4, 5):
        shard = [r for r in rows if int(r.get("expected_tcl", 0)) == lvl]
        write_jsonl(out_dir / f"{prefix}_L{lvl}.jsonl", shard)
    return {f"L{lvl}": per_tcl.get(lvl, 0) for lvl in (1, 2, 3, 4, 5)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(DEFAULT_IN))
    ap.add_argument("--sft-dir", default=str(DEFAULT_SFT_DIR))
    ap.add_argument("--grpo-dir", default=str(DEFAULT_GRPO_DIR))
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--grpo-ratio", type=float, default=0.2,
                    help="fraction of deduped records assigned to GRPO")
    args = ap.parse_args()

    in_path = Path(args.input)
    sft_dir = Path(args.sft_dir)
    grpo_dir = Path(args.grpo_dir)
    manifest_path = Path(args.manifest)

    if not in_path.exists():
        raise SystemExit(f"missing input: {in_path}")
    if not (0.0 < args.grpo_ratio < 1.0):
        raise SystemExit("--grpo-ratio must be in (0,1)")

    test_hashes = load_test_hashes()
    print(f"[guard] loaded {len(test_hashes)} test body hashes")

    raw_total = 0
    drop_no_nl = 0
    drop_no_sva = 0
    drop_test = 0
    drop_dup = 0
    by_hash: dict[str, dict] = {}

    with open(in_path) as f:
        for line in f:
            raw_total += 1
            rec = json.loads(line)
            if not (rec.get("sva") or "").strip():
                drop_no_sva += 1
                continue
            if not is_usable_nl(rec.get("specification", "")):
                drop_no_nl += 1
                continue

            converted = convert_record(rec)
            if converted is None:
                continue
            bh = converted["hash"]
            if bh in test_hashes:
                drop_test += 1
                continue
            if bh in by_hash:
                drop_dup += 1
                continue
            by_hash[bh] = converted

    rows = sorted(by_hash.values(), key=lambda r: r["id"])
    bucket_cutoff = int(args.grpo_ratio * 1000)
    sft_rows = []
    grpo_rows = []
    for row in rows:
        if stable_bucket(row["hash"]) < bucket_cutoff:
            grpo_rows.append(row)
        else:
            sft_rows.append(row)

    # Safety: ensure disjoint by hash.
    sft_hashes = {r["hash"] for r in sft_rows}
    grpo_hashes = {r["hash"] for r in grpo_rows}
    overlap = sft_hashes & grpo_hashes
    if overlap:
        raise SystemExit(f"disjoint split failed: {len(overlap)} overlapping hashes")

    sft_main = sft_dir / "codev_sft_unified.jsonl"
    grpo_main = grpo_dir / "codev_grpo_unified.jsonl"
    write_jsonl(sft_main, sft_rows)
    write_jsonl(grpo_main, grpo_rows)
    sft_tcl = write_per_tcl(sft_dir, "codev_sft_unified", sft_rows)
    grpo_tcl = write_per_tcl(grpo_dir, "codev_grpo_unified", grpo_rows)

    manifest = {
        "input": str(in_path),
        "policy": "CodeV 83K structured dataset converted to unified schema, "
                  "deduped by normalized SVA body hash, test overlap removed, "
                  "then split into body-hash-disjoint SFT / GRPO pools by "
                  "deterministic hash bucket.",
        "source": "codev_sva_83k",
        "raw_total": raw_total,
        "unique_kept": len(rows),
        "drops": {
            "no_nl": drop_no_nl,
            "no_sva": drop_no_sva,
            "test_overlap": drop_test,
            "duplicate_body": drop_dup,
        },
        "split": {
            "grpo_ratio": args.grpo_ratio,
            "sft_size": len(sft_rows),
            "grpo_size": len(grpo_rows),
            "intersection_check": "PASS",
        },
        "sft_per_tcl": sft_tcl,
        "grpo_per_tcl": grpo_tcl,
        "outputs": {
            "sft_main": str(sft_main),
            "grpo_main": str(grpo_main),
            "sft_dir": str(sft_dir),
            "grpo_dir": str(grpo_dir),
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[convert] raw={raw_total} kept={len(rows)}")
    print(f"[drops] no_nl={drop_no_nl} no_sva={drop_no_sva} "
          f"test_overlap={drop_test} dup={drop_dup}")
    print(f"[split] sft={len(sft_rows)} grpo={len(grpo_rows)}")
    print("[sft_tcl] " + " ".join(f"{k}={v}" for k, v in sft_tcl.items()))
    print("[grpo_tcl] " + " ".join(f"{k}={v}" for k, v in grpo_tcl.items()))
    print(f"[write] manifest={manifest_path}")


if __name__ == "__main__":
    main()
