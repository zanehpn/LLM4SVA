#!/usr/bin/env python3
"""
build_expand_tcl.py — construct TCL-focused expansion pools under
`experiments/data/expand_tcl` using three routes:

1. Filter existing structured datasets for L3/L5 examples.
2. Mine high-TCL assertions from locally cloned GitHub repos.
3. Programmatically expand L4 assertions into synthetic L3/L5 variants.

Outputs:
  experiments/data/expand_tcl/method1_existing_l3_l5.jsonl
  experiments/data/expand_tcl/method2_github_high_tcl.jsonl
  experiments/data/expand_tcl/method3_synthetic_l3_l5.jsonl
  experiments/data/expand_tcl/all_methods_merged.jsonl
  experiments/data/expand_tcl/manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.tcl import classify_tcl
from scripts.scrape_github_sva import scan_repo, detect_license


OUT_DIR = ROOT / "data" / "expand_tcl"
ALL_MERGED = ROOT / "data" / "train" / "all_merged_train.jsonl"
CODEV_83K = ROOT / "data" / "CodeV-SVA-datasets" / "CodeV-SVA-dataset-83K.jsonl"
GITHUB_REPOS = ROOT / "data" / "raw" / "github_repos"

PLACEHOLDER_RE = re.compile(r"^\s*\[(?:ASSERT|ASSUME|COVER)[^\]]*\]\s*$", re.I)
WS_RE = re.compile(r"\s+")
IMPL_RE = re.compile(r"(\|->|\|=>)")
RANGED_DELAY_RE = re.compile(r"##\s*\[\s*(\d+)\s*:\s*(\d+|\$)\s*\]")
FIXED_DELAY_RE = re.compile(r"##\s*(\d+)")
HIGH_TCL_HINT_RE = re.compile(
    r"##\s*\[|\[\s*(?:\*|=|-)\s*\d+\s*:\s*(?:\d+|\$)\s*\]|"
    r"\bs_eventually\b|\bs_until\b|\bs_always\b|\buntil_with\b|"
    r"\bthroughout\b|\bwithin\b|\bintersect\b|\bfirst_match\b",
    re.I,
)


def normalize_ws(text: str) -> str:
    return WS_RE.sub(" ", (text or "").strip())


def body_hash(sva: str) -> str:
    return hashlib.sha256(normalize_ws(sva).encode()).hexdigest()[:16]


def usable_nl(nl: str) -> bool:
    nl = (nl or "").strip()
    return len(nl) >= 8 and not PLACEHOLDER_RE.match(nl)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def summarize(rows: list[dict]) -> dict[str, int]:
    c = Counter(int(r.get("expected_tcl", 0)) for r in rows)
    return {f"L{i}": c.get(i, 0) for i in (1, 2, 3, 4, 5)}


def extract_module_snippet(rtl: str, top_name: str = "") -> str:
    rtl = (rtl or "").strip()
    if not rtl:
        return ""
    lines = rtl.splitlines()
    if len(lines) <= 80:
        return rtl
    if top_name:
        pat = re.compile(rf"\bmodule\s+{re.escape(top_name)}\b")
        for i, line in enumerate(lines):
            if pat.search(line):
                return "\n".join(lines[max(0, i - 5): i + 75])
    return "\n".join(lines[:80])


def method1_existing() -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()

    if ALL_MERGED.exists():
        with open(ALL_MERGED) as f:
            for line in f:
                rec = json.loads(line)
                nl = (rec.get("nl") or "").strip()
                sva = normalize_ws(rec.get("reference_sva") or rec.get("sva") or "")
                if not usable_nl(nl) or not sva:
                    continue
                level, _ = classify_tcl(sva)
                if level not in (3, 5):
                    continue
                bh = body_hash(sva)
                if bh in seen:
                    continue
                seen.add(bh)
                rows.append({
                    "id": rec.get("id") or f"existing_{bh}",
                    "source": "existing_structured",
                    "origin_dataset": "all_merged_train",
                    "nl": nl,
                    "reference_sva": sva,
                    "rtl_context": rec.get("rtl_context", ""),
                    "expected_tcl": level,
                    "hash": bh,
                    "source_path": str(ALL_MERGED),
                })

    if CODEV_83K.exists():
        with open(CODEV_83K) as f:
            for line in f:
                rec = json.loads(line)
                nl = (rec.get("specification") or "").strip()
                sva = normalize_ws(rec.get("sva") or "")
                if not usable_nl(nl) or not sva:
                    continue
                level, _ = classify_tcl(sva)
                if level not in (3, 5):
                    continue
                bh = body_hash(sva)
                if bh in seen:
                    continue
                seen.add(bh)
                rows.append({
                    "id": f"codev83k_{rec.get('name', bh)}",
                    "source": "existing_structured",
                    "origin_dataset": "CodeV-SVA-dataset-83K",
                    "nl": nl,
                    "reference_sva": sva,
                    "rtl_context": extract_module_snippet(
                        rec.get("rtl_code", ""), rec.get("top_name", "")
                    ),
                    "expected_tcl": level,
                    "hash": bh,
                    "source_path": str(CODEV_83K),
                    "clk": rec.get("clk", ""),
                    "reset": rec.get("reset", ""),
                })
    return sorted(rows, key=lambda r: r["id"])


def heuristic_nl_from_sva(sva: str, module_name: str = "", comment: str = "") -> str:
    comment = normalize_ws(comment)
    if usable_nl(comment):
        return comment
    prefix = f"For module {module_name}, " if module_name else ""
    if re.search(r"\bs_eventually\b", sva, re.I):
        return prefix + "the consequent condition should eventually hold after the trigger."
    if re.search(r"\bs_until\b|\buntil_with\b", sva, re.I):
        return prefix + "the antecedent condition should persist until the terminating condition holds."
    if re.search(r"##\s*\[", sva):
        return prefix + "after the trigger, the consequent should hold within a bounded cycle range."
    if re.search(r"\bthroughout\b", sva, re.I):
        return prefix + "the guarded condition should hold throughout the target sequence."
    if re.search(r"\bwithin\b", sva, re.I):
        return prefix + "the required sequence should complete within the enclosing sequence."
    if re.search(r"\bintersect\b", sva, re.I):
        return prefix + "two sequences should match over the same window."
    return prefix + "generate an SVA matching the extracted temporal property."


def method2_github() -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()
    if not GITHUB_REPOS.exists():
        return rows

    for repo_dir in sorted(GITHUB_REPOS.iterdir()):
        if not repo_dir.is_dir():
            continue
        lic = detect_license(repo_dir)
        for rec in scan_repo(repo_dir, lic):
            sva = normalize_ws(rec.get("sva", ""))
            if not sva or not HIGH_TCL_HINT_RE.search(sva):
                continue
            level, _ = classify_tcl(sva)
            if level not in (3, 5):
                continue
            bh = body_hash(sva)
            if bh in seen:
                continue
            seen.add(bh)
            rows.append({
                "id": f"github_{bh}",
                "source": "github_high_tcl_local",
                "origin_dataset": "data/raw/github_repos",
                "nl": heuristic_nl_from_sva(
                    sva, rec.get("module", ""), rec.get("nl_comment", "")
                ),
                "reference_sva": sva,
                "rtl_context": "",
                "expected_tcl": level,
                "hash": bh,
                "source_repo": rec.get("source_repo", ""),
                "source_file": rec.get("file", ""),
                "source_line": rec.get("line"),
                "module_name": rec.get("module", ""),
                "license": rec.get("license", ""),
                "nl_comment": rec.get("nl_comment", ""),
            })
    return sorted(rows, key=lambda r: (r.get("source_repo", ""), r["id"]))


def rewrite_delay_phrase(nl: str, phrase: str) -> str:
    nl = normalize_ws(nl)
    patterns = [
        r"\bnext cycle\b",
        r"\bwithin\b.+?\bcycles?\b",
        r"\bafter \d+ cycles?\b",
        r"\bwithin a bounded cycle range\b",
    ]
    for pat in patterns:
        if re.search(pat, nl, re.I):
            return re.sub(pat, phrase, nl, flags=re.I)
    return nl.rstrip(".") + f" The consequent should hold {phrase}."


def rewrite_liveness_phrase(nl: str, phrase: str) -> str:
    nl = normalize_ws(nl)
    patterns = [
        r"\beventually\b",
        r"\buntil\b",
        r"\balways\b",
        r"\bnext cycle\b",
    ]
    for pat in patterns:
        if re.search(pat, nl, re.I):
            return re.sub(pat, phrase, nl, flags=re.I)
    return nl.rstrip(".") + f" {phrase[0].upper() + phrase[1:]}."


def synthesize_l3_nl(nl: str, tag: str, phrase: str) -> str:
    nl = normalize_ws(nl)
    if tag == "range_0_1":
        return rewrite_delay_phrase(nl, "within 0 to 1 cycles")
    if tag == "range_1_3":
        return rewrite_delay_phrase(nl, "within 1 to 3 cycles")
    if tag == "range_2_5":
        return rewrite_delay_phrase(nl, "within 2 to 5 cycles")
    if tag == "range_1_inf":
        return rewrite_delay_phrase(nl, "at some point after 1 or more cycles")
    if tag == "lift_fixed_to_range":
        return rewrite_delay_phrase(nl, "within a bounded cycle window")
    if tag == "lift_fixed_to_open_range":
        return rewrite_delay_phrase(nl, "within an open-ended cycle window")
    return rewrite_delay_phrase(nl, phrase)


def synthesize_l5_nl(nl: str, tag: str, phrase: str) -> str:
    nl = normalize_ws(nl)
    if tag == "eventually":
        return rewrite_liveness_phrase(nl, "eventually")
    if tag == "strong_eventually":
        return rewrite_liveness_phrase(
            nl, "must eventually become true and cannot be deferred forever"
        )
    if tag == "until_with":
        return rewrite_liveness_phrase(
            nl, "must remain true until the terminating condition holds"
        )
    if tag == "s_until":
        return rewrite_liveness_phrase(
            nl, "must continue to hold until the consequent becomes true"
        )
    if tag == "s_always":
        return rewrite_liveness_phrase(
            nl, "must always remain true once triggered"
        )
    return rewrite_liveness_phrase(nl, phrase)


def split_implication(sva: str) -> tuple[str, str] | None:
    sva = normalize_ws(sva)
    m = IMPL_RE.search(sva)
    if not m:
        return None
    lhs = sva[:m.start()].strip()
    rhs = sva[m.end():].strip().rstrip(";")
    if not lhs or not rhs:
        return None
    return lhs, rhs


def make_variant_id(prefix: str, bh: str, tag: str) -> str:
    return f"{prefix}_{tag}_{bh}"


def expand_l4_to_l3_variants(sva: str) -> list[tuple[str, str, str]]:
    sva = normalize_ws(sva)
    parts = split_implication(sva)
    if not parts:
        return []
    if re.search(r"\bs_eventually\b|\bs_until\b|\bs_always\b|\buntil_with\b", sva, re.I):
        return []
    lhs, rhs = parts
    suffix = ";" if sva.endswith(";") else ""
    variants: list[tuple[str, str, str]] = []
    templates = [
        ("range_1_3", "within 1 to 3 cycles", f"{lhs} ##[1:3] {rhs}{suffix}"),
        ("range_0_1", "within 0 to 1 cycles", f"{lhs} ##[0:1] {rhs}{suffix}"),
        ("range_2_5", "within 2 to 5 cycles", f"{lhs} ##[2:5] {rhs}{suffix}"),
        ("range_1_inf", "within 1 or more cycles", f"{lhs} ##[1:$] {rhs}{suffix}"),
    ]
    if FIXED_DELAY_RE.search(rhs):
        def repl_1_3(md: re.Match[str]) -> str:
            n = int(md.group(1))
            return f"##[{n}:{n + 2}]"
        def repl_1_inf(md: re.Match[str]) -> str:
            n = int(md.group(1))
            return f"##[{n}:$]"
        templates.extend([
            ("lift_fixed_to_range", "within a bounded cycle window", f"{lhs} {FIXED_DELAY_RE.sub(repl_1_3, rhs, count=1)}{suffix}"),
            ("lift_fixed_to_open_range", "within an open-ended cycle window", f"{lhs} {FIXED_DELAY_RE.sub(repl_1_inf, rhs, count=1)}{suffix}"),
        ])
    for tag, phrase, out in templates:
        level, _ = classify_tcl(out)
        if level == 3:
            variants.append((tag, phrase, out))
    return variants


def expand_l4_to_l5_variants(sva: str) -> list[tuple[str, str, str]]:
    sva = normalize_ws(sva)
    parts = split_implication(sva)
    if not parts:
        return []
    if re.search(r"\bs_eventually\b|\bs_until\b|\bs_always\b|\buntil_with\b", sva, re.I):
        return []
    lhs, rhs = parts
    suffix = ";" if sva.endswith(";") else ""
    templates = [
        ("eventually", "eventually, the consequent should hold", f"{lhs} |-> s_eventually ({rhs}){suffix}"),
        ("strong_eventually", "the consequent should eventually become true in a strong sense", f"{lhs} |-> strong(##[1:$] ({rhs})){suffix}"),
        ("until_with", "the trigger condition should persist until the consequent holds", f"{lhs} |-> ({lhs}) until_with ({rhs}){suffix}"),
        ("s_until", "the trigger condition should continue until the consequent becomes true", f"{lhs} |-> ({lhs}) s_until ({rhs}){suffix}"),
        ("s_always", "once triggered, the consequent should always remain true", f"{lhs} |-> s_always ({rhs}){suffix}"),
    ]
    variants: list[tuple[str, str, str]] = []
    for tag, phrase, out in templates:
        level, _ = classify_tcl(out)
        if level == 5:
            variants.append((tag, phrase, out))
    return variants


def method3_synthetic(seed_rows: list[dict]) -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()

    for rec in seed_rows:
        sva = normalize_ws(rec["reference_sva"])
        nl = normalize_ws(rec["nl"])
        level, _ = classify_tcl(sva)
        if level != 4 or not usable_nl(nl):
            continue

        for tag, phrase, sva_l3 in expand_l4_to_l3_variants(sva):
            bh = body_hash(sva_l3)
            if bh in seen:
                continue
            seen.add(bh)
            rows.append({
                "id": make_variant_id("synthetic_l3", bh, tag),
                "source": "synthetic_from_l4",
                "origin_dataset": rec.get("origin_dataset", rec.get("source", "")),
                "nl": synthesize_l3_nl(nl, tag, phrase),
                "reference_sva": sva_l3,
                "rtl_context": rec.get("rtl_context", ""),
                "expected_tcl": 3,
                "hash": bh,
                "parent_id": rec.get("id", ""),
                "parent_hash": rec.get("hash", body_hash(sva)),
                "transform": f"l4_to_l3_{tag}",
            })

        for tag, phrase, sva_l5 in expand_l4_to_l5_variants(sva):
            bh = body_hash(sva_l5)
            if bh in seen:
                continue
            seen.add(bh)
            rows.append({
                "id": make_variant_id("synthetic_l5", bh, tag),
                "source": "synthetic_from_l4",
                "origin_dataset": rec.get("origin_dataset", rec.get("source", "")),
                "nl": synthesize_l5_nl(nl, tag, phrase),
                "reference_sva": sva_l5,
                "rtl_context": rec.get("rtl_context", ""),
                "expected_tcl": 5,
                "hash": bh,
                "parent_id": rec.get("id", ""),
                "parent_hash": rec.get("hash", body_hash(sva)),
                "transform": f"l4_to_l5_{tag}",
            })
    return sorted(rows, key=lambda r: r["id"])


def merge_unique(*groups: list[dict]) -> list[dict]:
    merged: list[dict] = []
    seen: set[str] = set()
    for group in groups:
        for rec in group:
            bh = rec["hash"]
            if bh in seen:
                continue
            seen.add(bh)
            merged.append(rec)
    return merged


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    method1 = method1_existing()
    method2 = method2_github()
    seeds_for_method3 = []
    seeds_for_method3.extend(method1)
    if ALL_MERGED.exists():
        with open(ALL_MERGED) as f:
            for line in f:
                rec = json.loads(line)
                nl = (rec.get("nl") or "").strip()
                sva = normalize_ws(rec.get("reference_sva") or "")
                if not usable_nl(nl) or not sva:
                    continue
                level, _ = classify_tcl(sva)
                if level != 4:
                    continue
                seeds_for_method3.append({
                    "id": rec.get("id") or f"seed_{body_hash(sva)}",
                    "source": "all_merged_train",
                    "origin_dataset": "all_merged_train",
                    "nl": nl,
                    "reference_sva": sva,
                    "rtl_context": rec.get("rtl_context", ""),
                    "expected_tcl": 4,
                    "hash": body_hash(sva),
                })
    method3 = method3_synthetic(seeds_for_method3)

    merged = merge_unique(method1, method2, method3)

    write_jsonl(out_dir / "method1_existing_l3_l5.jsonl", method1)
    write_jsonl(out_dir / "method2_github_high_tcl.jsonl", method2)
    write_jsonl(out_dir / "method3_synthetic_l3_l5.jsonl", method3)
    write_jsonl(out_dir / "all_methods_merged.jsonl", merged)

    manifest = {
        "policy": {
            "method1": "Filter existing structured datasets (all_merged_train + CodeV 83K) for TCL L3/L5.",
            "method2": "Mine locally cloned GitHub repos for high-TCL assertions using explicit temporal-operator hints, then keep TCL L3/L5.",
            "method3": "Programmatically expand TCL-L4 assertions into multi-template synthetic TCL-L3 and TCL-L5 variants; mark all such samples as synthetic.",
        },
        "inputs": {
            "all_merged_train": str(ALL_MERGED),
            "codev_83k": str(CODEV_83K),
            "github_repos": str(GITHUB_REPOS),
        },
        "outputs": {
            "method1": str(out_dir / "method1_existing_l3_l5.jsonl"),
            "method2": str(out_dir / "method2_github_high_tcl.jsonl"),
            "method3": str(out_dir / "method3_synthetic_l3_l5.jsonl"),
            "merged": str(out_dir / "all_methods_merged.jsonl"),
        },
        "counts": {
            "method1_total": len(method1),
            "method2_total": len(method2),
            "method3_total": len(method3),
            "merged_total": len(merged),
            "method1_tcl": summarize(method1),
            "method2_tcl": summarize(method2),
            "method3_tcl": summarize(method3),
            "merged_tcl": summarize(merged),
        },
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(manifest["counts"], indent=2))


if __name__ == "__main__":
    main()
