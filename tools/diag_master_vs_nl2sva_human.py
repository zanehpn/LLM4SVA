"""diag_master_vs_nl2sva_human.py
Side-by-side distribution diagnostic for SFT regression.
Quantifies surface-form gaps between master_train.jsonl and nl2sva_human.jsonl
on axes that PEC equivalence is sensitive to.
"""
from __future__ import annotations
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MASTER = ROOT / "data" / "master" / "master_train.jsonl"
HUMAN = ROOT / "data" / "test" / "nl2sva_human.jsonl"
MACHINE = ROOT / "data" / "test" / "nl2sva_machine.jsonl"


def read_jsonl(p: Path):
    rows = []
    with open(p) as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
    return rows


RESET_RE = re.compile(r"disable\s+iff\s*\(\s*([^)]+?)\s*\)", re.I)
CLK_RE = re.compile(r"@\s*\(\s*([^)]+?)\s*\)", re.I)


def extract_features(sva: str):
    sva = sva or ""
    has_disable_iff = "disable iff" in sva.lower()
    reset_match = RESET_RE.search(sva)
    reset_expr = reset_match.group(1).strip() if reset_match else ""
    clk_match = CLK_RE.search(sva)
    clk_edge = clk_match.group(1).strip() if clk_match else ""
    has_rarrow = "|->" in sva
    has_drarrow = "|=>" in sva
    has_case_eq = "===" in sva or "!==" in sva
    has_bit_lit = bool(re.search(r"1'b[01]", sva))
    has_past = "$past" in sva
    has_rose = "$rose" in sva
    has_fell = "$fell" in sva
    has_changed = "$changed" in sva
    has_stable = "$stable" in sva
    has_seq_delay = bool(re.search(r"##\d", sva))
    has_eventually = "s_eventually" in sva or "eventually" in sva
    has_throughout = "throughout" in sva
    has_within = "within" in sva
    has_intersect = "intersect" in sva
    has_named_prop = bool(re.search(r"\bproperty\s+\w+\s*;", sva))
    has_assert_property = bool(re.search(r"assert\s+property", sva))
    return dict(
        has_disable_iff=has_disable_iff,
        reset_expr=reset_expr,
        clk_edge=clk_edge,
        has_rarrow=has_rarrow,
        has_drarrow=has_drarrow,
        has_case_eq=has_case_eq,
        has_bit_lit=has_bit_lit,
        has_past=has_past,
        has_rose=has_rose,
        has_fell=has_fell,
        has_changed=has_changed,
        has_stable=has_stable,
        has_seq_delay=has_seq_delay,
        has_eventually=has_eventually,
        has_throughout=has_throughout,
        has_within=has_within,
        has_intersect=has_intersect,
        has_named_prop=has_named_prop,
        has_assert_property=has_assert_property,
        sva_len=len(sva),
    )


NL_TEMPLATES = [
    re.compile(r"^create a SVA", re.I),
    re.compile(r"^write a SVA", re.I),
    re.compile(r"use the signals", re.I),
    re.compile(r"checks that", re.I),
    re.compile(r"checks:\s", re.I),
]


def nl_features(nl: str):
    nl = nl or ""
    starts_template = any(p.search(nl) for p in NL_TEMPLATES[:2])
    mentions_signals = bool(NL_TEMPLATES[2].search(nl))
    return dict(starts_template=starts_template,
                mentions_signals=mentions_signals,
                nl_len=len(nl))


def summarize(rows, label):
    n = len(rows)
    feat_counts = Counter()
    reset_exprs = Counter()
    clk_edges = Counter()
    sva_lens = []
    nl_lens = []
    starts_template_n = 0
    mentions_signals_n = 0
    tcl_dist = Counter()
    for r in rows:
        sva = r.get("reference_sva") or r.get("sva") or ""
        nl = r.get("nl") or ""
        f = extract_features(sva)
        nf = nl_features(nl)
        for k, v in f.items():
            if k.startswith("has_") and v:
                feat_counts[k] += 1
        reset_exprs[f["reset_expr"]] += 1
        clk_edges[f["clk_edge"]] += 1
        sva_lens.append(f["sva_len"])
        nl_lens.append(nf["nl_len"])
        if nf["starts_template"]:
            starts_template_n += 1
        if nf["mentions_signals"]:
            mentions_signals_n += 1
        tcl = r.get("expected_tcl")
        if tcl is not None:
            tcl_dist[int(tcl)] += 1

    print(f"\n=== {label}  (n = {n}) ===")
    print(f"  SVA len   median={sorted(sva_lens)[n//2]:>5}  "
          f"mean={sum(sva_lens)//n:>5}  "
          f"min={min(sva_lens):>3}  max={max(sva_lens):>5}")
    print(f"  NL  len   median={sorted(nl_lens)[n//2]:>5}  "
          f"mean={sum(nl_lens)//n:>5}  "
          f"min={min(nl_lens):>3}  max={max(nl_lens):>5}")
    print(f"  TCL dist:  ", dict(sorted(tcl_dist.items())))
    print(f"  NL starts with 'create/write a SVA':  "
          f"{starts_template_n:>5}/{n}  ({100*starts_template_n/n:.1f}%)")
    print(f"  NL contains 'use the signals':         "
          f"{mentions_signals_n:>5}/{n}  ({100*mentions_signals_n/n:.1f}%)")
    print(f"  has_disable_iff:        "
          f"{feat_counts['has_disable_iff']:>5}/{n}  "
          f"({100*feat_counts['has_disable_iff']/n:.1f}%)")
    print(f"  has_assert_property:    "
          f"{feat_counts['has_assert_property']:>5}/{n}  "
          f"({100*feat_counts['has_assert_property']/n:.1f}%)")
    print(f"  has_named_prop:         "
          f"{feat_counts['has_named_prop']:>5}/{n}  "
          f"({100*feat_counts['has_named_prop']/n:.1f}%)")
    print(f"  has |-> :               "
          f"{feat_counts['has_rarrow']:>5}/{n}  "
          f"({100*feat_counts['has_rarrow']/n:.1f}%)")
    print(f"  has |=> :               "
          f"{feat_counts['has_drarrow']:>5}/{n}  "
          f"({100*feat_counts['has_drarrow']/n:.1f}%)")
    print(f"  has === / !== (case eq):"
          f"{feat_counts['has_case_eq']:>5}/{n}  "
          f"({100*feat_counts['has_case_eq']/n:.1f}%)")
    print(f"  has 1'b0 / 1'b1 lits:   "
          f"{feat_counts['has_bit_lit']:>5}/{n}  "
          f"({100*feat_counts['has_bit_lit']/n:.1f}%)")
    print(f"  has $past:              "
          f"{feat_counts['has_past']:>5}/{n}  "
          f"({100*feat_counts['has_past']/n:.1f}%)")
    print(f"  has $rose:              "
          f"{feat_counts['has_rose']:>5}/{n}  "
          f"({100*feat_counts['has_rose']/n:.1f}%)")
    print(f"  has $changed:           "
          f"{feat_counts['has_changed']:>5}/{n}  "
          f"({100*feat_counts['has_changed']/n:.1f}%)")
    print(f"  has ##N seq delay:      "
          f"{feat_counts['has_seq_delay']:>5}/{n}  "
          f"({100*feat_counts['has_seq_delay']/n:.1f}%)")
    print(f"  has s_eventually:       "
          f"{feat_counts['has_eventually']:>5}/{n}  "
          f"({100*feat_counts['has_eventually']/n:.1f}%)")
    print()
    print(f"  Top 12 reset expressions:")
    for expr, c in reset_exprs.most_common(12):
        if expr:
            print(f"    {c:>5}  {expr!r}")
        else:
            print(f"    {c:>5}  (no disable iff)")
    print(f"  Top 6 clk edges:")
    for edge, c in clk_edges.most_common(6):
        if edge:
            print(f"    {c:>5}  {edge!r}")
        else:
            print(f"    {c:>5}  (no @(...) clock)")


def side_by_side(master_rows, human_rows, n_each=4):
    print("\n\n==========================================================")
    print("Side-by-side examples (random per TCL)")
    print("==========================================================")
    import random
    random.seed(0)
    by_tcl_master = defaultdict(list)
    by_tcl_human = defaultdict(list)
    for r in master_rows:
        by_tcl_master[r.get("expected_tcl")].append(r)
    for r in human_rows:
        by_tcl_human[r.get("expected_tcl")].append(r)
    common_tcls = sorted(set(by_tcl_master) & set(by_tcl_human))
    for tcl in common_tcls:
        print(f"\n----------- TCL {tcl} -----------")
        m_samples = random.sample(by_tcl_master[tcl],
                                  min(n_each, len(by_tcl_master[tcl])))
        h_samples = random.sample(by_tcl_human[tcl],
                                  min(n_each, len(by_tcl_human[tcl])))
        for i in range(min(len(m_samples), len(h_samples))):
            mr = m_samples[i]; hr = h_samples[i]
            print(f"\n  [master_train {i}]")
            print(f"    NL : {(mr.get('nl') or '')[:160]}")
            print(f"    SVA: {(mr.get('reference_sva') or '')[:240]}")
            print(f"  [nl2sva_human {i}]")
            print(f"    NL : {(hr.get('nl') or '')[:160]}")
            print(f"    SVA: {(hr.get('reference_sva') or '')[:240]}")


def main():
    for p in (MASTER, HUMAN, MACHINE):
        if not p.exists():
            sys.exit(f"missing {p}")
    master = read_jsonl(MASTER)
    human = read_jsonl(HUMAN)
    machine = read_jsonl(MACHINE)
    summarize(master, "master_train.jsonl")
    norm = ROOT / "data" / "master" / "master_train_norm.jsonl"
    if norm.exists():
        summarize(read_jsonl(norm), "master_train_norm.jsonl (POST-NORMALIZE)")
    summarize(human, "nl2sva_human.jsonl")
    summarize(machine, "nl2sva_machine.jsonl")
    side_by_side(master, human, n_each=2)
    print("\n\n")
    side_by_side(master, machine, n_each=2)


if __name__ == "__main__":
    main()
