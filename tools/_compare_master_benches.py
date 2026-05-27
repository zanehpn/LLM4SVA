"""Cross-tab pass rates from raw / v1 / v2 / v3 master VCS benches and
print an ablation table by temporal_class. One-shot helper, called
manually after the benches complete."""
from __future__ import annotations
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# raw → already on disk (master_vcs_bench.jsonl)
# v1  → master_norm_vcs_bench.jsonl       (R1-R16, OLD wrapper)
# v2  → master_norm_v2_vcs_bench.jsonl    (R1-R17, OLD wrapper)
# v3  → master_norm_v2_w2_vcs_bench.jsonl (R1-R17, NEW wrapper) [optional]
LABELS = [
    ("raw",   "data/master/master_vcs_bench.jsonl",
              "data/master/master_train_rtl_filled_sva_capped_regen.jsonl"),
    ("v1",    "data/master/master_norm_vcs_bench.jsonl",
              "data/master/master_train_rtl_filled_sva_capped_regen_norm.jsonl"),
    ("v2",    "data/master/master_norm_v2_vcs_bench.jsonl",
              "data/master/master_train_rtl_filled_sva_capped_regen_norm_v2.jsonl"),
    ("v3",    "data/master/master_norm_v2_w2_vcs_bench.jsonl",
              "data/master/master_train_rtl_filled_sva_capped_regen_norm_v2.jsonl"),
]

def load_classes(path):
    classes = {}
    with open(ROOT / path) as f:
        for i, ln in enumerate(f):
            r = json.loads(ln)
            classes[i] = str(r.get("temporal_class") or "?")
    return classes

def load_bench(path):
    rows = []
    with open(ROOT / path) as f:
        for ln in f:
            rows.append(json.loads(ln))
    return rows

if __name__ == "__main__":
    rows_per_label = {}
    for label, bench, source in LABELS:
        if not (ROOT / bench).exists():
            print(f"[skip] {label}: {bench} not found")
            continue
        cls = load_classes(source)
        bench_rows = load_bench(bench)
        rows_per_label[label] = (bench_rows, cls)

    classes = sorted({"C1","C2","C3","?"})
    print(f"{'metric':<12}", *(f"{lbl:>10}" for lbl in rows_per_label), sep="")
    # Total
    line = ["total".ljust(12)]
    for label, (rows, _) in rows_per_label.items():
        line.append(f"{len(rows):>10}")
    print(*line, sep="")
    # Pass count
    line = ["passed".ljust(12)]
    for label, (rows, _) in rows_per_label.items():
        n_ok = sum(1 for r in rows if r.get("ok"))
        line.append(f"{n_ok:>10}")
    print(*line, sep="")
    # Pass %
    line = ["pass%".ljust(12)]
    for label, (rows, _) in rows_per_label.items():
        n_ok = sum(1 for r in rows if r.get("ok"))
        pct = 100*n_ok/max(len(rows),1)
        line.append(f"{pct:>9.1f}%")
    print(*line, sep="")
    print()
    # By class
    for c in classes:
        line = [f"  {c}".ljust(12)]
        for label, (rows, cls) in rows_per_label.items():
            tot = ok = 0
            for i, r in enumerate(rows):
                if cls.get(i, "?") != c: continue
                tot += 1
                if r.get("ok"): ok += 1
            pct = 100*ok/max(tot,1)
            line.append(f"{ok:>5}/{tot:<4} {pct:>4.1f}%")
        print(*line, sep="  ")
