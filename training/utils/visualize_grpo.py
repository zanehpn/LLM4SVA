#!/usr/bin/env python3
"""
visualize_grpo.py — Three-panel figure for the SVA4DAC GRPO experiment series,
in the style of Damani et al. 2025 (Figure 3 / Figure 5).

Panels (left to right):
  1. Per-task difficulty distribution: how often each NL2SVA task is solved
     across all evaluated models. A U-shape means most tasks are either always
     or never solved (binary difficulty).
  2. Best-GRPO vs baseline scatter: for each test task, plot the best GRPO
     variant's tcl_match (1 if any GRPO model solves it, else 0) against the
     SFT-without-GRPO baseline. The diagonal marks "no GRPO improvement".
  3. Training-step curves: tcl_match_pct vs GRPO checkpoint step, one line
     per training variant. Horizontal references for the SFT baseline and the
     zero-shot Qwen2.5-Coder-7B-Instruct.

Reads every `eval_nl2sva_human_*.json` in results/ and writes
results/figures/grpo_visualization.{pdf,png}.
"""
import glob
import json
import os
import re
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
FIG_DIR = os.path.join(RESULTS, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Parse run identity from the eval JSON
# ---------------------------------------------------------------------------
def parse_run(model_str: str):
    """Return (kind, family, step) where step may be int or None.

    Handles inconsistent naming patterns that appear across the SVA4DAC eval
    files, including:
        SFT_replay50+grpo_phase4_hilr_100       -> grpo_phase4_hilr / 100
        SFT_replay50+grpo_v4_step50             -> grpo_v4 / 50
        SFT_replay50+grpo_pec_v3_step200        -> grpo_pec_v3 / 200
        SFT_replay50+ipo_v2_final               -> ipo_v2 / None
        SFT_replay50+ipo_v2_250                 -> ipo_v2 / 250
        SFT_replay50+ipo_450                    -> ipo / 450
        Qwen2.5-Coder-7B_SFT_replay20+grpo_pilot_1 -> grpo_pilot / 1
        Qwen2.5-Coder-7B_SFT_replay50           -> sft_replay50 / None
        Qwen2.5-Coder-7B-Instruct_zero-shot     -> zero_shot / None
        CodeV-SVA-14B[_rerun32k]                -> codev_14b / None
    """
    s = model_str

    # Static baselines
    if re.match(r"^Qwen2\.5-Coder-7B-Instruct_zero-shot$", s):
        return "zero_shot", "zero_shot", None
    m = re.match(r"^Qwen2\.5-Coder-7B_SFT_replay(\d+)$", s)
    if m:
        return "sft", f"sft_replay{m.group(1)}", None
    if re.match(r"^CodeV-SVA-14B(?:_.*)?$", s):
        return "codev_14b", "codev_14b", None

    # GRPO / IPO checkpoints — strip optional Qwen prefix
    s2 = re.sub(r"^Qwen2\.5-Coder-7B_SFT_replay\d+\+", "", s)
    s2 = re.sub(r"^SFT_replay\d+\+", "", s2)

    # GRPO  (handles both "v4_step50" — no separator — and "phase4_hilr_100")
    m = re.match(r"^grpo_(.+?)_(?:step)?(\d+|final)$", s2)
    if m:
        family = m.group(1)
        step_raw = m.group(2)
        step = None if step_raw == "final" else int(step_raw)
        return "grpo", f"grpo_{family}", step

    # IPO
    m = re.match(r"^ipo_(.+?)_(?:step)?(\d+|final)$", s2)
    if m:
        family = m.group(1)
        step_raw = m.group(2)
        step = None if step_raw == "final" else int(step_raw)
        return "ipo", f"ipo_{family}", step
    m = re.match(r"^ipo_(\d+|final)$", s2)
    if m:
        step_raw = m.group(1)
        step = None if step_raw == "final" else int(step_raw)
        return "ipo", "ipo", step

    return "other", "other", None


# ---------------------------------------------------------------------------
# Load every eval JSON
# ---------------------------------------------------------------------------
def load_evals():
    rows = []
    for path in glob.glob(os.path.join(RESULTS, "eval_nl2sva_human_*.json")):
        try:
            with open(path) as f:
                d = json.load(f)
        except Exception as e:
            print(f"  skip {os.path.basename(path)}: {e}")
            continue
        kind, family, step = parse_run(d.get("model", ""))
        rows.append(
            dict(
                path=path,
                model=d.get("model", ""),
                kind=kind,
                family=family,
                step=step,
                num_tasks=d.get("num_tasks", 0),
                syntax_ok_pct=d["overall"]["syntax_ok_pct"],
                tcl_match_pct=d["overall"]["tcl_match_pct"],
                body_match_pct=d["overall"]["body_match_pct"],
                per_sample=d.get("per_sample", []),
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Helpers for the per-task aggregation (Panel 1 + Panel 2)
# ---------------------------------------------------------------------------
def per_task_solved_counts(rows, kinds=("grpo", "ipo")):
    """Returns dict task_id -> (solved_count, total_count) over selected kinds."""
    solved = defaultdict(int)
    total = defaultdict(int)
    for r in rows:
        if r["kind"] not in kinds:
            continue
        for s in r["per_sample"]:
            tid = s["id"]
            total[tid] += 1
            if s.get("tcl_match"):
                solved[tid] += 1
    return {tid: (solved[tid], total[tid]) for tid in total}


def best_per_task(rows, kind="grpo"):
    """Returns dict task_id -> 1 if any run of this kind solved it else 0."""
    best = {}
    for r in rows:
        if r["kind"] != kind:
            continue
        for s in r["per_sample"]:
            tid = s["id"]
            if s.get("tcl_match"):
                best[tid] = 1
            elif tid not in best:
                best[tid] = 0
    return best


def baseline_per_task(rows, family="sft_replay50"):
    """Returns task_id -> 1/0 from a single SFT baseline eval (most recent)."""
    cands = [r for r in rows if r["family"] == family]
    if not cands:
        return {}
    # pick the most recent one
    cands.sort(key=lambda r: r["path"], reverse=True)
    r = cands[0]
    return {s["id"]: 1 if s.get("tcl_match") else 0 for s in r["per_sample"]}


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def make_figure(rows):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    # ---------- Panel 1: per-task difficulty distribution ----------
    counts = per_task_solved_counts(rows, kinds=("grpo", "ipo"))
    if not counts:
        axes[0].text(0.5, 0.5, "No GRPO/IPO eval data", ha="center", va="center")
    else:
        success_rates = np.array([s / t for s, t in counts.values() if t > 0])
        axes[0].hist(success_rates, bins=11, range=(0, 1),
                     color="tab:blue", edgecolor="white", alpha=0.85)
        axes[0].set_xlabel("Per-task solved fraction across GRPO/IPO runs")
        axes[0].set_ylabel("Number of tasks")
        axes[0].set_title("Task difficulty distribution")
        axes[0].grid(True, alpha=0.3)

    # ---------- Panel 2: best-GRPO vs SFT baseline per task ----------
    grpo_best = best_per_task(rows, kind="grpo")
    sft_base = baseline_per_task(rows, family="sft_replay50")
    if not grpo_best or not sft_base:
        axes[1].text(0.5, 0.5, "Need SFT baseline + GRPO evals", ha="center", va="center")
    else:
        common = sorted(set(grpo_best) & set(sft_base))
        # 2x2 contingency
        n_both = sum(1 for t in common if sft_base[t] and grpo_best[t])
        n_only_sft = sum(1 for t in common if sft_base[t] and not grpo_best[t])
        n_only_grpo = sum(1 for t in common if not sft_base[t] and grpo_best[t])
        n_neither = sum(1 for t in common if not sft_base[t] and not grpo_best[t])
        cells = np.array([[n_both, n_only_sft], [n_only_grpo, n_neither]])
        im = axes[1].imshow(cells, cmap="Blues", aspect="auto")
        axes[1].set_xticks([0, 1])
        axes[1].set_xticklabels(["SFT solves", "SFT fails"])
        axes[1].set_yticks([0, 1])
        axes[1].set_yticklabels(["GRPO any solves", "GRPO any fails"])
        axes[1].set_title(f"Best GRPO vs SFT baseline (n={len(common)})")
        for (i, j), v in np.ndenumerate(cells):
            axes[1].text(j, i, str(int(v)), ha="center", va="center",
                         color="white" if v > cells.max() / 2 else "black",
                         fontsize=14, fontweight="bold")
        plt.colorbar(im, ax=axes[1], shrink=0.8)

    # ---------- Panel 3: GRPO step curves ----------
    by_family = defaultdict(list)
    for r in rows:
        if r["kind"] in ("grpo", "ipo") and r["step"] is not None:
            by_family[r["family"]].append((r["step"], r["tcl_match_pct"]))
    color_cycle = plt.cm.tab20(np.linspace(0, 1, max(1, len(by_family))))
    for (fam, pts), c in zip(sorted(by_family.items()), color_cycle):
        pts.sort()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        axes[2].plot(xs, ys, marker="o", linewidth=1.5, label=fam, color=c)

    # baseline reference lines
    base_lines = [
        ("zero_shot", "Qwen2.5-Coder-7B (zero-shot)", "gray", "--"),
        ("sft_replay50", "SFT replay50", "black", ":"),
        ("codev_14b", "CodeV-SVA-14B", "tab:red", "-."),
    ]
    for fam, label, color, ls in base_lines:
        cands = [r for r in rows if r["family"] == fam]
        if not cands:
            continue
        # use most recent
        cands.sort(key=lambda r: r["path"], reverse=True)
        y = cands[0]["tcl_match_pct"]
        axes[2].axhline(y, color=color, linestyle=ls, alpha=0.7, label=f"{label} ({y:.1f}%)")

    axes[2].set_xlabel("GRPO / IPO checkpoint step")
    axes[2].set_ylabel("TCL match (%)")
    axes[2].set_title("Training-step curves vs baselines")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=7, loc="best", ncol=2)

    fig.suptitle(
        "SVA4DAC GRPO results on NL2SVA-Human (n=79 tasks)",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()

    for ext in ("pdf", "png"):
        out = os.path.join(FIG_DIR, f"grpo_visualization.{ext}")
        fig.savefig(out, bbox_inches="tight", dpi=150 if ext == "png" else None)
        print(f"wrote {out}")


# ---------------------------------------------------------------------------
def summary_print(rows):
    print(f"\nLoaded {len(rows)} eval files\n")
    by_kind = defaultdict(int)
    for r in rows:
        by_kind[r["kind"]] += 1
    for k, c in sorted(by_kind.items(), key=lambda x: -x[1]):
        print(f"  {k}: {c} eval(s)")
    print()
    print(f"{'family':28} {'step':>6} {'syntax%':>8} {'tcl%':>6} {'body%':>6}")
    print("-" * 60)
    rows_sorted = sorted(
        rows, key=lambda r: (r["family"], r["step"] if r["step"] is not None else 0)
    )
    for r in rows_sorted:
        st = "" if r["step"] is None else r["step"]
        print(f"{r['family']:28} {str(st):>6} "
              f"{r['syntax_ok_pct']:>7.1f}  {r['tcl_match_pct']:>5.1f}  "
              f"{r['body_match_pct']:>5.1f}")


def main():
    rows = load_evals()
    if not rows:
        print("No eval JSON found.")
        sys.exit(1)
    summary_print(rows)
    make_figure(rows)


if __name__ == "__main__":
    main()
