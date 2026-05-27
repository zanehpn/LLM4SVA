"""Rescore the Syntax column of paper Table 4 (tab:pec) using VCS instead
of the heuristic mock_verifier.syntax_check().

Reads one or more existing eval JSONs, locates every `generated_sva`, runs
`vcs_compile_check(sva, "", "")` on each, and writes a sibling JSON with
the suffix `.rescored_vcs.json`. Original metrics are preserved; only
syntax_ok flags and the rolled-up syntax pass-rate are updated.

Two input shapes are supported:
  - old style (run_eval_nl2sva_human.py):
        per_sample[i].generated_sva
        overall.syntax_ok_pct
  - funcatk style (run_funcatk_eval.py):
        per_sample[i].candidates[j].generated_sva
        overall.candidate_syntax_ok_pct
"""
from __future__ import annotations
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "compile_gate"))
from vcs_compile_check import vcs_compile_check  # noqa: E402


def _vcs_one(sva: str) -> bool:
    return vcs_compile_check(sva, "", "")


def _collect_svas(d: dict):
    """Yield (path-tuple, sva) pairs for every generated_sva in the JSON.

    path-tuple identifies where to write the result back. Two shapes:
      ('flat',   i)        -> per_sample[i]
      ('nested', i, j)     -> per_sample[i].candidates[j]
    """
    ps = d.get("per_sample") or []
    if not ps:
        return
    if "candidates" in (ps[0] or {}):
        for i, s in enumerate(ps):
            for j, c in enumerate(s.get("candidates", [])):
                yield ("nested", i, j), c.get("generated_sva", "")
    else:
        for i, s in enumerate(ps):
            yield ("flat", i), s.get("generated_sva", "")


def _apply_results(d: dict, results: dict):
    """Write VCS verdicts back into the JSON dict in place."""
    for path, ok in results.items():
        if path[0] == "flat":
            d["per_sample"][path[1]]["syntax_ok"] = ok
        else:
            d["per_sample"][path[1]]["candidates"][path[2]]["syntax_ok"] = ok

    # Roll up.
    total = len(results)
    ok_count = sum(1 for v in results.values() if v)
    pct = round(100.0 * ok_count / max(total, 1), 2)
    overall = d.setdefault("overall", {})
    if any(p[0] == "nested" for p in results):
        overall["candidate_syntax_ok_pct"] = pct
    else:
        overall["syntax_ok_pct"] = pct
    d["syntax_engine"] = "vcs L-2016.06 (sverilog -assert svaext, free-input wrapper)"
    d["syntax_total"] = total
    d["syntax_ok_count"] = ok_count
    return ok_count, total, pct


def rescore(json_path: Path, workers: int = 8) -> dict:
    d = json.load(open(json_path))
    items = list(_collect_svas(d))
    print(f"[{json_path.name}] {len(items)} SVAs to compile", flush=True)
    t0 = time.time()
    results = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        fut_to_path = {pool.submit(_vcs_one, sva): path for path, sva in items}
        done = 0
        for fut in as_completed(fut_to_path):
            path = fut_to_path[fut]
            try:
                ok = bool(fut.result())
            except Exception:
                ok = False
            results[path] = ok
            done += 1
            if done % 25 == 0 or done == len(items):
                ok_so_far = sum(1 for v in results.values() if v)
                print(
                    f"  {done}/{len(items)}  ok={ok_so_far}  "
                    f"({100*ok_so_far/done:.1f}%)  "
                    f"elapsed={time.time()-t0:.0f}s",
                    flush=True,
                )
    ok_count, total, pct = _apply_results(d, results)
    out = json_path.with_suffix(".rescored_vcs.json")
    out.write_text(json.dumps(d, indent=2))
    print(
        f"[{json_path.name}] -> {out.name}  Syntax(VCS)={ok_count}/{total} "
        f"({pct}%)  in {time.time()-t0:.0f}s",
        flush=True,
    )
    return {"file": str(out), "ok": ok_count, "total": total, "pct": pct}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="Eval JSONs to rescore")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    summary = []
    for p in args.paths:
        summary.append(rescore(Path(p), args.workers))
    print("\n=== VCS rescore summary ===")
    for s in summary:
        print(f"  {Path(s['file']).name:<80}  {s['ok']}/{s['total']}  ({s['pct']}%)")
