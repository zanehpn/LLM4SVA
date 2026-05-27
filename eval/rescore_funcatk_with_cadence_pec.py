#!/usr/bin/env python3
"""rescore_funcatk_with_cadence_pec.py — re-score an existing funcatk
result JSON with **Cadence JasperGold's `prop_eq_checker`** (paper §5
headline pass@k oracle, App. E).

JasperGold is closed-source and not installable on the host; this
rescorer calls a Docker image that ships JG plus a thin TCL helper.
There is no silent fallback to the open SymbiYosys+Z3 PEC — if the
Docker image or JG license is missing the script exits with code 2.
For the open PEC diagnostic that runs in-loop during training/eval,
use `eval/run_eval_with_pec.py` instead.

Output: writes `<input>.rescored_cadence.json` next to the source, with
  - per_sample[*].candidates[*].pec_verdict_strict  (copy of original)
  - per_sample[*].candidates[*].pec_verdict_cadence (JG verdict)
  - per_sample[*].candidates[*].pec_fwd_cadence / pec_bwd_cadence
  - overall_strict  / overall_cadence               (pass@k for both)
  - per_tcl_strict  / per_tcl_cadence
  - verdict_counts_strict / verdict_counts_cadence

The Docker contract (paper §5 / App. E):
  * Image       — supplied via --jg-docker-image (default: `cadence-jg:latest`).
  * Entrypoint  — supplied via --jg-helper-script (default: `/work/jg_prop_eq.sh`).
                  The script reads one JSON object per line from stdin
                  (fields: id, lm_sva, ref_sva, rtl, depth, timeout) and
                  writes one JSON line per result to stdout (fields:
                  id, verdict, fwd_status, bwd_status, seconds).
                  Verdict ∈ {EQUIVALENT, IMPLIES_REF_TO_LM,
                  IMPLIES_LM_TO_REF, NOT_EQUIVALENT, UNSUPPORTED}.
  * License     — JG license is mounted into the container by the
                  helper script (e.g. via -v $CDS_LIC_FILE).

Usage:
    python3 eval/rescore_funcatk_with_cadence_pec.py \\
        --input  results/funcatk_eval_<...>.json \\
        --tasks  data/test/nl2sva_machine.jsonl \\
        --jg-docker-image cadence-jg:latest \\
        --jg-helper-script /work/jg_prop_eq.sh \\
        --workers 4 --depth 15 --timeout 60
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# Paper §4.3 / App. E — JasperGold returns one of these five verdicts.
_PUBLIC_VERDICTS = {
    "EQUIVALENT", "IMPLIES_REF_TO_LM", "IMPLIES_LM_TO_REF",
    "NOT_EQUIVALENT", "UNSUPPORTED",
}


class JasperUnavailable(RuntimeError):
    """Raised when the Cadence JG Docker stack is not callable."""


def pass_at_k(n: int, c: int, k: int) -> float:
    if c <= 0:
        return 0.0
    if k >= n:
        return 1.0 if c > 0 else 0.0
    if n - c < k:
        return 1.0
    return 1.0 - (math.comb(n - c, k) / math.comb(n, k))


def _check_docker(image: str) -> None:
    """Raise JasperUnavailable if `docker` or the JG image is missing.
    Silent fallback to open PEC is explicitly *not* allowed — the rescorer
    must use the same oracle the paper used (Cadence JG) or exit 2."""
    docker = shutil.which("docker")
    if docker is None:
        raise JasperUnavailable(
            "`docker` not found on PATH. JasperGold rescoring requires "
            "a Cadence JG docker image (see header docstring).")
    # We don't `docker image inspect` here because some setups pull on demand;
    # we just probe with a no-op exec and rely on the helper script's exit.


def _jg_docker_call(image: str, script: str, payload: list,
                    extra_args: list) -> list:
    """Invoke the JG helper script over docker exec with one JSON line per
    candidate on stdin. Returns the parsed list of result lines in the
    same order as `payload`. Raises JasperUnavailable on docker/JG error."""
    if not payload:
        return []
    cmd = ["docker", "run", "--rm", "-i", *extra_args, image, script]
    stdin_blob = "\n".join(json.dumps(p) for p in payload) + "\n"
    try:
        proc = subprocess.run(
            cmd, input=stdin_blob, text=True,
            capture_output=True, check=False,
        )
    except FileNotFoundError as e:
        raise JasperUnavailable(f"failed to spawn docker: {e}") from e
    if proc.returncode != 0:
        raise JasperUnavailable(
            f"JG docker exited {proc.returncode}; stderr tail:\n"
            f"{proc.stderr.strip()[-400:]}"
        )
    out = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="source funcatk result JSON")
    ap.add_argument("--tasks", default="",
                    help="override tasks_path; defaults to value in source")
    ap.add_argument("--workers", type=int, default=4,
                    help="parallel candidates per docker invocation. JG "
                         "license seats usually dominate; keep <= license count.")
    ap.add_argument("--depth", type=int, default=15)
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--jg-docker-image", default=os.environ.get(
                        "JG_DOCKER_IMAGE", "cadence-jg:latest"),
                    help="Docker image with a licensed Cadence JG install.")
    ap.add_argument("--jg-helper-script", default=os.environ.get(
                        "JG_HELPER_SCRIPT", "/work/jg_prop_eq.sh"),
                    help="path inside the image to a script that reads "
                         "JSON-per-line on stdin and prints results on stdout.")
    ap.add_argument("--jg-docker-args", default="",
                    help="extra args to pass to `docker run` (e.g. license "
                         "mounts: `-v /opt/cds.lic:/opt/cds.lic`).")
    ap.add_argument("--output", default="",
                    help="output path; defaults to <input>.rescored_cadence.json")
    args = ap.parse_args()

    try:
        _check_docker(args.jg_docker_image)
    except JasperUnavailable as e:
        print(f"[rescore] JasperGold rescoring not available: {e}",
              file=sys.stderr)
        print("[rescore] If you need an open-PEC diagnostic instead, run "
              "`eval/run_eval_with_pec.py`. Exiting 2.", file=sys.stderr)
        sys.exit(2)

    src_path = Path(args.input).resolve()
    src = json.load(open(src_path))
    tasks_path = Path(args.tasks or src.get("tasks_path"))
    if not tasks_path.is_absolute():
        tasks_path = (ROOT / tasks_path).resolve()

    by_id = {}
    with open(tasks_path) as f:
        for line in f:
            r = json.loads(line)
            by_id[r["id"]] = r

    per_sample = src["per_sample"]
    num_samples = src["num_samples"]
    ks = src["ks"]

    # Build the JG work list — one entry per evaluable candidate.
    payload = []
    work_keys = []  # (task_idx, cand_idx) parallel to payload
    for task_idx, row in enumerate(per_sample):
        tid = row["id"]
        task = by_id.get(tid)
        if task is None:
            raise SystemExit(f"task id not found in tasks jsonl: {tid}")
        ref_sva = task.get("reference_sva", "")
        rtl = task.get("rtl_context", "")
        if not row.get("evaluable", True):
            continue
        for cand_idx, cand in enumerate(row["candidates"]):
            gen_sva = cand.get("generated_sva", "")
            payload.append({
                "id":      f"{tid}:{cand_idx}",
                "lm_sva":  gen_sva,
                "ref_sva": ref_sva,
                "rtl":     rtl,
                "depth":   args.depth,
                "timeout": args.timeout,
            })
            work_keys.append((task_idx, cand_idx))

    print(f"[rescore] input: {src_path.name}")
    print(f"[rescore] tasks: {tasks_path}")
    print(f"[rescore] candidates to rescore: {len(payload)}")
    print(f"[rescore] JG image: {args.jg_docker_image}")
    print(f"[rescore] JG script: {args.jg_helper_script}")

    extra_args = args.jg_docker_args.split() if args.jg_docker_args else []
    # Chunk the payload so we don't spawn one container per candidate but
    # also don't keep one container alive for the entire run. Each chunk
    # exits cleanly so the helper script can release JG license seats.
    chunk_size = max(args.workers, 4)
    results = []
    for chunk_start in range(0, len(payload), chunk_size):
        chunk = payload[chunk_start:chunk_start + chunk_size]
        try:
            results.extend(_jg_docker_call(
                args.jg_docker_image, args.jg_helper_script,
                chunk, extra_args))
        except JasperUnavailable as e:
            print(f"[rescore] JasperGold call failed: {e}", file=sys.stderr)
            sys.exit(2)
        print(f"[rescore] {min(chunk_start + chunk_size, len(payload))}"
              f"/{len(payload)}")

    # Map results back by id.
    by_payload_id = {r["id"]: r for r in results}

    # Deep-copy per_sample and tag fields.
    out_per_sample = copy.deepcopy(per_sample)
    for row in out_per_sample:
        for cand in row["candidates"]:
            cand["pec_verdict_strict"] = cand.get("pec_verdict")
            cand["pec_fwd_strict"] = cand.get("pec_fwd", "")
            cand["pec_bwd_strict"] = cand.get("pec_bwd", "")
            cand["pec_verdict_cadence"] = None
            cand["pec_fwd_cadence"] = ""
            cand["pec_bwd_cadence"] = ""
            cand["pec_seconds_cadence"] = 0.0

    verdict_counts_cadence = Counter()
    for (task_idx, cand_idx), payload_row in zip(work_keys, payload):
        res = by_payload_id.get(payload_row["id"])
        cand = out_per_sample[task_idx]["candidates"][cand_idx]
        if res is None:
            cand["pec_verdict_cadence"] = "UNSUPPORTED"
            cand["pec_fwd_cadence"] = "NORESULT"
            cand["pec_bwd_cadence"] = "NORESULT"
            verdict_counts_cadence["UNSUPPORTED"] += 1
            continue
        verdict = res.get("verdict", "UNSUPPORTED")
        if verdict not in _PUBLIC_VERDICTS:
            verdict = "UNSUPPORTED"
        cand["pec_verdict_cadence"] = verdict
        cand["pec_fwd_cadence"] = res.get("fwd_status", "")
        cand["pec_bwd_cadence"] = res.get("bwd_status", "")
        cand["pec_seconds_cadence"] = float(res.get("seconds", 0.0) or 0.0)
        verdict_counts_cadence[verdict] += 1

    # Candidates that were not evaluable keep their open-PEC verdict so the
    # per-task counts remain comparable across rescore runs.
    verdict_counts_strict = Counter()
    for row in out_per_sample:
        for cand in row["candidates"]:
            verdict_counts_strict[cand.get("pec_verdict_strict")] += 1
            if cand["pec_verdict_cadence"] is None:
                cand["pec_verdict_cadence"] = cand["pec_verdict_strict"]

    # Recompute pass@k for both verdict streams (paper §5 unbiased estimator).
    def _aggregate(key: str) -> tuple:
        strict_sum = Counter()
        relaxed_sum = Counter()
        strict_by_tcl = defaultdict(Counter)
        relaxed_by_tcl = defaultdict(Counter)
        total_by_tcl = Counter()
        eval_by_tcl = Counter()
        evaluable_total = 0
        for row in out_per_sample:
            exp_tcl = row["expected_tcl"]
            total_by_tcl[exp_tcl] += 1
            if not row.get("evaluable", True):
                continue
            evaluable_total += 1
            eval_by_tcl[exp_tcl] += 1
            strict_c = sum(1 for c in row["candidates"]
                           if c.get(key) == "EQUIVALENT")
            relaxed_c = sum(1 for c in row["candidates"]
                            if c.get(key) in ("EQUIVALENT", "IMPLIES_REF_TO_LM",
                                              "IMPLIES_LM_TO_REF"))
            for k in ks:
                strict_sum[k] += pass_at_k(num_samples, strict_c, k)
                relaxed_sum[k] += pass_at_k(num_samples, relaxed_c, k)
                strict_by_tcl[exp_tcl][k] += pass_at_k(num_samples, strict_c, k)
                relaxed_by_tcl[exp_tcl][k] += pass_at_k(num_samples, relaxed_c, k)
        overall = {f"func@{k}": round(100 * strict_sum[k] / max(evaluable_total, 1), 2)
                   for k in ks}
        overall_relaxed = {
            f"func_relaxed@{k}": round(100 * relaxed_sum[k] / max(evaluable_total, 1), 2)
            for k in ks}
        per_tcl = {}
        for tcl in sorted(total_by_tcl):
            denom = eval_by_tcl.get(tcl, 0)
            per_tcl[str(tcl)] = {
                "total": total_by_tcl[tcl],
                "evaluable": denom,
                **{f"func@{k}": round(100 * strict_by_tcl[tcl][k] / max(denom, 1), 2)
                   for k in ks},
                **{f"func_relaxed@{k}": round(100 * relaxed_by_tcl[tcl][k] / max(denom, 1), 2)
                   for k in ks},
            }
        return {**overall, **overall_relaxed}, per_tcl, evaluable_total

    overall_strict, per_tcl_strict, evaluable_total = _aggregate("pec_verdict_strict")
    overall_cadence, per_tcl_cadence, _ = _aggregate("pec_verdict_cadence")

    out = {
        **{k: v for k, v in src.items() if k not in ("per_sample", "overall",
                                                     "per_tcl", "verdict_counts")},
        "rescored_from": str(src_path),
        "rescored_oracle": "cadence_jaspergold_docker",
        "jg_docker_image": args.jg_docker_image,
        "jg_helper_script": args.jg_helper_script,
        "evaluable_tasks": evaluable_total,
        "overall_strict": overall_strict,
        "overall_cadence": overall_cadence,
        "per_tcl_strict": per_tcl_strict,
        "per_tcl_cadence": per_tcl_cadence,
        "verdict_counts_strict": dict(verdict_counts_strict),
        "verdict_counts_cadence": dict(verdict_counts_cadence),
        "per_sample": out_per_sample,
    }

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = src_path.with_suffix(".rescored_cadence.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print()
    print("=" * 72)
    print(f"Rescored (Cadence JG via docker) -> {out_path}")
    print("=" * 72)
    print(f"Evaluable tasks: {evaluable_total}")
    for k in ks:
        s = overall_strict[f"func@{k}"]
        c = overall_cadence[f"func@{k}"]
        sr = overall_strict[f"func_relaxed@{k}"]
        cr = overall_cadence[f"func_relaxed@{k}"]
        print(f"  pass@{k:<2}  open-PEC={s:6.2f}  JG={c:6.2f}   "
              f"(Δ={c - s:+.2f})")
        print(f"  relax@{k:<2} open-PEC={sr:6.2f}  JG={cr:6.2f}   "
              f"(Δ={cr - sr:+.2f})")
    print("Verdict shift (open-PEC -> Cadence JG):")
    all_verdicts = (set(verdict_counts_strict) | set(verdict_counts_cadence))
    for v in sorted(all_verdicts, key=lambda x: (x is None, str(x))):
        s = verdict_counts_strict.get(v, 0)
        c = verdict_counts_cadence.get(v, 0)
        label = "NONE (non-evaluable)" if v is None else str(v)
        print(f"  {label:22s} {s:5d} -> {c:5d}   (Δ={c - s:+d})")


if __name__ == "__main__":
    main()
