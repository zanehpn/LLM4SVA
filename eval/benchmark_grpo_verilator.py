"""benchmark_grpo_verilator.py
Run Verilator `--lint-only --assert` on every (rtl_context, reference_sva)
pair in a jsonl pool and report pass/fail rates. Two compile modes:

  --mode free-rtl   (default): synthesize a minimal module declaring every
                                 identifier in the SVA as a free input;
                                 isolates "is the SVA itself well-formed".
                                 Equivalent to the GRPO `verilator_then_pec`
                                 reward gate.

  --mode bind-rtl              : append the actual rtl_context, then bind
                                 a module containing the SVA into it.
                                 Tests "does the SVA integrate with the
                                 actual RTL". Most industrial RTL fails
                                 here due to package / macro / hierarchical
                                 path dependencies Verilator can't resolve
                                 without the full project flist — that's
                                 informative on its own.

Usage:
    python scripts/benchmark_grpo_verilator.py \\
        --input data/master/grpo_unified_combined.jsonl \\
        --output data/master/grpo_verilator_bench.jsonl \\
        --workers 16 --mode free-rtl
"""
from __future__ import annotations
import argparse
import json
import multiprocessing as mp
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Reuse helpers from run_grpo_pilot
import importlib.util
_g_spec = importlib.util.spec_from_file_location(
    "_grpo_helpers", str(ROOT / "training" / "rlvf" / "run_grpo_pilot.py"))
_g = importlib.util.module_from_spec(_g_spec)
_g_spec.loader.exec_module(_g)

verilator_compile_check = _g.verilator_compile_check
_build_compileable_unit = _g._build_compileable_unit
_strip_sva_label = _g._strip_sva_label
_identifiers_in_sva = _g._identifiers_in_sva
_SV_KEYWORDS_FOR_FREE_RTL = _g._SV_KEYWORDS_FOR_FREE_RTL


MODULE_DECL_RE = re.compile(r"^\s*module\s+([A-Za-z_]\w*)", re.MULTILINE)


def build_bind_rtl_unit(sva: str, reference_sva: str, rtl_context: str) -> str:
    """rtl_context (full module) + a tiny sva_check module + a bind. If
    rtl_context isn't a parseable module declaration, fall back to the
    free-input wrapper."""
    rtl = (rtl_context or "").strip()
    m = MODULE_DECL_RE.search(rtl)
    if (not rtl) or (not m) or ("endmodule" not in rtl):
        return _build_compileable_unit(sva, reference_sva, rtl_context)
    target_module = m.group(1)
    exclude = {"clk", "tb_reset"}
    ids = _identifiers_in_sva(sva, exclude) | _identifiers_in_sva(reference_sva, exclude)
    decls = ["    input logic clk", "    input logic tb_reset"]
    for ident in sorted(ids):
        decls.append(f"    input logic [31:0] {ident}")
    body = (
        rtl.rstrip() + "\n\n"
        "module sva_check (\n"
        + ",\n".join(decls)
        + "\n);\n"
        "  " + (sva or "").strip() + "\n"
        "endmodule\n"
    )
    return body


def _verilator_run(code: str, timeout: int = 8) -> tuple[bool, str]:
    """Return (ok, short_error)."""
    try:
        with tempfile.TemporaryDirectory(prefix="vrlr_bench_") as td:
            f = Path(td) / "check.sv"
            f.write_text(code)
            r = subprocess.run(
                ["verilator", "--lint-only", "--assert", "-Wno-fatal",
                 "-Wno-DECLFILENAME", "-Wno-MULTITOP", "-Wno-MODDUP", str(f)],
                capture_output=True, timeout=timeout, text=True,
            )
            if r.returncode == 0:
                return True, ""
            err = (r.stderr or "")
            # Pick the first %Error: line, truncated
            m = re.search(r"%Error:[^\n]*", err)
            return False, (m.group(0)[:200] if m else err[:200].strip())
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except FileNotFoundError:
        return False, "NO_VERILATOR"
    except Exception as e:
        return False, f"EXCEPTION:{type(e).__name__}"


# VCS engine: docker exec into the vcs_lint container running the
# synopsys2016 image. The host path under experiments/runs/vcs_tmp/ is
# bind-mounted at /work/runs/vcs_tmp/ inside the container.
VCS_HOST_TMP = ROOT / "runs" / "vcs_tmp"
VCS_CONTAINER_TMP = "/work/runs/vcs_tmp"
VCS_CONTAINER_SCRIPT = "/work/scripts/vcs_check.sh"
VCS_CONTAINER_NAME = "vcs_lint"


def _vcs_run(code: str, timeout: int = 20) -> tuple[bool, str]:
    """Return (ok, short_error). Writes code to a host file under the
    bind-mounted scratch dir, then `docker exec` runs vcs_check.sh which
    returns 0 iff VCS emitted no `^Error-[` lines."""
    VCS_HOST_TMP.mkdir(parents=True, exist_ok=True)
    fd, host_path = tempfile.mkstemp(suffix=".sv", dir=str(VCS_HOST_TMP),
                                      prefix="vcs_bench_")
    os.close(fd)
    try:
        Path(host_path).write_text(code)
        ctn_path = host_path.replace(str(VCS_HOST_TMP), VCS_CONTAINER_TMP)
        r = subprocess.run(
            ["docker", "exec", VCS_CONTAINER_NAME,
             "bash", VCS_CONTAINER_SCRIPT, ctn_path],
            capture_output=True, timeout=timeout, text=True,
        )
        if r.returncode == 0:
            return True, ""
        # vcs_check.sh prints the first 2 Error-[..] lines on failure
        err = (r.stdout or "").strip().splitlines()
        msg = err[0] if err else (r.stderr or "")[:160].strip()
        return False, msg[:200]
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except FileNotFoundError:
        return False, "NO_DOCKER"
    except Exception as e:
        return False, f"EXCEPTION:{type(e).__name__}"
    finally:
        try:
            os.unlink(host_path)
        except OSError:
            pass


def _check_one(args):
    idx, mode, sva, ref_sva, rtl, timeout, engine = args
    if mode == "free-rtl":
        code = _build_compileable_unit(sva, ref_sva, rtl)
    else:
        code = build_bind_rtl_unit(sva, ref_sva, rtl)
    if engine == "vcs":
        ok, err = _vcs_run(code, timeout=timeout)
    else:
        ok, err = _verilator_run(code, timeout=timeout)
    return idx, ok, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="")
    ap.add_argument("--mode", choices=["free-rtl", "bind-rtl"], default="free-rtl")
    ap.add_argument("--engine", choices=["verilator", "vcs"], default="verilator",
                    help="verilator (open-source lint) | vcs (Synopsys "
                         "L-2016.06 in docker, full SVA spec — slower)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--timeout", type=int, default=0,
                    help="per-call timeout in seconds; 0 → engine default "
                         "(verilator=8, vcs=20)")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap rows for smoke testing")
    args = ap.parse_args()
    if args.timeout == 0:
        args.timeout = 20 if args.engine == "vcs" else 8

    src = Path(args.input)
    rows = []
    with open(src) as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
            if args.limit and len(rows) >= args.limit:
                break
    print(f"[input] {src.name}  rows={len(rows)}  mode={args.mode}  "
          f"engine={args.engine}  workers={args.workers}  "
          f"timeout={args.timeout}s")

    work = []
    for i, r in enumerate(rows):
        sva = r.get("reference_sva") or ""
        ref = r.get("reference_sva") or ""  # same as sva (this is reference set)
        rtl = r.get("rtl_context") or ""
        work.append((i, args.mode, sva, ref, rtl, args.timeout, args.engine))

    n_ok = 0
    err_counts = Counter()
    by_class_ok = Counter()
    by_class_total = Counter()
    by_origin_ok = Counter()
    by_origin_total = Counter()

    t0 = time.time()
    out_rows = [None] * len(rows)
    with mp.Pool(args.workers) as pool:
        for done, (idx, ok, err) in enumerate(
                pool.imap_unordered(_check_one, work, chunksize=8), 1):
            r = rows[idx]
            cls = str(r.get("temporal_class") or "?")
            origin = r.get("_origin", "unknown")
            by_class_total[cls] += 1
            by_origin_total[origin] += 1
            if ok:
                n_ok += 1
                by_class_ok[cls] += 1
                by_origin_ok[origin] += 1
            else:
                # Bucket the error class
                err_short = err.split(":", 2)[-1].strip()[:80] if err else "(empty)"
                err_counts[err[:60] or "(empty)"] += 1
            out_rows[idx] = {
                "id": r.get("id"),
                "ok": ok,
                "err": err if not ok else "",
                "tcl_class": cls,
                "origin": origin,
            }
            if done % 1000 == 0 or done == len(rows):
                rate = done / (time.time() - t0)
                eta = (len(rows) - done) / max(rate, 1e-3)
                print(f"[bench] {done}/{len(rows)}  ok={n_ok} ({100*n_ok/done:.1f}%)  "
                      f"{rate:.1f}/s  eta={eta:.0f}s")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            for r in out_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[bench] dumped per-row results to {args.output}")

    print("\n=== Summary ===")
    print(f"total : {len(rows)}")
    print(f"OK    : {n_ok} ({100*n_ok/max(len(rows),1):.1f}%)")
    print(f"FAIL  : {len(rows)-n_ok}")

    print(f"\nBy temporal_class:")
    for cls in sorted(by_class_total):
        tot = by_class_total[cls]; o = by_class_ok[cls]
        print(f"  {cls:<6} ok={o:>5} / {tot:<5} ({100*o/max(tot,1):.1f}%)")

    print(f"\nBy origin:")
    for origin in sorted(by_origin_total):
        tot = by_origin_total[origin]; o = by_origin_ok[origin]
        print(f"  {origin:<15} ok={o:>5} / {tot:<5} ({100*o/max(tot,1):.1f}%)")

    print(f"\nTop 15 error patterns:")
    for e, c in err_counts.most_common(15):
        print(f"  {c:>5}  {e}")


if __name__ == "__main__":
    main()
