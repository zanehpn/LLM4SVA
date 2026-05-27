"""vcs_dynamic_check.py
Differential simulation reward: drives random stimuli to a free-input
testbench that runs both the reference SVA and the LM-generated SVA as
concurrent assertions, counts each one's failure rate over N cycles,
and returns an [0, 1] agreement score.

Usage:
    from vcs_dynamic_check import vcs_dynamic_check
    score = vcs_dynamic_check(sva_gen, sva_ref, rtl_context)
    # score in [0, 1] or None if sim failed

Use case: GRPO reward fallback when the formal PEC oracle returns
UNSUPPORTED / TIMEOUT / PARSE_ERROR. About 50% of rollouts in the
current GRPO setup hit this floor (verdict cannot be decided by sby/z3),
producing a binary 0.15 reward with zero variance. Replacing it with a
continuous dynamic-agreement signal restores GRPO advantage estimation.
"""
from __future__ import annotations
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import importlib.util
_g_spec = importlib.util.spec_from_file_location(
    "_g", str(ROOT / "training" / "rlvf" / "run_grpo_pilot.py"))
_g = importlib.util.module_from_spec(_g_spec)
_g_spec.loader.exec_module(_g)
_identifiers_in_sva = _g._identifiers_in_sva
_classify_identifiers = _g._classify_identifiers


HOST_TMP = ROOT / "runs" / "vcs_tmp"
CONTAINER_TMP = "/work/runs/vcs_tmp"
CONTAINER_SCRIPT = "/work/scripts/vcs_dynamic_check.sh"
CONTAINER_NAME = "vcs_lint"


_AP_RE = re.compile(
    r"\b(?:assert|assume|cover)\s+property\s*\((.+)\)\s*;?\s*$",
    re.DOTALL | re.IGNORECASE)


def _extract_property_body(sva: str) -> str:
    """Peel off the outer `assert property (...);` to get the inner
    property expression. Returns the original string if the SVA doesn't
    match (caller should treat that as malformed)."""
    s = (sva or "").strip()
    s = s.rstrip(";").strip()
    m = _AP_RE.search(s)
    if m:
        return m.group(1).strip()
    return s


def _build_dynamic_tb(ref_body: str, gen_body: str,
                      ids: set, n_cycles: int = 200) -> str:
    """Generate a SystemVerilog testbench that drives random stimuli to
    every identifier and counts assertion-failure rates for both ref and
    gen properties."""
    sva_text = ref_body + " " + gen_body
    params, funcs, ports, mdas = _classify_identifiers(ids, sva_text)

    port_decls = []
    for ident in sorted(ports):
        port_decls.append(f"  logic [31:0] {ident};")
    for ident in sorted(mdas):
        port_decls.append(f"  logic [31:0][31:0][31:0] {ident};")

    body_decls = []
    for ident in sorted(params):
        body_decls.append(f"  parameter logic [31:0] {ident} = 32'd4;")
    for ident in sorted(funcs):
        body_decls.append(
            f"  function automatic logic [31:0] {ident}(\n"
            f"    input logic [31:0] a0 = 32'd0, input logic [31:0] a1 = 32'd0,\n"
            f"    input logic [31:0] a2 = 32'd0, input logic [31:0] a3 = 32'd0,\n"
            f"    input logic [31:0] a4 = 32'd0, input logic [31:0] a5 = 32'd0,\n"
            f"    input logic [31:0] a6 = 32'd0, input logic [31:0] a7 = 32'd0);\n"
            f"    return $urandom();\n"
            f"  endfunction"
        )

    # Drive each signal as a 32-bit value where the *value* is either 0
    # or 1 (with 50/50 prob). SystemVerilog's bool cast on multi-bit means
    # `if (sig)` is true iff sig is non-zero; under 32-bit $urandom() the
    # signal is non-zero with prob ~1, making implication antecedents
    # nearly always true and consequents nearly always satisfied (= no
    # discrimination). Driving with `$urandom_range(0, 1)` gives 50/50 on
    # the bool cast and exposes the antecedent/consequent structure to
    # the assertion. Multi-bit value equality checks (`x == 32'd5`) lose
    # informativeness, but those are rare in our SVA pool.
    drives = []
    for ident in sorted(ports):
        drives.append(f"      {ident} <= $urandom_range(0, 1);")
    drive_block = "\n".join(drives) if drives else "      ;"

    return f"""\
`timescale 1ns/1ps
module dyn_check;
  logic clk = 0;
  logic tb_reset = 1;
{chr(10).join(port_decls)}
{chr(10).join(body_decls)}

  always #5 clk = ~clk;

  int n_cycles = 0;
  int n_ref_fail = 0;
  int n_gen_fail = 0;

  property p_ref; {ref_body}; endproperty
  property p_gen; {gen_body}; endproperty

  ref_a: assert property (p_ref) else n_ref_fail++;
  gen_a: assert property (p_gen) else n_gen_fail++;

  initial begin
    tb_reset = 1; #25;
    tb_reset = 0;

    repeat({n_cycles}) @(posedge clk) begin
{drive_block}
      n_cycles++;
    end

    $display("AGREEMENT %0d %0d %0d", n_cycles, n_ref_fail, n_gen_fail);
    $finish;
  end

  // Wallclock safety net
  initial begin
    #50000;
    $display("AGREEMENT %0d %0d %0d", n_cycles, n_ref_fail, n_gen_fail);
    $finish;
  end
endmodule
"""


def _ensure_tmp():
    HOST_TMP.mkdir(parents=True, exist_ok=True)


def vcs_dynamic_check(sva_gen: str, sva_ref: str,
                       rtl_context: str = "",
                       n_cycles: int = 200,
                       timeout: int = 25) -> float | None:
    """Return an agreement score in [0, 1] or None if simulation failed.

    Score formula:
      * Both fire 0 times                    -> 1.0  (vacuously agree)
      * Equal nonzero fire counts            -> 1.0
      * Otherwise   1 - |fail_ref - fail_gen| / max(fail_ref, fail_gen)
    """
    if not sva_gen or not sva_ref:
        return None

    ref_body = _extract_property_body(sva_ref)
    gen_body = _extract_property_body(sva_gen)
    if not ref_body or not gen_body:
        return None

    exclude = {"clk", "tb_reset"}
    ids = (_identifiers_in_sva(sva_gen, exclude)
           | _identifiers_in_sva(sva_ref, exclude))

    tb_src = _build_dynamic_tb(ref_body, gen_body, ids, n_cycles=n_cycles)

    _ensure_tmp()
    fd, host_path = tempfile.mkstemp(suffix=".sv", dir=str(HOST_TMP),
                                      prefix="vcs_dyn_")
    os.close(fd)
    try:
        Path(host_path).write_text(tb_src)
        ctn_path = host_path.replace(str(HOST_TMP), CONTAINER_TMP)
        r = subprocess.run(
            ["docker", "exec", CONTAINER_NAME, "bash",
             CONTAINER_SCRIPT, ctn_path],
            capture_output=True, timeout=timeout, text=True,
        )
        out = (r.stdout or "") + (r.stderr or "")
        m = re.search(r"AGREEMENT\s+(\d+)\s+(\d+)\s+(\d+)", out)
        if not m:
            return None
        cycles, fail_ref, fail_gen = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if cycles == 0:
            return None
        if fail_ref == 0 and fail_gen == 0:
            return 1.0
        denom = max(fail_ref, fail_gen)
        if denom == 0:
            return 1.0
        return 1.0 - abs(fail_ref - fail_gen) / denom
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    except Exception:
        return None
    finally:
        try:
            os.unlink(host_path)
        except OSError:
            pass


if __name__ == "__main__":
    # Self-test
    test_cases = [
        ("identical",
         "assert property (@(posedge clk) a |=> b);",
         "assert property (@(posedge clk) a |=> b);",
         "expected: 1.0"),
        ("different but related (a|->b vs a|=>b)",
         "assert property (@(posedge clk) a |=> b);",
         "assert property (@(posedge clk) a |-> b);",
         "expected: lower (different timing)"),
        ("totally different",
         "assert property (@(posedge clk) c |=> d);",
         "assert property (@(posedge clk) a |=> b);",
         "expected: lower (different signals)"),
    ]
    for name, gen, ref, exp in test_cases:
        r = vcs_dynamic_check(gen, ref, "", n_cycles=100, timeout=20)
        print(f"{name:40s}  agreement={r}  ({exp})")
