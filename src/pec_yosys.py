"""
pec_yosys.py — open-source Property Equivalence Checker.

Replicates the function of Cadence JasperGold's `prop_eq_checker` using
SymbiYosys + yosys-slang + z3 (BMC engine). Determines whether an
LM-generated SVA is functionally equivalent to a reference SVA over the
given RTL context.

Algorithm (the standard PEC encoding):
  Direction ①  (ref → lm) :
        assume the reference SVA holds, then assert the LM SVA
        — if BMC finds a counterexample, lm is *more permissive* than ref
  Direction ②  (lm → ref) :
        swap roles
        — if BMC finds a counterexample, lm is *more strict* than ref

  Verdict matrix:
    ① PASS,  ② PASS  → EQUIVALENT
    ① PASS,  ② FAIL  → IMPLIES_REF_TO_LM   (lm covers ref but is stricter)
    ① FAIL,  ② PASS  → IMPLIES_LM_TO_REF   (lm is weaker than ref)
    ① FAIL,  ② FAIL  → NOT_EQUIVALENT
    any TIMEOUT/PARSE_ERROR/UNSUPPORTED → propagated as-is

This matches CodeV / FVEval's `functionality` metric (full bidirectional
equivalence) and `func_relaxed` metric (either direction implies).

Usage:
    from src.pec_yosys import prop_equivalence
    r = prop_equivalence(sva_lm, sva_ref, rtl_module, depth=20, timeout=30)
    print(r.verdict, r.fwd_status, r.bwd_status)
"""
from __future__ import annotations
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .sva_lowering import (
    _extract_body, _emit_immediate_assert, _wrap_always_ff,
    inject_lowered, lower_sva, strip_concurrent_asserts,
)


# -----------------------------------------------------------------------
# Cadence-aligned canonicalization
# -----------------------------------------------------------------------
# 4-state case-equality `===`/`!==` is indistinguishable from logical
# `==`/`!=` under Symbiyosys's default 2-state elaboration, but keeping the
# operators around can still confuse the lowering surface. Normalize both
# sides symmetrically so one side using `!==` and the other `!=` never
# masquerades as a semantic difference.
_CASE_EQ_RE = re.compile(r"([!=])==")


def _normalize_case_eq(sva: str) -> str:
    return _CASE_EQ_RE.sub(r"\1=", sva) if sva else sva


_DISABLE_IFF_RE = re.compile(r"disable\s+iff\s*\(([^)]+)\)")
_ASSIGN_RESET_RE = re.compile(
    r"assign\s+(tb_reset|rst_int|rst_sync|reset_sync)\s*=", re.IGNORECASE
)
_COMMON_RESET_NAMES = ("tb_reset", "rst_n", "reset_n", "rst", "reset", "reset_")


def infer_reset_expr(rtl_ctx: str,
                     ref_sva: str = "",
                     lm_sva: str = "") -> Optional[str]:
    """Best-effort recovery of the module's reset signal for Cadence-style
    BMC canonicalization.

    Priority:
      1. `disable iff (X)` appearing on either SVA side → use X.
      2. `assign <name> = …` of a known reset-alias wire in rtl_context.
      3. A common reset name declared in rtl_context (port/signal).
    Returns None when no signal is found; caller should skip reset-gating.
    """
    m = (_DISABLE_IFF_RE.search(ref_sva or "")
         or _DISABLE_IFF_RE.search(lm_sva or ""))
    if m:
        return m.group(1).strip()
    m = _ASSIGN_RESET_RE.search(rtl_ctx or "")
    if m:
        return m.group(1)
    for name in _COMMON_RESET_NAMES:
        if re.search(rf"\b{name}\b", rtl_ctx or ""):
            return f"!{name}" if name.endswith("_n") else name
    return None


# Paper §4.3 / App. E specifies five public verdicts. Internal failure
# modes (PARSE_ERROR / TIMEOUT / EXTRACT_ERROR / UNKNOWN / EMPTY) are
# collapsed onto UNSUPPORTED by `public_verdict(...)` so external consumers
# see the paper's 5-verdict surface while debug logs keep the fine grain.
PUBLIC_VERDICTS = {
    "EQUIVALENT", "IMPLIES_REF_TO_LM", "IMPLIES_LM_TO_REF",
    "NOT_EQUIVALENT", "UNSUPPORTED",
}


def public_verdict(verdict: str) -> str:
    """Collapse internal verdicts onto the paper's 5-verdict surface."""
    return verdict if verdict in PUBLIC_VERDICTS else "UNSUPPORTED"


@dataclass
class EquivResult:
    verdict: str            # one of PUBLIC_VERDICTS (paper §4.3, App. E)
    internal_status: str = ""  # fine-grained failure reason if verdict
                               # collapsed onto UNSUPPORTED: PARSE_ERROR,
                               # TIMEOUT, EXTRACT_ERROR, UNKNOWN, EMPTY
    fwd_status: str = ""    # status of `assume(ref); assert(lm)` BMC
    bwd_status: str = ""    # status of `assume(lm); assert(ref)` BMC
    wallclock_s: float = 0.0
    fwd_module: str = ""    # for debug
    bwd_module: str = ""
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "fwd_status": self.fwd_status,
            "bwd_status": self.bwd_status,
            "wallclock_s": round(self.wallclock_s, 2),
            "notes": list(self.notes),
        }


# -----------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------
def _lowered_to_pec_fragment(sva: str, prefix: str, verb: str,
                             liveness_bound: Optional[int] = None) -> Optional[str]:
    """Lower the SVA and rewrite its emitted `assert (...)` to `<verb> (...)`,
    plus rename helper vars so two PEC arms can coexist in one module.

    Returns the fragment text or None if lowering failed."""
    lo = lower_sva(sva, helper_prefix=prefix, liveness_bound=liveness_bound)
    if not lo["ok"]:
        return None
    text = lo["lowered"]
    # Lowering hard-codes `svlow_inited` regardless of helper_prefix — rename
    # so the two PEC arms each get their own init flag.
    inited_name = f"{prefix}_inited"
    text = text.replace("svlow_inited", inited_name)
    # The lowering always emits `assert (...)`. Convert to the requested verb
    # (assume / assert / cover) — only the *guarded statement* should change,
    # not the prefix-substring `assert` accidentally matched elsewhere. The
    # lowering always emits exactly `<spaces>assert (` after the inited guard.
    text = re.sub(r"\bassert\s*\(", f"{verb} (", text)
    return text


def _build_two_arm_module(rtl: str, ref_sva: str, lm_sva: str,
                          assume_first: str,
                          reset_expr: Optional[str] = None,
                          liveness_bound: Optional[int] = None) -> Optional[str]:
    """Build an RTL module where one SVA is an `assume` and the other an
    `assert`, used to check directional implication.

    `assume_first` ∈ {"ref", "lm"}:
        "ref"  → assume(ref); assert(lm)   (direction ①: ref → lm)
        "lm"   → assume(lm);  assert(ref)  (direction ②: lm → ref)

    If `reset_expr` is given, a clocked `assume(!(reset_expr))` is injected
    so BMC never exercises reset cycles — mirrors JasperGold's
    `reset -expression` declaration and keeps the `disable iff` asymmetry
    from showing up as a spurious directional failure.
    """
    # Strip any pre-existing concurrent assertions from the RTL module
    stripped = strip_concurrent_asserts(rtl)
    if "endmodule" not in stripped:
        return None

    if assume_first == "ref":
        a_sva, a_pfx = ref_sva, "pec_a"
        b_sva, b_pfx = lm_sva, "pec_b"
    else:
        a_sva, a_pfx = lm_sva, "pec_a"
        b_sva, b_pfx = ref_sva, "pec_b"

    a_frag = _lowered_to_pec_fragment(a_sva, a_pfx, verb="assume",
                                      liveness_bound=liveness_bound)
    b_frag = _lowered_to_pec_fragment(b_sva, b_pfx, verb="assert",
                                      liveness_bound=liveness_bound)
    if a_frag is None or b_frag is None:
        return None

    # Re-use sva_lowering.inject_lowered's clock-alias + endmodule plumbing
    # by passing the combined fragment through.
    parsed_a = _extract_body(a_sva) or {}
    clk_in_sva = parsed_a.get("clk", "clk")

    reset_frag = ""
    if reset_expr:
        reset_frag = (
            f"  always_ff @(posedge {clk_in_sva}) "
            f"assume (!({reset_expr}));\n"
        )

    combined = reset_frag + a_frag + "\n" + b_frag
    return inject_lowered(stripped, combined, clk_in_sva=clk_in_sva)


def _write_sby(work: Path, module_src: str, depth: int):
    (work / "m.sv").write_text(module_src)
    (work / "check.sby").write_text(f"""[options]
mode bmc
depth {depth}

[engines]
smtbmc z3

[script]
plugin -i slang
read_slang m.sv
prep

[files]
m.sv
""")


def _run_bmc(module_src: str, depth: int, timeout: int) -> tuple:
    """Run a single BMC. Returns (status, log_tail)."""
    work = Path(tempfile.mkdtemp(prefix="sva_pec_"))
    try:
        _write_sby(work, module_src, depth)
        try:
            proc = subprocess.run(
                ["sby", "-f", "check.sby"],
                cwd=str(work), capture_output=True, text=True,
                timeout=timeout,
            )
            log = (proc.stdout or "") + "\n" + (proc.stderr or "")
        except subprocess.TimeoutExpired as e:
            log = (e.stdout or b"").decode("utf-8", errors="ignore") + "\n(TIMEOUT)"
            return "TIMEOUT", log[-1500:]

        # Parse — same patterns formal_verify uses
        parse_err_patterns = [
            r"ERROR: Compilation failed",
            r"ERROR: (?:syntax|elaborat|compile|SLANG|No such command)",
            r"error: unknown system name",
            r"error: use of undeclared identifier",
            r"error: could not find module",
            r"Build failed",
        ]
        for pat in parse_err_patterns:
            if re.search(pat, log, re.IGNORECASE):
                return "PARSE_ERROR", log[-1500:]
        if "engine_0 (smtbmc z3) returned PASS" in log or \
           re.search(r"summary:\s+engine_0[^\n]*pass\b", log, re.I):
            return "PASS", log[-500:]
        if "engine_0 (smtbmc z3) returned FAIL" in log or \
           "returned FAIL for basecase" in log:
            return "FAIL", log[-500:]
        return "UNKNOWN", log[-1500:]
    finally:
        shutil.rmtree(work, ignore_errors=True)


# -----------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------
def prop_equivalence(
    sva_lm: str,
    sva_ref: str,
    rtl_module: str,
    depth: int = 20,
    timeout: int = 30,
    reset_expr: Optional[str] = None,
    liveness_bound: Optional[int] = None,
) -> EquivResult:
    """Check whether sva_lm is functionally equivalent to sva_ref over rtl.

    `reset_expr`: optional Verilog expression for the reset signal. When
    provided, BMC is constrained to traces where `!(reset_expr)` always
    holds — matches Cadence JasperGold's `reset -expression {...}` setup
    so that `disable iff` wrapping on one side vs the other is not scored
    as a directional implication.

    `liveness_bound`: optional BMC cycle bound for the bounded-liveness
    rewrite (`s_eventually` / `nexttime` / `s_always`). Applied
    symmetrically to both ref and lm. None means keep liveness operators
    as UNSUPPORTED. Typical value: match or slightly exceed `depth`.
    """
    t0 = time.time()

    # Symmetric canonicalization: 4-state case-equality is evaluated the
    # same as logical equality under 2-state Symbiyosys anyway; collapse
    # both sides so any residual surface mismatch never shows up.
    sva_lm = _normalize_case_eq(sva_lm)
    sva_ref = _normalize_case_eq(sva_ref)

    # Lowering coverage gate — bail out early on UNSUPPORTED so we don't
    # waste BMC time on obviously-untranslatable SVAs. With liveness_bound
    # set, `s_eventually` / `nexttime` / `s_always` are rewritten into
    # bounded form before this gate runs; the remaining unsupported
    # operators (`s_until`, `throughout`, repeats, etc.) still bail.
    lo_lm = lower_sva(sva_lm, liveness_bound=liveness_bound)
    lo_ref = lower_sva(sva_ref, liveness_bound=liveness_bound)
    if not lo_lm["ok"]:
        return EquivResult(
            verdict="UNSUPPORTED",
            wallclock_s=time.time() - t0,
            notes=[f"lm unsupported: {lo_lm.get('notes', [])}"],
        )
    if not lo_ref["ok"]:
        return EquivResult(
            verdict="UNSUPPORTED",
            wallclock_s=time.time() - t0,
            notes=[f"ref unsupported: {lo_ref.get('notes', [])}"],
        )

    fwd_module = _build_two_arm_module(rtl_module, sva_ref, sva_lm,
                                       assume_first="ref",
                                       reset_expr=reset_expr,
                                       liveness_bound=liveness_bound)
    bwd_module = _build_two_arm_module(rtl_module, sva_ref, sva_lm,
                                       assume_first="lm",
                                       reset_expr=reset_expr,
                                       liveness_bound=liveness_bound)
    if fwd_module is None or bwd_module is None:
        return EquivResult(
            verdict="UNSUPPORTED",
            internal_status="EXTRACT_ERROR",
            wallclock_s=time.time() - t0,
            notes=["could not inject two-arm module into RTL"],
        )

    fwd_status, fwd_log = _run_bmc(fwd_module, depth, timeout)
    bwd_status, bwd_log = _run_bmc(bwd_module, depth, timeout)

    # Verdict matrix (paper §4.3 / App. E). Internal BMC failure modes
    # (PARSE_ERROR / TIMEOUT / anything else) collapse onto UNSUPPORTED;
    # the fine-grained reason is preserved in `internal_status` for debug.
    internal_status = ""
    if fwd_status == "PASS" and bwd_status == "PASS":
        verdict = "EQUIVALENT"
    elif fwd_status == "PASS" and bwd_status == "FAIL":
        verdict = "IMPLIES_REF_TO_LM"   # lm is stricter
    elif fwd_status == "FAIL" and bwd_status == "PASS":
        verdict = "IMPLIES_LM_TO_REF"   # lm is more permissive
    elif fwd_status == "FAIL" and bwd_status == "FAIL":
        verdict = "NOT_EQUIVALENT"
    elif "PARSE_ERROR" in (fwd_status, bwd_status):
        verdict, internal_status = "UNSUPPORTED", "PARSE_ERROR"
    elif "TIMEOUT" in (fwd_status, bwd_status):
        verdict, internal_status = "UNSUPPORTED", "TIMEOUT"
    else:
        verdict, internal_status = "UNSUPPORTED", "UNKNOWN"

    return EquivResult(
        verdict=verdict,
        internal_status=internal_status,
        fwd_status=fwd_status,
        bwd_status=bwd_status,
        wallclock_s=time.time() - t0,
        fwd_module=fwd_module,
        bwd_module=bwd_module,
        notes=[fwd_log[-300:], bwd_log[-300:]],
    )


# -----------------------------------------------------------------------
# Smoke test
# -----------------------------------------------------------------------
if __name__ == "__main__":
    sample_rtl = """\
module top (input logic clk, input logic a, input logic b, input logic c);
endmodule
"""
    cases = [
        # Self-reflexivity (must EQUIV)
        ("assert property (@(posedge clk) a |-> b);",
         "assert property (@(posedge clk) a |-> b);",
         "EQUIVALENT"),
        # Swap LHS/RHS — must NOT_EQUIV (with free a, b)
        ("assert property (@(posedge clk) a |-> b);",
         "assert property (@(posedge clk) b |-> a);",
         "NOT_EQUIVALENT"),
        # |=> vs |-> — different cycle, must NOT_EQUIV
        ("assert property (@(posedge clk) a |-> b);",
         "assert property (@(posedge clk) a |=> b);",
         "NOT_EQUIVALENT"),
        # Logical rewriting — should still EQUIV
        ("assert property (@(posedge clk) (a && b) |-> c);",
         "assert property (@(posedge clk) (b && a) |-> c);",
         "EQUIVALENT"),
    ]
    print("Smoke test (depth=10):")
    print()
    for lm, ref, expected in cases:
        r = prop_equivalence(lm, ref, sample_rtl, depth=10, timeout=20)
        ok = "✓" if r.verdict == expected else "✗"
        print(f"  {ok}  expected={expected:<22s}  got={r.verdict:<22s}  "
              f"({r.fwd_status}/{r.bwd_status}, {r.wallclock_s:.1f}s)")
        print(f"     LM:  {lm}")
        print(f"     REF: {ref}")
        print()
