"""
formal_verify.py — run SymbiYosys over a (SVA, RTL module) pair and return a
graded reward signal suitable for GRPO.

Pipeline:
  1. Lower the SVA to immediate form with src.sva_lowering.lower_sva.
  2. Inject the lowered fragment into a temporary copy of the RTL module
     (stripping any prior concurrent assertions so they don't trip the parser).
  3. Write a .sby config that loads yosys-slang and runs bmc (basecase only)
     with a configurable depth.
  4. Parse sby output → one of {"PASS", "FAIL", "VACUOUS", "TIMEOUT",
     "PARSE_ERROR", "UNSUPPORTED", "EXTRACT_ERROR"}.

Reward mapping (default, tunable):
  PASS            → 1.0      (SVA holds on the RTL for all traces up to depth)
  VACUOUS         → 0.2      (trivially true — likely a bad SVA)
  FAIL            → 0.5      (sat'ble counterexample — still shows coherent
                              SVA; partial credit so RL doesn't collapse to
                              tautology)
  TIMEOUT         → 0.3      (verifier didn't finish; harmless-to-mild penalty)
  PARSE_ERROR     → 0.0      (lowered RTL didn't compile — bad SVA)
  UNSUPPORTED     → None     (pattern not lowerable — caller should fall back
                              to syntax+AST_sim only, not punish the model)
  EXTRACT_ERROR   → 0.0

Usage:
  from src.formal_verify import formal_verify
  result = formal_verify(sva, rtl_module, timeout=15)
  print(result.status, result.reward)
"""
from __future__ import annotations
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from .sva_lowering import lower_sva, inject_lowered
from .vacuity import is_vacuous_syntactic

DEFAULT_BMC_DEPTH = 10
DEFAULT_TIMEOUT_S = 15

# Reward grading
REWARDS = {
    "PASS":          1.0,
    "VACUOUS":       0.2,
    "FAIL":          0.5,
    "TIMEOUT":       0.3,
    "PARSE_ERROR":   0.0,
    "EXTRACT_ERROR": 0.0,
}


@dataclass
class VerifyResult:
    status: str                     # see REWARDS keys + "UNSUPPORTED"
    reward: Optional[float]         # None if UNSUPPORTED (caller falls back)
    pattern: Optional[str] = None   # from lower_sva
    tcl: int = 0
    wallclock_s: float = 0.0
    lowered_sva: str = ""
    sby_tail: str = ""              # last ~15 lines of sby log (diagnostic)
    cex_trace: Optional[str] = None
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "status": self.status, "reward": self.reward,
            "pattern": self.pattern, "tcl": self.tcl,
            "wallclock_s": round(self.wallclock_s, 2),
            "lowered_sva": self.lowered_sva,
            "sby_tail": self.sby_tail,
            "cex_trace_present": self.cex_trace is not None,
            "notes": list(self.notes),
        }


def _cache_key(sva: str, rtl: str) -> str:
    h = hashlib.sha256()
    h.update(re.sub(r"\s+", " ", sva).strip().encode())
    h.update(b"|||")
    h.update(re.sub(r"\s+", " ", rtl).strip().encode())
    return h.hexdigest()[:16]


def _write_sby(dir: Path, module_src: str, depth: int):
    (dir / "m.sv").write_text(module_src)
    (dir / "check.sby").write_text(
        f"""[options]
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


def _parse_sby_output(log: str) -> str:
    """Map sby log text → PASS|FAIL|VACUOUS|TIMEOUT|PARSE_ERROR|UNKNOWN."""
    # Parse / elaboration errors first (these mask everything else)
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
            return "PARSE_ERROR"
    if "engine_0 (smtbmc z3) returned PASS" in log or \
       re.search(r"summary:\s+engine_0[^\n]*pass\b", log, re.I):
        return "PASS"
    if "engine_0 (smtbmc z3) returned FAIL" in log or \
       "returned FAIL for basecase" in log:
        return "FAIL"
    # sby prints this when property is trivially true (cover cannot be hit)
    if "unreachable" in log.lower() and "cover" in log.lower():
        return "VACUOUS"
    return "UNKNOWN"


def formal_verify(
    sva: str,
    rtl_module: str,
    timeout: int = DEFAULT_TIMEOUT_S,
    depth: int = DEFAULT_BMC_DEPTH,
    cache_dir: Optional[Path] = None,
    keep_workdir: bool = False,
) -> VerifyResult:
    """Run SymbiYosys BMC on (sva, rtl_module)."""
    t0 = time.time()

    # 0. Vacuity gate — fail fast on trivially-true SVAs so RL cannot exploit
    #    "assert 1'b1" as a cheap PASS reward.
    vac, vac_reason = is_vacuous_syntactic(sva)
    if vac:
        return VerifyResult(
            status="VACUOUS", reward=REWARDS["VACUOUS"],
            pattern="vacuous", tcl=0,
            wallclock_s=time.time() - t0,
            notes=[f"vacuity detected: {vac_reason}"],
        )

    # 1. Lower SVA
    lo = lower_sva(sva)
    if not lo["ok"]:
        return VerifyResult(
            status="UNSUPPORTED", reward=None,
            pattern=lo.get("pattern"), tcl=lo.get("tcl", 0),
            wallclock_s=time.time() - t0,
            notes=list(lo.get("notes", [])),
        )

    # Extract the clock name used by the lowered SVA (default "clk")
    clk_in_sva = "clk"
    from .sva_lowering import _extract_body
    parsed_sva = _extract_body(sva)
    if parsed_sva:
        clk_in_sva = parsed_sva.get("clk", "clk")

    # 2. Inject into RTL, creating clock alias if needed
    injected = inject_lowered(rtl_module, lo["lowered"], clk_in_sva=clk_in_sva)
    if injected is None:
        return VerifyResult(
            status="EXTRACT_ERROR", reward=REWARDS["EXTRACT_ERROR"],
            pattern=lo.get("pattern"), tcl=lo.get("tcl"),
            wallclock_s=time.time() - t0, lowered_sva=lo["lowered"],
            notes=["no endmodule found in RTL"],
        )

    # 3. Cache hit?
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = _cache_key(sva, rtl_module)
        cpath = cache_dir / f"{key}.json"
        if cpath.exists():
            d = json.load(open(cpath))
            return VerifyResult(
                status=d["status"], reward=d["reward"],
                pattern=d.get("pattern"), tcl=d.get("tcl", 0),
                wallclock_s=0.0, lowered_sva=lo["lowered"],
                notes=["cache hit"] + list(d.get("notes", [])),
            )

    # 4. Run sby
    work = Path(tempfile.mkdtemp(prefix="sva_fv_"))
    try:
        _write_sby(work, injected, depth)
        try:
            proc = subprocess.run(
                ["sby", "-f", "check.sby"],
                cwd=str(work),
                capture_output=True, text=True, timeout=timeout,
            )
            log = (proc.stdout or "") + "\n" + (proc.stderr or "")
            status = _parse_sby_output(log)
        except subprocess.TimeoutExpired as e:
            log = (e.stdout or b"").decode("utf-8", errors="ignore") + "\n(TIMEOUT)"
            status = "TIMEOUT"

        if status == "UNKNOWN":
            status = "FAIL"   # treat UNKNOWN as mild failure rather than reward=1
        reward = REWARDS.get(status, 0.0)

        tail_lines = log.strip().splitlines()
        sby_tail = "\n".join(tail_lines[-15:])

        cex = None
        cex_path = work / "check" / "engine_0" / "trace.vcd"
        if cex_path.exists():
            try:
                cex = cex_path.read_text()[-5000:]
            except Exception:
                pass

        result = VerifyResult(
            status=status, reward=reward,
            pattern=lo.get("pattern"), tcl=lo.get("tcl"),
            wallclock_s=time.time() - t0, lowered_sva=lo["lowered"],
            sby_tail=sby_tail, cex_trace=cex,
            notes=list(lo.get("notes", [])),
        )

        if cache_dir is not None:
            with open(cpath, "w") as f:
                json.dump({
                    "status": status, "reward": reward,
                    "pattern": lo.get("pattern"), "tcl": lo.get("tcl"),
                    "notes": list(lo.get("notes", [])),
                }, f)

        return result
    finally:
        if not keep_workdir:
            shutil.rmtree(work, ignore_errors=True)


# -----------------------------------------------------------------------
# Smoke test
# -----------------------------------------------------------------------
if __name__ == "__main__":
    sample_rtl = """\
module top (input logic clk, input logic req, output logic gnt);
    always_ff @(posedge clk) gnt <= req;
endmodule
"""
    svas = [
        "assert property (@(posedge clk) req |=> gnt);",
        "assert property (@(posedge clk) req |-> gnt);",
        "assert property (@(posedge clk) valid);",     # undefined signal → parse error
        "assert property (@(posedge clk) s_eventually gnt);",  # unsupported
    ]
    for s in svas:
        print("\n---", s)
        r = formal_verify(s, sample_rtl, timeout=10)
        print(json.dumps(r.to_dict(), indent=2))
