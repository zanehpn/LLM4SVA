"""vcs_compile_check.py
Drop-in counterpart to `verilator_compile_check` that uses VCS L-2016.06
inside the `vcs_lint` docker container.

VCS is the only engine that natively accepts the SVA features Verilator
chokes on: ranged delay `##[N:M]`, `s_eventually`, `intersect`,
`throughout`, recursive properties. The container needs the
`/tmp/vcs_shim.so` LD_PRELOAD shim and `--cap-add=SYS_PTRACE` to defeat
glibc-2.27 incompatibility — see experiments/scripts/vcs_shim.c and
docs in vcs_lint.sh.

Usage:
    from vcs_compile_check import vcs_compile_check
    ok = vcs_compile_check(sva, reference_sva, rtl_context)

Architecture: the SV file is written to a host path under
`experiments/runs/vcs_tmp/` which is bind-mounted into the container at
`/work/runs/vcs_tmp/`. A single `docker exec` invokes
`/work/scripts/vcs_check.sh` which compiles in a private /tmp scratch
dir and returns 0 iff VCS emitted no `^Error-[` lines.
"""
from __future__ import annotations
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import importlib.util
_g_spec = importlib.util.spec_from_file_location(
    "_grpo_helpers", str(ROOT / "training" / "rlvf" / "run_grpo_pilot.py"))
_g = importlib.util.module_from_spec(_g_spec)
_g_spec.loader.exec_module(_g)
_build_compileable_unit = _g._build_compileable_unit


HOST_TMP = ROOT / "runs" / "vcs_tmp"
CONTAINER_TMP = "/work/runs/vcs_tmp"
CONTAINER_SCRIPT = "/work/scripts/vcs_check.sh"
CONTAINER_NAME = "vcs_lint"


def _ensure_tmp():
    HOST_TMP.mkdir(parents=True, exist_ok=True)


def vcs_compile_check(sva: str, reference_sva: str = "",
                       rtl_context: str = "",
                       timeout: int = 20) -> bool:
    """Return True iff VCS `-sverilog -assert svaext` accepts the SVA in
    a free-input wrapper module (no `Error-[...]` lines emitted)."""
    if not sva:
        return False
    code = _build_compileable_unit(sva, reference_sva, rtl_context)
    _ensure_tmp()
    fd, host_path = tempfile.mkstemp(suffix=".sv", dir=str(HOST_TMP),
                                      prefix="vcs_chk_")
    os.close(fd)
    try:
        Path(host_path).write_text(code)
        ctn_path = host_path.replace(str(HOST_TMP), CONTAINER_TMP)
        r = subprocess.run(
            ["docker", "exec", CONTAINER_NAME,
             "bash", CONTAINER_SCRIPT, ctn_path],
            capture_output=True, timeout=timeout, text=True,
        )
        return r.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False
    except Exception:
        return False
    finally:
        try:
            os.unlink(host_path)
        except OSError:
            pass


if __name__ == "__main__":
    # Self-test
    good = """assert property (@(posedge clk) a |=> ##[1:5] b);"""
    bad  = """assert property (@(posedge clk) a |-> oops_undeclared)"""
    print("good ->", vcs_compile_check(good, ""))
    print("bad  ->", vcs_compile_check(bad, ""))
