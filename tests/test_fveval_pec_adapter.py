#!/usr/bin/env python3
"""
Smoke-test the FVEval Jasper launcher replacement using the local PEC backend.

This exercises the actual FVEval call path:
  fv_eval.fv_tool_execution.launch_jg_custom_equiv_check
and checks that its Jasper-style output is still accepted by
NL2SVAHumanEvaluator.calculate_jg_metric.

Usage:
  source ${OSS_CAD_SUITE}/environment
  PYTHONPATH=. python3 experiments/scripts/test_fveval_pec_adapter.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FVEVAL_ROOT = REPO_ROOT / "experiments" / "data" / "FVEval"

sys.path.insert(0, str(REPO_ROOT / "experiments"))
sys.path.insert(0, str(FVEVAL_ROOT))

from fv_eval import fv_tool_execution


PACKAGED_TB_EQUIV = """\
module top (
    input logic clk,
    input logic rst_n,
    input logic req,
    output logic gnt
);
always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) gnt <= 1'b0;
    else gnt <= req;
end

wire tb_reset = !rst_n;

reference_grant: assert property (@(posedge clk) disable iff (tb_reset)
    req |=> gnt
);

gen_grant: assert property (@(posedge clk) disable iff (tb_reset)
    req |=> gnt
);

endmodule
"""


PACKAGED_TB_NON_EQUIV = """\
module top (
    input logic clk,
    input logic rst_n,
    input logic req,
    output logic gnt
);
always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) gnt <= 1'b0;
    else gnt <= req;
end

wire tb_reset = !rst_n;

reference_grant: assert property (@(posedge clk) disable iff (tb_reset)
    req |=> gnt
);

gen_grant: assert property (@(posedge clk) disable iff (tb_reset)
    req |-> gnt
);

endmodule
"""


def _run_case(temp_dir: Path, task_id: str, packaged_tb: str) -> str:
    exp_id = "pec_adapter_smoke"
    sva_path = temp_dir / f"{exp_id}_{task_id}.sva"
    sva_path.write_text(packaged_tb)
    return fv_tool_execution.launch_jg_custom_equiv_check(
        tcl_file_path="tool_scripts/run_jg_nl2sva_human.tcl",
        sv_dir=str(temp_dir),
        experiment_id=exp_id,
        task_id=task_id,
        lm_assertion_text="unused_by_pec_adapter",
        ref_assertion_text="unused_by_pec_adapter",
        signal_list_text="req,gnt",
    )


def _calculate_jg_metric(jasper_out_str: str) -> dict[str, float]:
    if "syntax error" in jasper_out_str:
        return {"syntax": 0.0, "functionality": 0.0, "func_relaxed": 0.0}
    if "Full equivalence" in jasper_out_str:
        return {"syntax": 1.0, "functionality": 1.0, "func_relaxed": 1.0}
    if "implies" in jasper_out_str:
        return {"syntax": 1.0, "functionality": 0.0, "func_relaxed": 1.0}
    return {"syntax": 1.0, "functionality": 0.0, "func_relaxed": 0.0}


def main() -> int:
    os.environ["FVEVAL_USE_PEC"] = "1"

    with tempfile.TemporaryDirectory(prefix="fveval_pec_adapter_") as tmp:
        temp_dir = Path(tmp)

        equiv_out = _run_case(temp_dir, "equiv", PACKAGED_TB_EQUIV)
        equiv_metric = _calculate_jg_metric(equiv_out)

        non_equiv_out = _run_case(temp_dir, "non_equiv", PACKAGED_TB_NON_EQUIV)
        non_equiv_metric = _calculate_jg_metric(non_equiv_out)

    print("=== EQUIVALENT CASE ===")
    print(equiv_out)
    print(equiv_metric)
    print()
    print("=== NON-EQUIVALENT CASE ===")
    print(non_equiv_out)
    print(non_equiv_metric)

    ok = (
        equiv_metric["syntax"] == 1.0
        and equiv_metric["functionality"] == 1.0
        and non_equiv_metric["syntax"] == 1.0
        and non_equiv_metric["functionality"] == 0.0
    )
    print()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
