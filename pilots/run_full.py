"""
Full TemporalSVA protocol runner.

SCALE-BLOCKED: This script documents the complete pipeline but cannot run
the training components locally (no GPU, no formal verifier).

Components and their status:
  [RUNNABLE]       TCL classification
  [RUNNABLE]       WaveformLens constraint extraction
  [RUNNABLE]       SVA generation via GPT-4o-mini
  [RUNNABLE]       Temporal-weighted CE loss unit tests
  [SCALE-BLOCKED]  Curriculum fine-tuning (requires 4× A100, ~300 GPU-hours)
  [SCALE-BLOCKED]  RLVF / GRPO training (requires formal verifier + GPU)
  [SCALE-BLOCKED]  SymbiYosys formal verification
  [SCALE-BLOCKED]  SVA-TempBench full eval (requires fine-tuned model)
  [SCALE-BLOCKED]  Industrial case study on RocketChip/BOOM
"""

import subprocess
import sys
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Pilots historically lived under one scripts/ tree; today they're spread
# across pilots/, eval/, data_pipeline/, ... — resolve the first match.
_PILOT_SEARCH_DIRS = ("pilots", "eval", "data_pipeline", "training/rlvf",
                      "training/curriculum_sft", "training/rwopd",
                      "compile_gate", "tools", "tests")


def run_script(name: str, label: str):
    path = None
    for sub in _PILOT_SEARCH_DIRS:
        cand = os.path.join(REPO_ROOT, sub, name)
        if os.path.exists(cand):
            path = cand
            break
    if path is None:
        path = os.path.join(REPO_ROOT, "pilots", name)  # fall back to original
    print(f"\n{'='*60}")
    print(f"RUNNING: {label}")
    print(f"{'='*60}")
    result = subprocess.run(
        [sys.executable, path],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": REPO_ROOT},
    )
    if result.returncode != 0:
        print(f"WARNING: {name} exited with code {result.returncode}")
    return result.returncode


def print_blocked():
    print("""
=== SCALE-BLOCKED COMPONENTS ===

The following components require infrastructure not available locally:

1. CURRICULUM FINE-TUNING (Component 2)
   Requires: 4× A100 GPUs (~80GB VRAM), PyTorch, HuggingFace Transformers
   Estimated cost: ~300 GPU-hours, 3 days
   Reference impl: src/temporal_loss.py (PYTORCH_REFERENCE variable)

2. RLVF / GRPO TRAINING (Component 3)
   Requires: SymbiYosys or JasperGold + GPU cluster
   Estimated cost: ~180 GPU-hours (GRPO, no critic) + ~250 CPU-hours verifier
   Blocked by: no formal verifier in local environment
   Advantage math (group_relative_advantages) is unit-tested in run_rlvf.py

   Evaluation MUST compare against SoTA per FINAL_PROPOSAL.md:
     - CodeV-SVA-14B (primary, 75.8% NL2SVA-Human)
     - AssertLLM, Hybrid-NL2SVA, GPT-5, DeepSeek-R1
     - AssertFix (repair baseline for WaveformLens)
   Success condition: ≥30pp gain on L4/L5 over CodeV-SVA-14B.

3. FORMAL VERIFICATION
   Requires: SymbiYosys (Linux) or JasperGold (license)
   Mock available: src/mock_verifier.py (syntax check only)
   Real reward components blocked: formal (0.40), vacuity (0.20)

4. SVA-TEMPBENCH FULL EVAL
   Requires: fine-tuned TemporalSVA model + formal verifier
   Can build benchmark data; cannot run model inference

5. INDUSTRIAL CASE STUDY (RocketChip/BOOM)
   Requires: Chisel/FIRRTL toolchain + formal verifier
   RTL available at: https://github.com/chipsalliance/rocket-chip
""")


if __name__ == "__main__":
    print("=== TemporalSVA Full Protocol Runner ===")
    print("Running all locally-executable pilots...\n")

    codes = []
    codes.append(run_script("build_tcl_dataset.py", "Build TCL Dataset"))
    codes.append(run_script("run_tcl_pilot.py", "TCL Classifier Pilot"))
    codes.append(run_script("run_waveform_pilot.py", "WaveformLens Pilot"))
    codes.append(run_script("run_sva_gen_pilot.py", "SVA Generation Pilot"))

    print_blocked()

    failed = sum(1 for c in codes if c != 0)
    print(f"\n{'='*60}")
    print(f"DONE: {len(codes) - failed}/{len(codes)} pilots completed successfully")
    if failed:
        print(f"WARNING: {failed} script(s) failed — check output above")
