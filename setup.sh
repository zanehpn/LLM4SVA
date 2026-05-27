#!/usr/bin/env bash
# LLM4SVA — environment setup
#
# Installs Python deps, sources the OSS CAD Suite (yosys + sby + z3) used by
# the open Property-Equivalence Checker, and imports the src/ modules.
#
# Environment variables you should set before running this script
# (the placeholders below are referenced from every training / eval entry
# point so the code stays free of hard-coded site paths):
#
#   OSS_CAD_SUITE   path to an OSS CAD Suite install (contains bin/yosys,
#                   bin/sby, bin/z3). Used by the open PEC oracle.
#   TEACHER_MODEL   path to the teacher model checkpoint (CodeV-SVA-14B).
#   STUDENT_MODEL   path to the base student model (Qwen2.5-Coder-7B-Instruct).
#   MODELS_DIR      optional — root directory holding multiple models if you
#                   prefer to point individual scripts at sub-paths.
#   JG_DOCKER_IMAGE / JG_HELPER_SCRIPT — Cadence JG docker image + helper
#                   script (only needed for eval/rescore_funcatk_with_cadence_pec.py).
#   OPENAI_API_KEY  needed by the NL backfill scripts in data_pipeline/.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== LLM4SVA setup ==="
echo "Working directory: $SCRIPT_DIR"

# Check Python
PYTHON=$(command -v python3 || true)
if [ -z "$PYTHON" ]; then
    echo "ERROR: python3 not found"
    exit 1
fi
echo "Python: $($PYTHON --version)"

# Check OPENAI_API_KEY
if [ -z "${OPENAI_API_KEY:-}" ]; then
    if [ -f ~/.openai_key ]; then
        export OPENAI_API_KEY="$(cat ~/.openai_key)"
        echo "Loaded OPENAI_API_KEY from ~/.openai_key"
    else
        echo "WARNING: OPENAI_API_KEY not set. GPT-based pilots will fail."
        echo "  Set it with: export OPENAI_API_KEY=your_key_here"
    fi
else
    echo "OPENAI_API_KEY: set (${#OPENAI_API_KEY} chars)"
fi

# Install deps
echo ""
echo "Installing Python dependencies..."
$PYTHON -m pip install -q -r requirements.txt
echo "Dependencies installed."

# Output directories (gitignored)
mkdir -p data results runs logs

# OSS CAD Suite — provides sby + yosys + z3 for GRPO-RLVF formal reward
OSS_CAD_ENV="${OSS_CAD_SUITE}/environment"
if [ -f "$OSS_CAD_ENV" ]; then
    # shellcheck disable=SC1090
    source "$OSS_CAD_ENV"
    echo "OSS CAD Suite: $(sby --help 2>&1 | head -1 | cut -c1-60)"
    echo "  yosys: $(yosys -V 2>&1 | head -1)"
else
    echo "WARNING: OSS CAD Suite not found at $OSS_CAD_ENV"
    echo "  RLVF formal reward will be BLOCKED until sby+yosys are on PATH."
fi

# Verify src imports
echo ""
echo "Checking src modules..."
PYTHONPATH="$SCRIPT_DIR" $PYTHON -c "
from src.tcl import classify_tcl
from src.sva_parser import parse_sva
from src.temporal_loss import run_all_tests
from src.waveform_lens import extract_temporal_constraints
from src.mock_verifier import MockVerifier
print('All src modules import OK')
"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps (see README.md and docs/ for the full pipeline):"
echo "  1. TCL classifier self-test:  python3 -c 'from src.tcl import _self_test; _self_test()'"
echo "  2. Temporal-token-weighted CE tests:  python3 src/temporal_loss.py"
echo "  3. Curriculum SFT (§4.2):     python3 training/curriculum_sft/run_curriculum_sft_v2.py --help"
echo "  4. OPD training (§4.1):       python3 training/rwopd/run_opd_codev_to_qwen.py --help"
echo "  5. GRPO RLVF baseline (§4.4): python3 training/rlvf/run_grpo_pilot.py --help"
echo "  6. NL2SVA-Human eval (§5):    python3 eval/run_funcatk_eval.py --help"
