#!/bin/bash
# Run PEC eval over all 5 model eval JSONs sequentially.
set -u
cd "$(dirname "$0")/.."
source ${OSS_CAD_SUITE}/environment
export PYTHONPATH=.

LOG_DIR=logs
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)

EVAL_FILES=(
    "results/eval_nl2sva_human_CodeV-SVA-14B_20260421_095600.json"
    "results/eval_nl2sva_human_Qwen2.5-Coder-7B-Instruct_zero-shot_20260421_114243.json"
    "results/eval_nl2sva_human_Qwen2.5-Coder-7B_SFT_replay20_20260421_114329.json"
    "results/eval_nl2sva_human_Qwen2.5-Coder-7B_SFT_replay50_20260421_114412.json"
    "results/eval_nl2sva_human_Qwen2.5-Coder-7B_SFT_replay20+grpo_pilot_1_20260421_114457.json"
)

for f in "${EVAL_FILES[@]}"; do
    base=$(basename "$f" .json)
    lf="$LOG_DIR/pec_${base}_${TS}.log"
    echo
    echo "============================================================"
    echo "[$(date +%H:%M:%S)] === PEC eval: $base ==="
    echo "============================================================"
    python3 scripts/run_eval_with_pec.py \
        --eval-result "$f" \
        --workers 8 --depth 15 --timeout 30 \
        2>&1 | tee "$lf"
done

echo
echo "ALL_PEC_EVALS_DONE @ $(date +%H:%M:%S)"
