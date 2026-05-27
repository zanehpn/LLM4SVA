#!/bin/bash
# Run all 5 model evals on NL2SVA-Human held-out (79 tasks) with the unified
# FVEval-style protocol (rtl_context + wrapped NL + extract_sva).
#
# Usage:  CUDA_VISIBLE_DEVICES=0 bash scripts/eval_all_nl2sva_human.sh
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=.

LOG_DIR=logs
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)

run() {
    local tag="$1" ; shift
    local lf="$LOG_DIR/eval_${tag}_${TS}.log"
    echo
    echo "============================================================"
    echo "[$(date +%H:%M:%S)] === $tag ==="
    echo "============================================================"
    # shellcheck disable=SC2068
    python3 scripts/run_eval_nl2sva_human.py "$@" --tag "$tag" 2>&1 | tee "$lf"
    echo "[$(date +%H:%M:%S)] === $tag DONE ==="
}

# 1. CodeV-SVA-14B (reasoning model, needs large max-new-tokens)
run "CodeV-SVA-14B" \
    --model ${TEACHER_MODEL} \
    --device cuda:0 --dtype bf16 --max-new-tokens 8192

# 2. Qwen2.5-Coder-7B-Instruct zero-shot baseline
run "Qwen2.5-Coder-7B-Instruct_zero-shot" \
    --model ${STUDENT_MODEL} \
    --device cuda:0 --dtype bf16 --max-new-tokens 1024

# 3. Qwen2.5-Coder-7B + SFT (replay 20%)
run "Qwen2.5-Coder-7B_SFT_replay20" \
    --model results/sft_qwen_coder_7b/checkpoint_20260420_201925 \
    --device cuda:0 --dtype bf16 --max-new-tokens 1024

# 4. Qwen2.5-Coder-7B + SFT (replay 50%)
run "Qwen2.5-Coder-7B_SFT_replay50" \
    --model results/sft_qwen_coder_7b_replay50/checkpoint_20260420_210826 \
    --device cuda:0 --dtype bf16 --max-new-tokens 1024

# 5. GRPO LoRA on top of SFT replay20
run "Qwen2.5-Coder-7B_SFT_replay20+grpo_pilot_1" \
    --model results/sft_qwen_coder_7b/checkpoint_20260420_201925 \
    --adapter results/grpo_pilot_1/final \
    --device cuda:0 --dtype bf16 --max-new-tokens 1024

echo
echo "ALL_EVALS_DONE @ $(date +%H:%M:%S)"
