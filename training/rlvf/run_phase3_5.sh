#!/bin/bash
# Phase 3.5: combine the strongest axes — FVEval prompt format + extended
# (v2) multi-ref pool + multi-ref reward. Run after the main v2 chain.
set -eu
cd ${REPO_ROOT}
LOG=${REFINE_LOGS}/phase2

echo "=== [$(date +%H:%M:%S)] Phase 3.5 GRPO: FVEval format + v2 pool ==="
PATH=${OSS_CAD_SUITE}/bin:$PATH \
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python3 -u \
    scripts/run_grpo_pilot.py \
    --policy results/sft_qwen_coder_7b_replay50/checkpoint_20260420_210826 \
    --output-dir results/grpo_pec_phase3_v2_fveval \
    --reward-mode pec_multiref \
    --pool data/train/grpo/grpo_pool_phase2_v2.jsonl \
    --prompt-format fveval \
    --num-generations 4 --batch-size 4 --max-steps 100 \
    --max-completion-length 256 --lr 1e-6 --beta 0.04 --lora-r 16 --seed 0 \
    > "$LOG/grpo_phase3_5.log" 2>&1
echo "=== [$(date +%H:%M:%S)] Phase 3.5 done ==="
