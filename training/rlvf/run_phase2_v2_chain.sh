#!/bin/bash
# Pipeline:
#   1. Stage C — PEC-verify extended alts (CPU, ~1 min, 32 workers)
#   2. Phase 2.5 GRPO training on grpo_pool_phase2_v2.jsonl (simple format)
#   3. Phase 3   GRPO training on grpo_pool_phase2.jsonl     (fveval format)
# Each step short-circuits the chain on failure (set -e).
set -eu
cd ${REPO_ROOT}
LOG=${REFINE_LOGS}/phase2

# Step 1 — Stage C PEC verify
echo "=== [$(date +%H:%M:%S)] Stage C: PEC-verify extended alts ==="
PATH=${OSS_CAD_SUITE}/bin:$PATH PYTHONPATH=. python3 -u \
    scripts/extend_phase2_alts.py --stage C --workers 32 \
    > "$LOG/extend_stage_C.log" 2>&1

# Step 2 — Phase 2.5 GRPO (simple format, multi-ref reward, v2 pool)
echo "=== [$(date +%H:%M:%S)] Phase 2.5 GRPO on grpo_pool_phase2_v2 ==="
PATH=${OSS_CAD_SUITE}/bin:$PATH \
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python3 -u \
    scripts/run_grpo_pilot.py \
    --policy results/sft_qwen_coder_7b_replay50/checkpoint_20260420_210826 \
    --output-dir results/grpo_pec_phase2_v2 \
    --reward-mode pec_multiref \
    --pool data/train/grpo/grpo_pool_phase2_v2.jsonl \
    --num-generations 4 --batch-size 4 --max-steps 100 \
    --max-completion-length 192 --lr 1e-6 --beta 0.04 --lora-r 16 --seed 0 \
    > "$LOG/grpo_phase2_v2.log" 2>&1

# Step 3 — Phase 3 GRPO (FVEval format, multi-ref reward, v1 pool)
echo "=== [$(date +%H:%M:%S)] Phase 3 GRPO on grpo_pool_phase2 (FVEval format) ==="
PATH=${OSS_CAD_SUITE}/bin:$PATH \
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python3 -u \
    scripts/run_grpo_pilot.py \
    --policy results/sft_qwen_coder_7b_replay50/checkpoint_20260420_210826 \
    --output-dir results/grpo_pec_phase3_fveval \
    --reward-mode pec_multiref \
    --pool data/train/grpo/grpo_pool_phase2.jsonl \
    --prompt-format fveval \
    --num-generations 4 --batch-size 4 --max-steps 100 \
    --max-completion-length 256 --lr 1e-6 --beta 0.04 --lora-r 16 --seed 0 \
    > "$LOG/grpo_phase3_fveval.log" 2>&1

echo "=== [$(date +%H:%M:%S)] phase2_v2 + phase3 chain done ==="
