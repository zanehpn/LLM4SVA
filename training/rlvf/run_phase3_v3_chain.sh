#!/bin/bash
# Corrected Phase 3 chain — uses grpo_pool_phase2_v3.jsonl which has
# disable-iff-wrapped canonical refs that match the FVEval-format model
# outputs and the NL2SVA-Human eval refs.
#
# Pipeline:
#   1. Eval already-trained Phase 2.5 (data point on simple-format GRPO)
#   2. Phase 3 v3 GRPO (FVEval format + v3 pool with disable-iff refs)
#   3. Eval Phase 3 v3 (both checkpoints)
set -eu
cd ${REPO_ROOT}
LOG=${REFINE_LOGS}/phase2

# Step 1 — eval Phase 2.5 (already trained, just need PEC scoring)
echo "=== [$(date +%H:%M:%S)] Eval Phase 2.5 ==="
SFT_BASE="results/sft_qwen_coder_7b_replay50/checkpoint_20260420_210826"
TS=$(date +%Y%m%d_%H%M%S)
for ck in checkpoint-50 checkpoint-100; do
    [ -d "results/grpo_pec_phase2_v2/$ck" ] || continue
    tag="SFT_replay50+grpo_phase2_v2_${ck#checkpoint-}"
    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python3 scripts/run_eval_nl2sva_human.py \
        --model "$SFT_BASE" --adapter "results/grpo_pec_phase2_v2/$ck" \
        --tag "$tag" --max-new-tokens 4096 \
        > "$LOG/eval_${tag}_${TS}.log" 2>&1
    eval_json=$(ls -t results/eval_nl2sva_human_*${tag}*.json 2>/dev/null | head -1)
    [ -z "$eval_json" ] && { echo "[err] no eval JSON for $tag"; continue; }
    PATH=${OSS_CAD_SUITE}/bin:$PATH \
    PYTHONPATH=. python3 scripts/run_eval_with_pec.py \
        --eval-result "$eval_json" --workers 16 --depth 15 --timeout 30 \
        > "$LOG/pec_${tag}_${TS}.log" 2>&1
    echo "  -> $tag: $(grep -E 'Func@1 \(full' $LOG/pec_${tag}_${TS}.log | head -1)"
done

# Step 2 — Phase 3 v3 GRPO
echo "=== [$(date +%H:%M:%S)] Phase 3 v3 GRPO (FVEval + v3 pool with disable iff) ==="
PATH=${OSS_CAD_SUITE}/bin:$PATH \
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python3 -u \
    scripts/run_grpo_pilot.py \
    --policy "$SFT_BASE" \
    --output-dir results/grpo_pec_phase3_v3 \
    --reward-mode pec_multiref \
    --pool data/train/grpo/grpo_pool_phase2_v3.jsonl \
    --prompt-format fveval \
    --num-generations 4 --batch-size 4 --max-steps 100 \
    --max-completion-length 256 --lr 1e-6 --beta 0.04 --lora-r 16 --seed 0 \
    > "$LOG/grpo_phase3_v3.log" 2>&1

# Step 3 — eval Phase 3 v3
echo "=== [$(date +%H:%M:%S)] Eval Phase 3 v3 ==="
TS2=$(date +%Y%m%d_%H%M%S)
for ck in checkpoint-50 checkpoint-100; do
    [ -d "results/grpo_pec_phase3_v3/$ck" ] || continue
    tag="SFT_replay50+grpo_phase3_v3_${ck#checkpoint-}"
    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python3 scripts/run_eval_nl2sva_human.py \
        --model "$SFT_BASE" --adapter "results/grpo_pec_phase3_v3/$ck" \
        --tag "$tag" --max-new-tokens 4096 \
        > "$LOG/eval_${tag}_${TS2}.log" 2>&1
    eval_json=$(ls -t results/eval_nl2sva_human_*${tag}*.json 2>/dev/null | head -1)
    [ -z "$eval_json" ] && { echo "[err] no eval JSON for $tag"; continue; }
    PATH=${OSS_CAD_SUITE}/bin:$PATH \
    PYTHONPATH=. python3 scripts/run_eval_with_pec.py \
        --eval-result "$eval_json" --workers 16 --depth 15 --timeout 30 \
        > "$LOG/pec_${tag}_${TS2}.log" 2>&1
    echo "  -> $tag: $(grep -E 'Func@1 \(full' $LOG/pec_${tag}_${TS2}.log | head -1)"
done

echo "=== [$(date +%H:%M:%S)] phase3_v3 chain done ==="
