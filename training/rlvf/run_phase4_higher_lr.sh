#!/bin/bash
# Phase 4 fallback: if Phase 3 v3 still hits the SFT ceiling (≤ 30%),
# the issue may be that LR=1e-6 / beta=0.04 are too conservative — the
# policy can't escape the SFT manifold. Try LR=5e-6 + beta=0.01 with the
# same v3 pool and FVEval format.
set -eu
cd ${REPO_ROOT}
LOG=${REFINE_LOGS}/phase2
SFT_BASE="results/sft_qwen_coder_7b_replay50/checkpoint_20260420_210826"

echo "=== [$(date +%H:%M:%S)] Phase 4 GRPO: higher LR + lower KL ==="
PATH=${OSS_CAD_SUITE}/bin:$PATH \
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python3 -u \
    scripts/run_grpo_pilot.py \
    --policy "$SFT_BASE" \
    --output-dir results/grpo_pec_phase4_hilr \
    --reward-mode pec_multiref \
    --pool data/train/grpo/grpo_pool_phase2_v3.jsonl \
    --prompt-format fveval \
    --num-generations 4 --batch-size 4 --max-steps 100 \
    --max-completion-length 256 \
    --lr 5e-6 --beta 0.01 \
    --lora-r 32 \
    --seed 0 \
    > "$LOG/grpo_phase4_hilr.log" 2>&1

# Eval both checkpoints
echo "=== [$(date +%H:%M:%S)] Eval Phase 4 ==="
TS=$(date +%Y%m%d_%H%M%S)
for ck in checkpoint-50 checkpoint-100; do
    [ -d "results/grpo_pec_phase4_hilr/$ck" ] || continue
    tag="SFT_replay50+grpo_phase4_hilr_${ck#checkpoint-}"
    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python3 scripts/run_eval_nl2sva_human.py \
        --model "$SFT_BASE" --adapter "results/grpo_pec_phase4_hilr/$ck" \
        --tag "$tag" --max-new-tokens 4096 \
        > "$LOG/eval_${tag}_${TS}.log" 2>&1
    eval_json=$(ls -t results/eval_nl2sva_human_*${tag}*.json 2>/dev/null | head -1)
    [ -z "$eval_json" ] && continue
    PATH=${OSS_CAD_SUITE}/bin:$PATH PYTHONPATH=. python3 scripts/run_eval_with_pec.py \
        --eval-result "$eval_json" --workers 16 --depth 15 --timeout 30 \
        > "$LOG/pec_${tag}_${TS}.log" 2>&1
    echo "  -> $tag: $(grep -E 'Func@1 \(full' $LOG/pec_${tag}_${TS}.log | head -1)"
done
echo "=== [$(date +%H:%M:%S)] phase4 done ==="
