#!/bin/bash
# IPO pipeline: build preference dataset → train → eval
set -eu
cd ${REPO_ROOT}
LOG=${REFINE_LOGS}/phase2
SFT_BASE="results/sft_qwen_coder_7b_replay50/checkpoint_20260420_210826"

# Step 1 — build IPO preference dataset (vLLM rollouts + PEC labels + pairs)
echo "=== [$(date +%H:%M:%S)] Step 1: build IPO preference dataset ==="
PATH=${OSS_CAD_SUITE}/bin:$PATH \
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python3 -u \
    scripts/build_ipo_dataset.py --stage all \
    --k-rollouts 8 --temperature 1.0 --max-tokens 256 \
    --gpu-memory-utilization 0.7 \
    --workers 32 \
    --max-pos-per-prompt 2 --max-neg-per-prompt 2 \
    --max-pairs-total 20000 \
    > "$LOG/ipo_build.log" 2>&1

# Step 2 — IPO training
echo "=== [$(date +%H:%M:%S)] Step 2: IPO training ==="
PATH=${OSS_CAD_SUITE}/bin:$PATH \
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python3 -u \
    scripts/run_ipo_pilot.py \
    --policy "$SFT_BASE" \
    --output-dir results/ipo_pilot \
    --beta 0.1 --epochs 2 --lr 5e-6 \
    --batch-size 2 --grad-accum 4 \
    --lora-r 16 --save-steps 50 --seed 0 \
    --loss-type ipo \
    > "$LOG/ipo_train.log" 2>&1

# Step 3 — eval IPO checkpoints
echo "=== [$(date +%H:%M:%S)] Step 3: eval IPO ==="
TS=$(date +%Y%m%d_%H%M%S)
# Find all saved checkpoints (including final)
for ck in $(ls -d results/ipo_pilot/checkpoint-* results/ipo_pilot/final 2>/dev/null); do
    base_ck=$(basename "$ck")
    [ "$base_ck" = "final" ] && tag="SFT_replay50+ipo_final" || tag="SFT_replay50+ipo_${base_ck#checkpoint-}"
    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python3 scripts/run_eval_nl2sva_human.py \
        --model "$SFT_BASE" --adapter "$ck" --tag "$tag" \
        --max-new-tokens 4096 \
        > "$LOG/eval_${tag}_${TS}.log" 2>&1
    eval_json=$(ls -t results/eval_nl2sva_human_*${tag}*.json 2>/dev/null | head -1)
    [ -z "$eval_json" ] && { echo "[err] no eval JSON for $tag"; continue; }
    PATH=${OSS_CAD_SUITE}/bin:$PATH PYTHONPATH=. python3 \
        scripts/run_eval_with_pec.py --eval-result "$eval_json" \
        --workers 16 --depth 15 --timeout 30 \
        > "$LOG/pec_${tag}_${TS}.log" 2>&1
    echo "  -> $tag: $(grep -E 'Func@1 \(full' $LOG/pec_${tag}_${TS}.log | head -1)"
done
echo "=== [$(date +%H:%M:%S)] IPO chain done ==="
