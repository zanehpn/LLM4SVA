#!/bin/bash
# IPO retry: rollouts already exist from previous run; just rerun P (PEC,
# now with PATH set) + X (pairs), then train, then eval.
set -eu
cd ${REPO_ROOT}
LOG=${REFINE_LOGS}/phase2
SFT_BASE="results/sft_qwen_coder_7b_replay50/checkpoint_20260420_210826"

echo "=== [$(date +%H:%M:%S)] Stage P: PEC-verify rollouts ==="
PATH=${OSS_CAD_SUITE}/bin:$PATH PYTHONPATH=. python3 -u \
    scripts/build_ipo_dataset.py --stage P --workers 32 \
    > "$LOG/ipo_pec.log" 2>&1

echo "=== [$(date +%H:%M:%S)] Stage X: build preference pairs ==="
PYTHONPATH=. python3 -u \
    scripts/build_ipo_dataset.py --stage X \
    --max-pos-per-prompt 2 --max-neg-per-prompt 2 --max-pairs-total 20000 \
    > "$LOG/ipo_pairs.log" 2>&1

# Check pairs file non-empty before training
N_PAIRS=$(wc -l < data/train/ipo/ipo_pairs.jsonl)
echo "=== pairs count: $N_PAIRS ==="
if [ "$N_PAIRS" -lt 100 ]; then
    echo "[error] too few pairs ($N_PAIRS); aborting"
    exit 2
fi

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

echo "=== [$(date +%H:%M:%S)] Step 3: eval IPO ==="
TS=$(date +%Y%m%d_%H%M%S)
for ck in $(ls -d results/ipo_pilot/checkpoint-* results/ipo_pilot/final 2>/dev/null); do
    base_ck=$(basename "$ck")
    [ "$base_ck" = "final" ] && tag="SFT_replay50+ipo_final" || tag="SFT_replay50+ipo_${base_ck#checkpoint-}"
    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python3 scripts/run_eval_nl2sva_human.py \
        --model "$SFT_BASE" --adapter "$ck" --tag "$tag" \
        --max-new-tokens 4096 \
        > "$LOG/eval_${tag}_${TS}.log" 2>&1
    eval_json=$(ls -t results/eval_nl2sva_human_*${tag}*.json 2>/dev/null | head -1)
    [ -z "$eval_json" ] && continue
    PATH=${OSS_CAD_SUITE}/bin:$PATH PYTHONPATH=. python3 \
        scripts/run_eval_with_pec.py --eval-result "$eval_json" \
        --workers 16 --depth 15 --timeout 30 \
        > "$LOG/pec_${tag}_${TS}.log" 2>&1
    echo "  -> $tag: $(grep -E 'Func@1 \(full' $LOG/pec_${tag}_${TS}.log | head -1)"
done
echo "=== [$(date +%H:%M:%S)] IPO retry chain done ==="
