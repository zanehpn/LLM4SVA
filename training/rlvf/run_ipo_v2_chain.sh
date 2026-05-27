#!/bin/bash
# IPO v2 retry with conservative hyperparameters.
# Diagnosis from v1 (10.4% — reference collapse):
#   v1 used lr=5e-6, β=0.1, 2 epochs → rewards/chosen fell to -0.08 while
#   rewards/rejected fell to -0.23. Both pushed down from the SFT base;
#   the policy learned "produce nothing" rather than "prefer chosen over
#   rejected". This is classic IPO/DPO reference collapse at too-high LR.
# v2 fixes (standard DPO/IPO best practices):
#   lr 5e-6 → 5e-7  (10× smaller — keep margin growth gentle)
#   β  0.1 → 0.5    (target margin 1/(2β) shrinks 5 → 1 nat — tighter trust region)
#   epochs 2 → 1    (avoid overfitting on 2204 pairs)
#   LoRA r 16 → 8   (smaller adapter — less capacity to deviate from SFT)
# Reuses existing data/train/ipo/ipo_pairs.jsonl (2204 pairs).
set -eu
cd ${REPO_ROOT}
LOG=${REFINE_LOGS}/phase2
SFT_BASE="results/sft_qwen_coder_7b_replay50/checkpoint_20260420_210826"

N_PAIRS=$(wc -l < data/train/ipo/ipo_pairs.jsonl)
echo "=== [$(date +%H:%M:%S)] IPO v2 on $N_PAIRS existing pairs ==="

# Step 1 — IPO v2 training
PATH=${OSS_CAD_SUITE}/bin:$PATH \
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python3 -u \
    scripts/run_ipo_pilot.py \
    --policy "$SFT_BASE" \
    --output-dir results/ipo_pilot_v2 \
    --beta 0.5 --epochs 1 --lr 5e-7 \
    --batch-size 2 --grad-accum 4 \
    --lora-r 8 --save-steps 50 --seed 0 \
    --loss-type ipo \
    > "$LOG/ipo_train_v2.log" 2>&1

# Step 2 — eval IPO v2 checkpoints
echo "=== [$(date +%H:%M:%S)] Step 2: eval IPO v2 ==="
TS=$(date +%Y%m%d_%H%M%S)
for ck in $(ls -d results/ipo_pilot_v2/checkpoint-* results/ipo_pilot_v2/final 2>/dev/null); do
    base_ck=$(basename "$ck")
    [ "$base_ck" = "final" ] && tag="SFT_replay50+ipo_v2_final" || tag="SFT_replay50+ipo_v2_${base_ck#checkpoint-}"
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
echo "=== [$(date +%H:%M:%S)] IPO v2 chain done ==="
