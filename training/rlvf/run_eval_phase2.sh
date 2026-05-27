#!/bin/bash
# Eval all GRPO Phase 2 checkpoints with PEC scoring.
# Run from project root with PATH=oss-cad-suite-bin:$PATH so PEC works.
set -u
cd ${REPO_ROOT}
SFT_BASE="results/sft_qwen_coder_7b_replay50/checkpoint_20260420_210826"
PHASE2_OUT="results/grpo_pec_phase2"
LOG_DIR="${REFINE_LOGS}/phase2"
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)

CKPTS=("checkpoint-50" "checkpoint-100")

for ckpt in "${CKPTS[@]}"; do
    apath="$PHASE2_OUT/$ckpt"
    [ -d "$apath" ] || { echo "[skip] $apath not found"; continue; }
    tag="SFT_replay50+grpo_phase2_${ckpt#checkpoint-}"
    eval_log="$LOG_DIR/eval_${tag}_${TS}.log"
    pec_log="$LOG_DIR/pec_${tag}_${TS}.log"
    echo "============================================================"
    echo "[$(date +%H:%M:%S)] === eval $tag ==="
    echo "============================================================"
    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python3 scripts/run_eval_nl2sva_human.py \
        --model "$SFT_BASE" \
        --adapter "$apath" \
        --tag "$tag" \
        --max-new-tokens 4096 \
        2>&1 | tee "$eval_log"
    # find the just-written eval JSON
    eval_json=$(ls -t results/eval_nl2sva_human_*${tag}*.json 2>/dev/null | head -1)
    [ -z "$eval_json" ] && { echo "[err] no eval JSON for $tag"; continue; }
    echo "[$(date +%H:%M:%S)] === PEC score: $eval_json ==="
    PYTHONPATH=. python3 scripts/run_eval_with_pec.py \
        --eval-result "$eval_json" \
        --workers 16 --depth 15 --timeout 30 \
        2>&1 | tee "$pec_log"
done
echo
echo "ALL_PHASE2_EVALS_DONE @ $(date +%H:%M:%S)"
