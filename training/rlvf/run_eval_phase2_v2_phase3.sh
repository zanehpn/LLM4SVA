#!/bin/bash
# Eval Phase 2.5 (grpo_pec_phase2_v2) and Phase 3 (grpo_pec_phase3_fveval)
# checkpoints with PEC scoring. Run with PATH including oss-cad-suite/bin.
set -u
cd ${REPO_ROOT}
SFT_BASE="results/sft_qwen_coder_7b_replay50/checkpoint_20260420_210826"
LOG_DIR=${REFINE_LOGS}/phase2
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)

declare -A RUNS=(
    ["SFT_replay50+grpo_phase2_v2_50"]="results/grpo_pec_phase2_v2/checkpoint-50"
    ["SFT_replay50+grpo_phase2_v2_100"]="results/grpo_pec_phase2_v2/checkpoint-100"
    ["SFT_replay50+grpo_phase3_fveval_50"]="results/grpo_pec_phase3_fveval/checkpoint-50"
    ["SFT_replay50+grpo_phase3_fveval_100"]="results/grpo_pec_phase3_fveval/checkpoint-100"
    ["SFT_replay50+grpo_phase3_v2_fveval_50"]="results/grpo_pec_phase3_v2_fveval/checkpoint-50"
    ["SFT_replay50+grpo_phase3_v2_fveval_100"]="results/grpo_pec_phase3_v2_fveval/checkpoint-100"
)

for tag in "${!RUNS[@]}"; do
    apath="${RUNS[$tag]}"
    [ -d "$apath" ] || { echo "[skip] $apath not found"; continue; }
    eval_log="$LOG_DIR/eval_${tag}_${TS}.log"
    pec_log="$LOG_DIR/pec_${tag}_${TS}.log"
    echo "============================================================"
    echo "[$(date +%H:%M:%S)] === eval $tag ==="
    echo "============================================================"
    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python3 scripts/run_eval_nl2sva_human.py \
        --model "$SFT_BASE" --adapter "$apath" --tag "$tag" \
        --max-new-tokens 4096 \
        2>&1 | tee "$eval_log"
    eval_json=$(ls -t results/eval_nl2sva_human_*${tag}*.json 2>/dev/null | head -1)
    [ -z "$eval_json" ] && { echo "[err] no eval JSON for $tag"; continue; }
    echo "[$(date +%H:%M:%S)] === PEC score: $eval_json ==="
    PATH=${OSS_CAD_SUITE}/bin:$PATH PYTHONPATH=. python3 scripts/run_eval_with_pec.py \
        --eval-result "$eval_json" --workers 16 --depth 15 --timeout 30 \
        2>&1 | tee "$pec_log"
done
echo
echo "PHASE2_V2_PHASE3_EVALS_DONE @ $(date +%H:%M:%S)"
