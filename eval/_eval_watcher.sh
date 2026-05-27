#!/bin/bash
# Watch the SFT-reasoning output dir for new checkpoints; when one
# appears, fire a Func@1 eval on Human + Machine and append the result
# to a summary file. Skip checkpoints already evaluated.
#
# Usage:
#   bash _eval_watcher.sh <output_dir> <gpu_id>
set -u
OUT_DIR="${1:-results/sft_reasoning_20260429_110010}"
GPU="${2:-0}"
SUMMARY="$OUT_DIR/_eval_summary.tsv"

mkdir -p "$OUT_DIR"
[ -f "$SUMMARY" ] || echo -e "ckpt\thuman_func1\thuman_relax1\tmachine_func1\tmachine_relax1\twallclock_s" > "$SUMMARY"

export PATH=${OSS_CAD_SUITE}/bin:$PATH
cd ${REPO_ROOT}

while true; do
  for CK in $(ls -d "$OUT_DIR"/checkpoint-* 2>/dev/null | sort -V); do
    STEP=$(basename "$CK" | sed 's/checkpoint-//')
    HUMAN_OUT="$OUT_DIR/_eval_ckpt${STEP}_human.json"
    [ -f "$HUMAN_OUT" ] && continue   # already done

    echo "[watcher] running eval on ckpt-$STEP at $(date +%H:%M:%S)"
    T0=$(date +%s)

    CUDA_VISIBLE_DEVICES=$GPU \
    python -u scripts/run_funcatk_eval.py \
      --model ${STUDENT_MODEL} \
      --adapter "$CK" \
      --tasks data/test/nl2sva_human.jsonl \
      --prompt-format fveval \
      --num-samples 1 --ks 1,16 \
      --temperature 0.0 --max-new-tokens 4096 --max-model-len 16384 \
      --gpu-memory-utilization 0.30 --tensor-parallel-size 1 \
      --workers 8 --depth 15 --timeout 30 \
      --pec-reset-mode auto --liveness-bound 15 \
      --skip-coverage-check --skip-greedy-diagnostic \
      --tag "ckpt${STEP}_human" \
      --output "$HUMAN_OUT" \
      > "$OUT_DIR/_eval_ckpt${STEP}_human.log" 2>&1

    MACH_OUT="$OUT_DIR/_eval_ckpt${STEP}_machine.json"
    CUDA_VISIBLE_DEVICES=$GPU \
    python -u scripts/run_funcatk_eval.py \
      --model ${STUDENT_MODEL} \
      --adapter "$CK" \
      --tasks data/test/nl2sva_machine.jsonl \
      --prompt-format fveval \
      --num-samples 1 --ks 1,16 \
      --temperature 0.0 --max-new-tokens 4096 --max-model-len 16384 \
      --gpu-memory-utilization 0.30 --tensor-parallel-size 1 \
      --workers 8 --depth 15 --timeout 30 \
      --pec-reset-mode auto --liveness-bound 15 \
      --skip-coverage-check --skip-greedy-diagnostic \
      --tag "ckpt${STEP}_machine" \
      --output "$MACH_OUT" \
      > "$OUT_DIR/_eval_ckpt${STEP}_machine.log" 2>&1

    T1=$(date +%s); DT=$((T1 - T0))

    HF1=$(python3 -c "import json; r=json.load(open('$HUMAN_OUT')); print(r['overall']['func@1'])" 2>/dev/null)
    HR1=$(python3 -c "import json; r=json.load(open('$HUMAN_OUT')); print(r['overall']['func_relaxed@1'])" 2>/dev/null)
    MF1=$(python3 -c "import json; r=json.load(open('$MACH_OUT')); print(r['overall']['func@1'])" 2>/dev/null)
    MR1=$(python3 -c "import json; r=json.load(open('$MACH_OUT')); print(r['overall']['func_relaxed@1'])" 2>/dev/null)
    echo -e "ckpt-${STEP}\t${HF1}\t${HR1}\t${MF1}\t${MR1}\t${DT}" >> "$SUMMARY"
    echo "[watcher] ckpt-${STEP}  human=${HF1}/${HR1}  machine=${MF1}/${MR1}  ${DT}s"
  done
  sleep 60
done
