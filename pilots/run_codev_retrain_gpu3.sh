#!/usr/bin/env bash
set -euo pipefail

ROOT="${ANON_ROOT}"
MODEL="${STUDENT_MODEL}"
SFT_DIR="$ROOT/experiments/data/CodeV-SVA-datasets/sft"
GRPO_POOL="$ROOT/experiments/data/CodeV-SVA-datasets/grpo/codev_grpo_unified.jsonl"
RESULTS_DIR="$ROOT/experiments/results"
RUN_TAG="${RUN_TAG:-codev_retrain_$(date +%Y%m%d_%H%M%S)}"
SFT_OUT="$RESULTS_DIR/${RUN_TAG}_sft"
GRPO_OUT="$RESULTS_DIR/${RUN_TAG}_grpo"
VLLM_EVAL_GPU="${VLLM_EVAL_GPU:-}"

mkdir -p "$RESULTS_DIR"

echo "[run] tag=$RUN_TAG"
echo "[run] sft_out=$SFT_OUT"
echo "[run] grpo_out=$GRPO_OUT"
echo "[run] vllm_eval_gpu=${VLLM_EVAL_GPU:-<disabled>}"

SFT_EXTRA_ARGS=()
GRPO_EXTRA_ARGS=()
if [[ -n "$VLLM_EVAL_GPU" ]]; then
  SFT_EXTRA_ARGS+=(--vllm-eval-gpu "$VLLM_EVAL_GPU" --eval-prompt-format simple)
  GRPO_EXTRA_ARGS+=(--vllm-eval-gpu "$VLLM_EVAL_GPU" --eval-prompt-format fveval)
fi

CUDA_VISIBLE_DEVICES=3 python3 "$ROOT/experiments/scripts/run_curriculum_sft_v2.py" \
  --model "$MODEL" \
  --device cuda:0 \
  --dtype bf16 \
  --sft-dir "$SFT_DIR" \
  --output-dir "$SFT_OUT" \
  --batch-size 4 \
  --epochs-per-stage 3 \
  --lr 2e-5 \
  --alpha 3.0 \
  --max-len 512 \
  --seed 0 \
  --eval-each-stage \
  --eval-every-steps 500 \
  --patience 3 \
  --early-stop-min-delta 0.0 \
  --eval-max-new 256 \
  --gradient-checkpointing \
  --replay-ratio 0.2 \
  "${SFT_EXTRA_ARGS[@]}"

LATEST_CKPT="$(find "$SFT_OUT" -maxdepth 1 -type d -name 'checkpoint_*' | sort | tail -n 1)"
if [[ -z "$LATEST_CKPT" ]]; then
  echo "[error] no SFT checkpoint found in $SFT_OUT" >&2
  exit 1
fi
echo "[run] latest_sft_checkpoint=$LATEST_CKPT"

CUDA_VISIBLE_DEVICES=3 python3 "$ROOT/experiments/scripts/run_grpo_pilot.py" \
  --policy "$LATEST_CKPT" \
  --output-dir "$GRPO_OUT" \
  --device cuda:0 \
  --reward-mode pec_multiref \
  --pool "$GRPO_POOL" \
  --prompt-format fveval \
  --num-generations 4 \
  --batch-size 1 \
  --max-steps 200 \
  --eval-every-steps 50 \
  --patience 3 \
  --early-stop-min-delta 0.0 \
  --eval-max-new 256 \
  --lr 1e-6 \
  --beta 0.04 \
  --seed 0 \
  "${GRPO_EXTRA_ARGS[@]}"
