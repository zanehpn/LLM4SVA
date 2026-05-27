#!/bin/bash
# RTL-aware GRPO pilot (Pilot 8).
# Chain: NL-filled pool → disable-iff rewrite → alt-gen → PEC verify → train → eval
set -eu
cd ${REPO_ROOT}
LOG=${REFINE_LOGS}/phase2
SFT_BASE="results/sft_qwen_coder_7b_replay50/checkpoint_20260420_210826"

# ---------------------------------------------------------------------
# Step 1 — disable-iff wrap (point add_disable_iff_refs at our new file)
# ---------------------------------------------------------------------
echo "=== [$(date +%H:%M:%S)] Step 1: disable-iff wrap ==="
PYTHONPATH=. python3 - <<'EOF'
import json, re
from pathlib import Path
import sys
sys.path.insert(0, 'scripts')
from add_disable_iff_refs import add_disable_iff

IN = Path('data/train/grpo/industrial_with_rtl_nl_filled.jsonl')
OUT = Path('data/train/grpo/industrial_with_rtl_disable_iff.jsonl')
recs = [json.loads(l) for l in open(IN)]
n_wrap = n_has = 0
for r in recs:
    raw = r['reference_sva']
    di = add_disable_iff(raw)
    if di == raw:
        if 'disable iff' in raw.lower():
            n_has += 1
            r['ref_svas'] = [raw]
        else:
            r['ref_svas'] = [raw]
    else:
        n_wrap += 1
        r['reference_sva'] = di
        r['ref_svas'] = [di, raw]
    r['n_extra_refs'] = len(r['ref_svas']) - 1

with open(OUT, 'w') as f:
    for r in recs: f.write(json.dumps(r) + '\n')
print(f'wrapped: {n_wrap}  already had: {n_has}  total: {len(recs)}')
print(f'write: {OUT}')
EOF

# ---------------------------------------------------------------------
# Step 2 — alt-gen + PEC verify (adapted from build_industrial_grpo_pool)
# ---------------------------------------------------------------------
echo "=== [$(date +%H:%M:%S)] Step 2: alt-gen + PEC (multi-ref) ==="

PATH=${OSS_CAD_SUITE}/bin:$PATH \
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python3 scripts/build_rtl_aware_multiref.py \
    > "$LOG/rtl_aware_multiref.log" 2>&1

# ---------------------------------------------------------------------
# Step 3 — Train GRPO Pilot 8 (FVEval format, multi-ref reward)
# ---------------------------------------------------------------------
echo "=== [$(date +%H:%M:%S)] Step 3: GRPO Pilot 8 training ==="
PATH=${OSS_CAD_SUITE}/bin:$PATH \
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python3 -u \
    scripts/run_grpo_pilot.py \
    --policy "$SFT_BASE" \
    --output-dir results/grpo_pec_pilot8_rtl \
    --reward-mode pec_multiref \
    --pool data/train/grpo/grpo_pool_pilot8_rtl.jsonl \
    --prompt-format fveval \
    --num-generations 4 --batch-size 4 --max-steps 100 \
    --max-completion-length 256 --lr 1e-6 --beta 0.04 --lora-r 16 --seed 0 \
    > "$LOG/grpo_pilot8.log" 2>&1

# ---------------------------------------------------------------------
# Step 4 — Eval Pilot 8 checkpoints
# ---------------------------------------------------------------------
echo "=== [$(date +%H:%M:%S)] Step 4: eval Pilot 8 ==="
TS=$(date +%Y%m%d_%H%M%S)
for ck in checkpoint-50 checkpoint-100; do
    [ -d "results/grpo_pec_pilot8_rtl/$ck" ] || continue
    tag="SFT_replay50+grpo_pilot8_rtl_${ck#checkpoint-}"
    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python3 scripts/run_eval_nl2sva_human.py \
        --model "$SFT_BASE" --adapter "results/grpo_pec_pilot8_rtl/$ck" --tag "$tag" \
        --max-new-tokens 4096 > "$LOG/eval_${tag}_${TS}.log" 2>&1
    eval_json=$(ls -t results/eval_nl2sva_human_*${tag}*.json 2>/dev/null | head -1)
    [ -z "$eval_json" ] && continue
    PATH=${OSS_CAD_SUITE}/bin:$PATH PYTHONPATH=. python3 scripts/run_eval_with_pec.py \
        --eval-result "$eval_json" --workers 16 --depth 15 --timeout 30 \
        > "$LOG/pec_${tag}_${TS}.log" 2>&1
    echo "  -> $tag: $(grep -E 'Func@1 \(full' $LOG/pec_${tag}_${TS}.log | head -1)"
done
echo "=== [$(date +%H:%M:%S)] Pilot 8 chain done ==="
