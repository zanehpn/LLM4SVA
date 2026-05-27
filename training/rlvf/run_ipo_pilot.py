#!/usr/bin/env python3
"""
run_ipo_pilot.py — IPO training (TRL DPOTrainer with loss_type='ipo') on
the preference dataset built by build_ipo_dataset.py.

IPO vs DPO vs GRPO for this task:
  - DPO's Bradley-Terry σ link assumes noisy human preferences; PEC is a
    deterministic oracle, so the σ is actively harmful.
  - GRPO's KL penalty failed to activate (KL≈0) at our scale; policy
    overfit inside a trusted-looking neighborhood.
  - IPO's squared loss on a bounded margin (1/(2β)) is a per-preference-
    pair implicit trust region that regulates behavioral change, not
    distribution distance. For tied positives (multi-ref pool: 12.4% of
    prompts) the target margin collapses to 0, preserving SFT on
    ambiguous cases.

Usage:
  CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python3 scripts/run_ipo_pilot.py \\
      --policy results/sft_qwen_coder_7b_replay50/checkpoint_20260420_210826 \\
      --output-dir results/ipo_pilot \\
      --beta 0.1 --epochs 2 --lr 5e-6
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXPERIMENTS_DIR))

F_PAIRS = EXPERIMENTS_DIR / "data" / "train" / "ipo" / "ipo_pairs.jsonl"


def load_dataset(pairs_path: Path, max_pairs: int | None = None):
    from datasets import Dataset
    rows = []
    with open(pairs_path) as f:
        for line in f:
            r = json.loads(line)
            rows.append({
                "prompt": r["prompt"],
                "chosen": r["chosen"],
                "rejected": r["rejected"],
            })
    if max_pairs:
        rows = rows[:max_pairs]
    return Dataset.from_list(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--pairs", default=str(F_PAIRS))
    ap.add_argument("--beta", type=float, default=0.1,
                    help="IPO β — controls implicit-trust-region width "
                         "(margin cap = 1/(2β))")
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--max-prompt-length", type=int, default=768)
    ap.add_argument("--save-steps", type=int, default=50)
    ap.add_argument("--max-pairs", type=int, default=0)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--loss-type", default="ipo",
                    choices=["ipo", "sigmoid", "hinge", "kto_pair"],
                    help="ipo = squared loss, bounded margin (this task); "
                         "sigmoid = DPO; kto_pair = KTO on paired data")
    args = ap.parse_args()

    from trl import DPOTrainer, DPOConfig
    from peft import LoraConfig
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    print(f"[loss] {args.loss_type}, β={args.beta}")
    print(f"[policy] {args.policy}")
    print(f"[pairs] {args.pairs}")

    dataset = load_dataset(Path(args.pairs),
                           args.max_pairs or None)
    print(f"[data] {len(dataset)} preference pairs")

    # Load tokenizer and model explicitly so we can pass them into
    # DPOTrainer (some TRL versions expect this pattern for LoRA)
    tok = AutoTokenizer.from_pretrained(args.policy, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.policy, torch_dtype=torch.bfloat16,
        trust_remote_code=True, low_cpu_mem_usage=True,
    )
    # Keep reference as same base (TRL will handle the LoRA split)

    peft_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=2 * args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    print(f"[lora] r={args.lora_r}, α={2*args.lora_r}")

    cfg = DPOConfig(
        output_dir=str(out),
        loss_type=args.loss_type,
        beta=args.beta,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        logging_steps=5,
        save_steps=args.save_steps,
        save_total_limit=4,
        bf16=True,
        gradient_checkpointing=True,
        report_to=[],
        seed=args.seed,
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model,
        args=cfg,
        train_dataset=dataset,
        processing_class=tok,
        peft_config=peft_cfg,
    )

    t0 = time.time()
    trainer.train()
    print(f"[ipo] training done in {time.time()-t0:.1f}s")
    trainer.save_model(str(out / "final"))
    print(f"[ipo] saved to {out / 'final'}")


if __name__ == "__main__":
    main()
