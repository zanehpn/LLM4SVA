"""run_sft_with_reasoning.py
Reasoning-augmented SFT on the CodeV-83K reasoning split. Each row's
target is the original assistant turn (`<think>...</think>` + ```sv```
block), so the trained model learns to emit reasoning before the SVA.

Loss is temporal-token-weighted cross-entropy (paper §4.2 / App. D):
TEMPORAL_OPS tokens (`##`, `|->`, `s_eventually`, ...) get α=3× weight,
all other response tokens stay at weight 1, prompt tokens are masked.
This matches the curriculum-SFT loss in run_curriculum_sft_v2.py so the
two seeds share the same operator-aware gradient signal.

Why no stratified curriculum here:
    The 83K reasoning split has only 17 L5 (liveness) rows total
    (4 SFT + 13 GRPO). A 5-stage TCL curriculum is meaningless on this
    distribution; we use a single flat pool with TT-CE instead.

Usage:
    python scripts/run_sft_with_reasoning.py \\
      --model ${STUDENT_MODEL} \\
      --sft-jsonl data/train/sft_with_reasoning/sft_train.jsonl \\
      --output-dir results/sft_reasoning_<TS> \\
      --max-len 6144 --batch-size 1 --epochs 1 \\
      --use-lora --lora-r 16 --lora-alpha 32
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          get_cosine_schedule_with_warmup)

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "eval"))

from run_funcatk_eval import (
    FVEVAL_SYSTEM_PROMPT,
    build_fveval_user_prompt as build_user,
)
from src.temporal_loss import TEMPORAL_OPS


# TT-CE loss matches the curriculum-SFT one in run_curriculum_sft_v2.py.
# Paper §4.2: L = (1/T) Σ w_t · CE_t, with w_t = α for temporal-operator
# tokens and 1 otherwise.
def temporal_weighted_loss(logits, labels, weights):
    sl = logits[:, :-1, :].contiguous()
    slab = labels[:, 1:].contiguous()
    sw = weights[:, 1:].contiguous()
    pt = F.cross_entropy(
        sl.reshape(-1, sl.size(-1)), slab.reshape(-1),
        reduction="none", ignore_index=-100,
    ).view(slab.shape)
    valid = (slab != -100).float()
    n_tokens = valid.sum().clamp(min=1.0)
    return (pt * sw * valid).sum() / n_tokens


def temporal_weights_for_labels(tok, label_ids, alpha=3.0):
    """Return a per-token weight tensor matching label_ids: α for any
    label token whose decoded string contains a TEMPORAL_OPS substring,
    1 elsewhere. -100 (prompt-mask) keeps weight 1; the loss already masks
    those positions via the cross-entropy ignore_index."""
    w = [1.0] * len(label_ids)
    for i, tid in enumerate(label_ids):
        if tid == -100:
            continue
        try:
            s = tok.decode([tid])
        except Exception:
            continue
        if any(op in s for op in TEMPORAL_OPS):
            w[i] = alpha
    return w


# ------------------- format ---------------------------------------------
def format_example(tok, nl: str, rtl_context: str, assistant_full: str,
                    max_len: int = 6144, rtl_cap: int = 6000) -> tuple:
    """Build (input_ids, labels) where target = assistant_full (already
    contains <think>...</think>...```sv``` block from CodeV teacher).
    Prompt-portion labels = -100 so the loss only sees the assistant turn."""
    rtl = (rtl_context or "")
    if rtl_cap and len(rtl) > rtl_cap:
        rtl = rtl[:rtl_cap]
    msgs = [
        {"role": "system", "content": FVEVAL_SYSTEM_PROMPT},
        {"role": "user", "content": build_user(nl, rtl)},
    ]
    prompt = tok.apply_chat_template(msgs, tokenize=False,
                                      add_generation_prompt=True)
    response = assistant_full.rstrip() + tok.eos_token

    pids = tok(prompt, add_special_tokens=False)["input_ids"]
    rids = tok(response, add_special_tokens=False)["input_ids"]
    ids = pids + rids
    lbl = [-100] * len(pids) + rids[:]
    if len(ids) > max_len:
        # Right-truncate the response (reasoning is long, the SVA tail
        # gets dropped if the budget is exceeded). Better than truncating
        # the prompt (which would drop the question).
        ids = ids[:max_len]; lbl = lbl[:max_len]
    return ids, lbl


def collate(batch, pad_id: int, weights_alpha: float = 3.0, tok=None):
    max_len = max(len(b[0]) for b in batch)
    ids_list, lbl_list, attn_list, w_list = [], [], [], []
    for ids, lbl in batch:
        pad = max_len - len(ids)
        ids_list.append(ids + [pad_id] * pad)
        lbl_list.append(lbl + [-100] * pad)
        attn_list.append([1] * len(ids) + [0] * pad)
        if tok is not None:
            w = temporal_weights_for_labels(tok, lbl, alpha=weights_alpha)
        else:
            w = [1.0] * len(lbl)
        w_list.append(w + [1.0] * pad)
    return {
        "input_ids": torch.tensor(ids_list, dtype=torch.long),
        "labels": torch.tensor(lbl_list, dtype=torch.long),
        "attention_mask": torch.tensor(attn_list, dtype=torch.long),
        "weights": torch.tensor(w_list, dtype=torch.float32),
    }


class ReasoningSFTDataset(Dataset):
    def __init__(self, tok, examples, max_len, rtl_cap):
        self.tok = tok; self.ex = examples
        self.max_len = max_len; self.rtl_cap = rtl_cap

    def __len__(self): return len(self.ex)

    def __getitem__(self, i):
        e = self.ex[i]
        return format_example(
            self.tok,
            e.get("nl") or "",
            e.get("rtl_context") or "",
            e.get("assistant_full") or "",
            max_len=self.max_len, rtl_cap=self.rtl_cap,
        )


# ------------------- main ----------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--sft-jsonl", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=6144)
    ap.add_argument("--rtl-cap", type=int, default=4500)
    ap.add_argument("--warmup-frac", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--use-lora", action="store_true", default=True)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--alpha", type=float, default=3.0,
                    help="TT-CE temporal-token weight (paper §4.2 default: 3.0)")
    ap.add_argument("--gradient-checkpointing", action="store_true",
                    default=True)
    ap.add_argument("--save-every-steps", type=int, default=2000)
    ap.add_argument("--log-every-steps", type=int, default=25)
    ap.add_argument("--limit", type=int, default=0,
                    help="cap pool size for smoke testing")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / "train.log"

    def log(*a, **kw):
        msg = " ".join(str(x) for x in a)
        print(msg, **kw, flush=True)
        with open(log_file, "a") as f:
            f.write(msg + "\n")

    log(f"[args] {vars(args)}")

    # 1. tokenizer & model
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    log(f"[model] loading {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, trust_remote_code=True,
        device_map={"": args.device})

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    if args.use_lora:
        from peft import LoraConfig, get_peft_model
        peft_cfg = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout, bias="none",
            target_modules=["q_proj","k_proj","v_proj","o_proj",
                            "gate_proj","up_proj","down_proj"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_cfg)
        model.print_trainable_parameters()

    # 2. data
    log(f"[data] loading {args.sft_jsonl}")
    rows = []
    with open(args.sft_jsonl) as f:
        for ln in f:
            ln = ln.strip()
            if ln: rows.append(json.loads(ln))
            if args.limit and len(rows) >= args.limit: break
    log(f"[data] {len(rows)} rows")

    ds = ReasoningSFTDataset(tok, rows, args.max_len, args.rtl_cap)
    dl = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b: collate(b, tok.pad_token_id,
                                     weights_alpha=args.alpha, tok=tok),
        num_workers=2, drop_last=False,
    )

    # 3. optim + sched
    total_steps = (len(ds) // args.batch_size) * args.epochs
    warmup_steps = int(total_steps * args.warmup_frac)
    optim = AdamW([p for p in model.parameters() if p.requires_grad],
                   lr=args.lr, weight_decay=0.01)
    sched = get_cosine_schedule_with_warmup(optim, warmup_steps, total_steps)
    log(f"[train] total_steps={total_steps}  warmup={warmup_steps}  lr={args.lr}")

    # 4. training loop
    model.train()
    t0 = time.time()
    step = 0
    for epoch in range(args.epochs):
        for batch in dl:
            batch = {k: v.to(args.device) for k, v in batch.items()}
            weights = batch.pop("weights")
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            loss = temporal_weighted_loss(
                out.logits.float(), batch["labels"], weights,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optim.step(); sched.step(); optim.zero_grad()
            step += 1

            if step % args.log_every_steps == 0:
                elapsed = time.time() - t0
                rate = step / max(elapsed, 1)
                eta = (total_steps - step) / max(rate, 1e-3)
                log(f"  step {step}/{total_steps}  epoch={epoch}  "
                    f"loss={loss.item():.4f}  lr={sched.get_last_lr()[0]:.2e}  "
                    f"{rate:.2f} step/s  eta={eta:.0f}s")

            if step % args.save_every_steps == 0:
                ck = out_dir / f"checkpoint-{step}"
                model.save_pretrained(str(ck))
                tok.save_pretrained(str(ck))
                log(f"  [ckpt] saved {ck}")

    # 5. final save
    final = out_dir / "final"
    model.save_pretrained(str(final))
    tok.save_pretrained(str(final))
    log(f"[done] saved {final}  total_time={time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
