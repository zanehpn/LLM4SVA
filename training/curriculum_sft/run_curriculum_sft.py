#!/usr/bin/env python3
"""
run_curriculum_sft.py — small-scale curriculum fine-tuning demo.

Scale-limited (150 train examples, 0.5 B model) but end-to-end proof of the
paper's Component 2 pipeline:
  1. Baseline eval of Qwen2.5-0.5B-Instruct on 30 NL2SVA tasks
  2. 5-stage curriculum SFT (L1 → L5) with 20% replay
  3. Temporal-token-weighted CE loss (alpha = 3.0)
  4. Post-train eval on the same 30 tasks
  5. Before/after comparison

Output:
  results/curriculum_sft_<ts>.json  — metrics + per-example predictions
  results/curriculum_sft_<ts>.log   — training loss per step

Not a serious training run — it's a demonstration that the pipeline is
correct. The paper spec needs ~10 k examples + 4×A100 for the full curriculum.
"""

import argparse
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tcl import classify_tcl
from src.mock_verifier import syntax_check
from src.temporal_loss import TEMPORAL_OPS

EXPERIMENTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(EXPERIMENTS_DIR, "data")
RESULTS_DIR = os.path.join(EXPERIMENTS_DIR, "results")

SYSTEM_PROMPT = (
    "You are an expert in SystemVerilog Assertions (SVA). "
    "Generate ONE syntactically correct SVA assertion for the given natural "
    "language specification. Output ONLY the SVA code — no explanation, no "
    "markdown fences. Use the `assert property (@(posedge clk) ...);` form. "
    "Match the temporal complexity implied by the specification: use bare "
    "`##N` for simple fixed delays, `##[a:b]` for ranged delays, `|->`/`|=>` "
    "only when the spec genuinely has an antecedent-consequent structure, and "
    "`s_eventually`/`s_until` for liveness."
)


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def build_training_set():
    """Combine sample_svas.json (NL-SVA) + sva_examples.json (desc-SVA) → flat list."""
    with open(os.path.join(DATA_DIR, "sample_svas.json")) as f:
        sample = json.load(f)["assertions"]
    with open(os.path.join(DATA_DIR, "sva_examples.json")) as f:
        more = json.load(f)

    train = []
    for a in sample:
        train.append({
            "nl": a["nl"],
            "sva": a["sva"],
            "tcl": a["tcl"],
            "source": "sample_svas",
        })
    for a in more:
        # sva_examples has a "description" field but not full NL — use the
        # description text (stripped of the "L1: " prefix) as a proxy NL.
        desc = a.get("description", "").strip()
        desc = re.sub(r"^L\d+(\+L\d+)?:\s*", "", desc)
        if not desc:
            continue
        train.append({
            "nl": desc,
            "sva": a["sva"],
            "tcl": a["expected_level"],
            "source": "sva_examples",
        })
    return train


def build_eval_set():
    path = os.path.join(DATA_DIR, "nl2sva_tasks_expanded.json")
    with open(path) as f:
        data = json.load(f)
    return data["tasks"]


# --------------------------------------------------------------------------- #
# Tokenization + temporal weight
# --------------------------------------------------------------------------- #
def format_example(tok, system: str, nl: str, sva: str, max_len: int = 256):
    """
    Produce input_ids, labels (with -100 masking on prompt), and temporal
    weight vector (alpha where decoded token contains a temporal op).

    The assistant turn is `sva` terminated with the tokenizer's EOS.
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Generate an SVA assertion for: {nl}"},
    ]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    response = sva + tok.eos_token

    prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    resp_ids = tok(response, add_special_tokens=False)["input_ids"]

    input_ids = prompt_ids + resp_ids
    labels = [-100] * len(prompt_ids) + resp_ids[:]

    if len(input_ids) > max_len:
        input_ids = input_ids[:max_len]
        labels = labels[:max_len]
    return input_ids, labels


def build_temporal_weight(tok, labels, alpha: float = 3.0):
    """Return weight vector: alpha where decoded token contains a temporal op."""
    weights = [1.0] * len(labels)
    for i, tid in enumerate(labels):
        if tid == -100:
            weights[i] = 0.0
            continue
        s = tok.decode([tid])
        if any(op in s for op in TEMPORAL_OPS):
            weights[i] = alpha
    return weights


class SVADataset(Dataset):
    def __init__(self, tok, examples, max_len=256, alpha=3.0):
        self.tok = tok
        self.examples = examples
        self.max_len = max_len
        self.alpha = alpha

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        ids, labels = format_example(self.tok, SYSTEM_PROMPT, ex["nl"], ex["sva"], self.max_len)
        weights = build_temporal_weight(self.tok, labels, self.alpha)
        return {"input_ids": ids, "labels": labels, "weights": weights, "tcl": ex["tcl"]}


def collate(batch, pad_id: int):
    mx = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, weights, attn = [], [], [], []
    for b in batch:
        n = mx - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * n)
        labels.append(b["labels"] + [-100] * n)
        weights.append(b["weights"] + [0.0] * n)
        attn.append([1] * len(b["input_ids"]) + [0] * n)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels":    torch.tensor(labels,    dtype=torch.long),
        "weights":   torch.tensor(weights,   dtype=torch.float32),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
    }


def temporal_weighted_loss(logits, labels, weights):
    """logits: (B, T, V); labels: (B, T); weights: (B, T)."""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_weights = weights[:, 1:].contiguous()

    per_tok = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        reduction="none",
        ignore_index=-100,
    ).view(shift_labels.shape)

    total = (per_tok * shift_weights).sum()
    denom = shift_weights.sum().clamp(min=1e-8)
    return total / denom


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def extract_sva(text: str) -> str:
    text = text.strip()
    m = re.search(r"```(?:systemverilog|sv|verilog)?\s*(.+?)```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    m = re.search(r"(assert\s+property\s*\(.*?\)\s*;)", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text.splitlines()[0].strip() if text else ""


@torch.no_grad()
def eval_model(tag: str, tok, model, tasks, device, max_new_tokens=128):
    model.eval()
    per_tcl = defaultdict(lambda: {"total": 0, "syntax_ok": 0, "tcl_match": 0})
    gen_dist = defaultdict(int)
    rows = []
    for t in tasks:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Generate an SVA assertion for: {t['nl']}"},
        ]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inp = tok(prompt, return_tensors="pt").to(device)
        out = model.generate(
            **inp,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            pad_token_id=tok.eos_token_id,
        )
        gen_ids = out[0, inp["input_ids"].shape[1]:]
        raw = tok.decode(gen_ids, skip_special_tokens=True)
        sva = extract_sva(raw)
        syn = syntax_check(sva)
        if syn["ok"]:
            cls = classify_tcl(sva)
            gen_tcl = cls[0] if isinstance(cls, tuple) else cls
        else:
            gen_tcl = None
        match = gen_tcl == t["expected_tcl"] if gen_tcl is not None else False

        per_tcl[t["expected_tcl"]]["total"] += 1
        if syn["ok"]:
            per_tcl[t["expected_tcl"]]["syntax_ok"] += 1
        if match:
            per_tcl[t["expected_tcl"]]["tcl_match"] += 1
        if gen_tcl is not None:
            gen_dist[gen_tcl] += 1

        rows.append({
            "id": t["id"], "expected_tcl": t["expected_tcl"],
            "generated_sva": sva, "syntax_ok": syn["ok"],
            "gen_tcl": gen_tcl, "match": match,
        })
    total = len(tasks)
    syn_ok = sum(r["syntax_ok"] for r in rows)
    match_ct = sum(r["match"] for r in rows)
    print(f"[{tag}] syntax {syn_ok}/{total} = {100*syn_ok/total:.1f}%   "
          f"TCL match {match_ct}/{total} = {100*match_ct/total:.1f}%")
    for lv in sorted(per_tcl):
        s = per_tcl[lv]
        print(f"  L{lv}: syn {s['syntax_ok']}/{s['total']}  "
              f"match {s['tcl_match']}/{s['total']}")
    print(f"  Generated TCL distribution: {dict(sorted(gen_dist.items()))}")
    return {
        "tag": tag,
        "syntax_valid_count": syn_ok,
        "tcl_match_count": match_ct,
        "total": total,
        "per_tcl": {str(k): v for k, v in per_tcl.items()},
        "gen_tcl_distribution": {str(k): v for k, v in gen_dist.items()},
        "rows": rows,
    }


# --------------------------------------------------------------------------- #
# Curriculum training
# --------------------------------------------------------------------------- #
def curriculum_train(tok, model, train_pool, device, args, log_path):
    random.seed(args.seed)
    by_level = defaultdict(list)
    for ex in train_pool:
        by_level[ex["tcl"]].append(ex)

    loss_log = []
    step = 0
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    if args.no_curriculum:
        # Ablation: single mixed stage. Match total gradient updates to curriculum run
        # (5 stages × epochs_per_stage epochs over each level's subset).
        pool = list(train_pool)
        random.shuffle(pool)
        total_epochs = 5 * args.epochs_per_stage
        ds = SVADataset(tok, pool, max_len=args.max_len, alpha=args.alpha)
        loader = DataLoader(
            ds, batch_size=args.batch_size, shuffle=True,
            collate_fn=lambda b: collate(b, pad_id=tok.pad_token_id or tok.eos_token_id),
        )
        model.train()
        print(f"\n=== No-curriculum mixed training: {len(pool)} examples × {total_epochs} epochs ===")
        for epoch in range(total_epochs):
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                loss = temporal_weighted_loss(out.logits.float(), batch["labels"], batch["weights"])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                loss_log.append({"step": step, "stage": 0, "epoch": epoch, "loss": float(loss.item())})
                if step % 10 == 0 or step == 1:
                    print(f"  step {step:3d}  epoch {epoch}  loss {loss.item():.4f}")
        with open(log_path, "w") as f:
            for row in loss_log:
                f.write(json.dumps(row) + "\n")
        return loss_log

    stages = [1, 2, 3, 4, 5]
    seen = []
    for stage_idx, lv in enumerate(stages):
        current = list(by_level.get(lv, []))
        replay = random.sample(seen, min(len(seen), max(1, int(0.2 * len(current))))) if seen else []
        stage_examples = current + replay
        random.shuffle(stage_examples)
        print(f"\n=== Stage {stage_idx+1}: L{lv}  examples={len(current)} replay={len(replay)} total={len(stage_examples)} ===")

        ds = SVADataset(tok, stage_examples, max_len=args.max_len, alpha=args.alpha)
        loader = DataLoader(
            ds, batch_size=args.batch_size, shuffle=True,
            collate_fn=lambda b: collate(b, pad_id=tok.pad_token_id or tok.eos_token_id),
        )
        model.train()
        for epoch in range(args.epochs_per_stage):
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                loss = temporal_weighted_loss(out.logits.float(), batch["labels"], batch["weights"])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                loss_log.append({"step": step, "stage": lv, "epoch": epoch, "loss": float(loss.item())})
                if step % 5 == 0 or step == 1:
                    print(f"  step {step:3d}  stage L{lv}  epoch {epoch}  loss {loss.item():.4f}")
        seen.extend(current)

    with open(log_path, "w") as f:
        for row in loss_log:
            f.write(json.dumps(row) + "\n")
    return loss_log


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/ssd2/xiacong/models/Qwen_Qwen2.5-0.5B-Instruct")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs-per-stage", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--alpha", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-curriculum", action="store_true",
                    help="Ablation: train on mixed-level batches (no stage ordering)")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_json = os.path.join(RESULTS_DIR, f"curriculum_sft_{timestamp}.json")
    out_log = os.path.join(RESULTS_DIR, f"curriculum_sft_{timestamp}.log")

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    print(f"Loading model from {args.model} …")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True,
    ).to(args.device)
    print(f"Model loaded. Params: {sum(p.numel() for p in model.parameters())/1e6:.1f} M")

    train_pool = build_training_set()
    eval_tasks = build_eval_set()
    print(f"Train pool: {len(train_pool)}  Eval tasks: {len(eval_tasks)}")

    # --- baseline ---
    print("\n=== Baseline evaluation (pre-training) ===")
    baseline = eval_model("baseline", tok, model, eval_tasks, args.device)

    # --- curriculum ---
    print("\n=== Curriculum fine-tuning (5 stages) ===")
    loss_log = curriculum_train(tok, model, train_pool, args.device, args, out_log)

    # --- post ---
    print("\n=== Post-training evaluation ===")
    post = eval_model("post", tok, model, eval_tasks, args.device)

    # --- delta ---
    delta_syntax = post["syntax_valid_count"] - baseline["syntax_valid_count"]
    delta_match = post["tcl_match_count"] - baseline["tcl_match_count"]
    per_level_delta = {}
    for lv in baseline["per_tcl"]:
        bs = baseline["per_tcl"][lv]
        ps = post["per_tcl"][lv]
        per_level_delta[lv] = {
            "syntax_ok_delta": ps["syntax_ok"] - bs["syntax_ok"],
            "tcl_match_delta": ps["tcl_match"] - bs["tcl_match"],
        }

    summary = {
        "timestamp": timestamp,
        "model": os.path.basename(args.model.rstrip("/")),
        "args": vars(args),
        "train_pool_size": len(train_pool),
        "eval_tasks_size": len(eval_tasks),
        "baseline": {k: v for k, v in baseline.items() if k != "rows"},
        "post_training": {k: v for k, v in post.items() if k != "rows"},
        "delta": {
            "syntax_valid": delta_syntax,
            "tcl_match": delta_match,
            "per_level": per_level_delta,
        },
        "final_losses_last5": loss_log[-5:] if len(loss_log) >= 5 else loss_log,
        "train_loss_first5": loss_log[:5],
        "baseline_rows": baseline["rows"],
        "post_rows": post["rows"],
    }
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("CURRICULUM SFT SUMMARY")
    print("=" * 60)
    print(f"Baseline    — syntax {baseline['syntax_valid_count']}/{baseline['total']}   "
          f"TCL match {baseline['tcl_match_count']}/{baseline['total']}")
    print(f"Post-train  — syntax {post['syntax_valid_count']}/{post['total']}   "
          f"TCL match {post['tcl_match_count']}/{post['total']}")
    print(f"Δ syntax = {delta_syntax:+d},  Δ TCL match = {delta_match:+d}")
    print(f"Saved: {out_json}")
    print(f"Saved: {out_log}")


if __name__ == "__main__":
    main()
