"""run_opd_codev_to_qwen.py
On-Policy Distillation (OPD) and Reward-Weighted OPD (RWOPD) of
CodeV-SVA-14B (teacher) into Qwen2.5-Coder-7B + LoRA (student) — paper §4.1.

Pipeline per step:
  1. Sample K rollouts from the student given a training prompt
     (K=1 by default → plain OPD; K>1 → RWOPD)
  2. If --enable-pec-filter, score each rollout via the open SymbiYosys+Z3
     PEC against the prompt's reference SVA; map verdict → weight per
     paper Eq. 2 (EQUIV 1.0 / IMPL_REF→LM 0.6 / IMPL_LM→REF 0.4 / else 0.0).
     In --filter-mode=strict only EQUIVALENT rollouts are kept ('Strict RWOPD').
  3. Forward teacher (no grad) and student (with grad) on each surviving
     (prompt + rollout); compute forward-KL(teacher || student) on the
     response-token slice.
  4. Loss = Σ_i w_i · L_OPD(y_i) / Σ_i w_i  (paper Eq. 3). Empty S(p) →
     skip the prompt (zero gradient that step).
  5. Backward + LoRA update.

Vocab compatibility (paper §4.1 / App. B): both tokenizers share the
first V_MIN = 151,643 token IDs in identical positions. Everything
past that index is special-token slots that diverge between Qwen2.5
and Qwen3, so both logit heads are truncated to V_MIN before softmax.

Why OPD instead of plain SFT on teacher data:
  - Plain SFT (already done in run_sft_with_reasoning.py) only learns
    what the teacher *did* say on its own training prompts. OPD learns
    what the teacher *would* say on the student's actual rollout
    distribution. This corrects student-specific failure modes that
    offline data can't reach.

Usage:
  # Plain OPD (Table 1 'OPD from CodeV-SVA-14B' row)
  python training/rwopd/run_opd_codev_to_qwen.py \\
    --teacher ${TEACHER_MODEL} \\
    --student-base ${STUDENT_MODEL} \\
    --student-adapter results/sft_reasoning_*/checkpoint-12000 \\
    --pool data/train/sft_with_reasoning/sft_train.jsonl \\
    --output-dir results/opd_<TS> \\
    --max-steps 2000 --max-new-tokens 1024 --lr 5e-6

  # RWOPD (Table 1 headline '+ RWOPD from CodeV-SVA-14B' row, K=4 paper default)
  python training/rwopd/run_opd_codev_to_qwen.py \\
    --teacher ${TEACHER_MODEL} \\
    --student-base ${STUDENT_MODEL} \\
    --student-adapter results/sft_reasoning_*/checkpoint-12000 \\
    --pool data/train/sft_with_reasoning/sft_train.jsonl \\
    --output-dir results/rwopd_<TS> \\
    --k-rollouts 4 --enable-pec-filter --filter-mode implies \\
    --max-steps 200 --max-new-tokens 1024 --lr 5e-6
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
from torch.optim import AdamW
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          get_cosine_schedule_with_warmup)

# Paper §4.1 / App. B: shared vocab intersection between Qwen2.5-Coder-7B
# (student) and CodeV-SVA-14B / Qwen3-14B (teacher).
V_MIN = 151643

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "eval"))

from run_funcatk_eval import (
    FVEVAL_SYSTEM_PROMPT,
    build_fveval_user_prompt as build_user,
)


def build_prompt(tok, nl: str, rtl_context: str, rtl_cap: int = 4500) -> torch.Tensor:
    rtl = (rtl_context or "")
    if rtl_cap and len(rtl) > rtl_cap: rtl = rtl[:rtl_cap]
    msgs = [
        {"role": "system", "content": FVEVAL_SYSTEM_PROMPT},
        {"role": "user", "content": build_user(nl, rtl)},
    ]
    text = tok.apply_chat_template(msgs, tokenize=False,
                                    add_generation_prompt=True)
    return tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True,
                    help="path to teacher checkpoint (CodeV-SVA-14B)")
    ap.add_argument("--student-base", required=True,
                    help="path to base student (Qwen2.5-Coder-7B-Instruct)")
    ap.add_argument("--student-adapter", required=True,
                    help="path to LoRA adapter to start from "
                         "(SFT v4 reasoning ckpt-12000)")
    ap.add_argument("--pool", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--prompt-cap", type=int, default=4096,
                    help="hard cap on prompt token length (after chat tmpl)")
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="student sampling temperature")
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--save-every-steps", type=int, default=500)
    ap.add_argument("--log-every-steps", type=int, default=10)
    ap.add_argument("--rtl-cap", type=int, default=4500)
    ap.add_argument("--seed", type=int, default=0)

    # ---- RWOPD: K-rollout + PEC filter + reward weighting (paper §4.1 / Eq. 2–3)
    ap.add_argument("--k-rollouts", type=int, default=1,
                    help="Number of rollouts sampled per prompt. Paper §4.1 "
                         "default for RWOPD: 4. K=1 (default) keeps the original "
                         "unfiltered OPD behavior; K>1 enables the RWOPD code "
                         "path. The Fig. 3 ablation runs K ∈ {1,2,4,8}.")
    ap.add_argument("--enable-pec-filter", action="store_true",
                    help="Filter the K rollouts via the open SymbiYosys+Z3 PEC "
                         "against the prompt's reference SVA. Only rollouts "
                         "with verdict in {EQUIVALENT, IMPLIES_REF_TO_LM, "
                         "IMPLIES_LM_TO_REF} contribute gradient (paper Eq. 2).")
    ap.add_argument("--filter-mode", default="implies",
                    choices=["strict", "implies"],
                    help="strict = EQUIVALENT-only ('Strict RWOPD' in paper). "
                         "implies (default) = EQUIVALENT + both implication "
                         "verdicts kept and reward-weighted ('RWOPD' headline).")
    ap.add_argument("--pec-depth", type=int, default=15,
                    help="PEC BMC depth in cycles (paper App. B: 60s budget).")
    ap.add_argument("--pec-timeout", type=int, default=60,
                    help="per-BMC-instance timeout in seconds.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / "opd.log"

    def log(*a):
        msg = " ".join(str(x) for x in a)
        print(msg, flush=True)
        with open(log_file, "a") as f: f.write(msg + "\n")

    log(f"[args] {vars(args)}")

    # ---- 1. tokenizer (shared, verified compatible) ----
    tok = AutoTokenizer.from_pretrained(args.student_base, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    # ---- 2. load teacher (frozen, eval) ----
    log(f"[teacher] loading {args.teacher}")
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher, torch_dtype=torch.bfloat16,
        trust_remote_code=True, device_map={"": args.device})
    teacher.eval()
    for p in teacher.parameters(): p.requires_grad_(False)

    # ---- 3. load student (LoRA, train) ----
    log(f"[student] loading base {args.student_base}")
    student_base = AutoModelForCausalLM.from_pretrained(
        args.student_base, torch_dtype=torch.bfloat16,
        trust_remote_code=True, device_map={"": args.device})
    from peft import PeftModel
    log(f"[student] loading adapter {args.student_adapter}")
    student = PeftModel.from_pretrained(student_base, args.student_adapter,
                                          is_trainable=True)
    student.train()
    # gradient_checkpointing + LoRA: input embeddings must propagate grad
    # so the recomputed forward in the checkpointed segment connects to
    # the LoRA adapter's trainable params. Without this, loss has no
    # grad_fn (RuntimeError: element 0 ... does not require grad).
    student.enable_input_require_grads()
    student.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    student.config.use_cache = False
    student.print_trainable_parameters()

    # ---- 4. data ----
    log(f"[data] loading {args.pool}")
    rows = []
    with open(args.pool) as f:
        for ln in f:
            ln = ln.strip()
            if ln: rows.append(json.loads(ln))
    log(f"[data] {len(rows)} prompts")

    # ---- 5. optim ----
    trainable = [p for p in student.parameters() if p.requires_grad]
    optim = AdamW(trainable, lr=args.lr, weight_decay=0.01)
    warmup = int(args.max_steps * args.warmup_frac)
    sched = get_cosine_schedule_with_warmup(optim, warmup, args.max_steps)
    log(f"[opt] {sum(p.numel() for p in trainable)/1e6:.1f}M trainable  "
        f"lr={args.lr}  warmup={warmup}  max_steps={args.max_steps}")

    # ---- 5b. PEC oracle (only loaded when --enable-pec-filter) ----
    pec_call = None
    if args.enable_pec_filter:
        from src.pec_yosys import prop_equivalence
        pec_call = prop_equivalence
        log(f"[pec] filter enabled mode={args.filter_mode} K={args.k_rollouts} "
            f"depth={args.pec_depth} timeout={args.pec_timeout}")

    # Paper §4.1 Eq. 2 — verifier-equivalence rollout weights.
    PEC_WEIGHTS = {
        "EQUIVALENT":        1.0,
        "IMPLIES_REF_TO_LM": 0.6,
        "IMPLIES_LM_TO_REF": 0.4,
    }

    def _rollout_weight(verdict: str) -> float:
        """Map PEC verdict → rollout weight (paper Eq. 2). Strict mode keeps
        only EQUIVALENT; implies mode keeps all three positive verdicts."""
        if args.filter_mode == "strict":
            return 1.0 if verdict == "EQUIVALENT" else 0.0
        return PEC_WEIGHTS.get(verdict, 0.0)

    def _decode_response(rollout_ids, prompt_len):
        """Decode the rollout's response-only token slice as a string."""
        resp = rollout_ids[0, prompt_len:].detach().cpu().tolist()
        return tok.decode(resp, skip_special_tokens=True)

    def _opd_loss_for_rollout(rollout_ids, P, R):
        """Run teacher (no grad) and student (with grad) over (prompt+rollout),
        truncate both heads to V_MIN, and return the response-token forward-KL
        averaged over R tokens. This is L_OPD(y) from paper Eq. 1."""
        with torch.no_grad():
            tlog = teacher(rollout_ids, use_cache=False).logits
            t_resp_logits = tlog[:, P - 1:P + R - 1, :]
        slog = student(rollout_ids, use_cache=False).logits
        s_resp_logits = slog[:, P - 1:P + R - 1, :]
        assert t_resp_logits.size(-1) >= V_MIN, (
            f"teacher head {t_resp_logits.size(-1)} < V_MIN={V_MIN}")
        assert s_resp_logits.size(-1) >= V_MIN, (
            f"student head {s_resp_logits.size(-1)} < V_MIN={V_MIN}")
        t_resp_logits = t_resp_logits[..., :V_MIN]
        s_resp_logits = s_resp_logits[..., :V_MIN]
        t_probs = F.softmax(t_resp_logits.float(), dim=-1)
        s_log_probs = F.log_softmax(s_resp_logits.float(), dim=-1)
        return -(t_probs * s_log_probs).sum(-1).mean()

    # ---- 6. training loop (paper §4.1 RWOPD) ----
    log(f"[train] start  device={args.device}  K={args.k_rollouts}  "
        f"filter={'on' if args.enable_pec_filter else 'off'}")
    step = 0; t0 = time.time()
    n_truncated = 0
    n_empty_set = 0
    while step < args.max_steps:
        for row in rows:
            if step >= args.max_steps: break
            prompt_ids = build_prompt(tok, row.get("nl") or "",
                                       row.get("rtl_context") or "",
                                       args.rtl_cap).to(args.device)
            if prompt_ids.size(1) > args.prompt_cap:
                # Skip overlong prompts (would OOM during forward)
                n_truncated += 1; continue
            P = prompt_ids.size(1)
            ref_sva = row.get("reference_sva") or row.get("sva") or ""
            rtl_ctx = row.get("rtl_context") or ""

            # 6a. Sample K rollouts from the current student (no grad).
            #     K=1 with --enable-pec-filter off reduces to plain OPD.
            student.eval()
            rollouts = []
            with torch.no_grad():
                for _ in range(max(1, args.k_rollouts)):
                    out_ids = student.generate(
                        prompt_ids,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=True,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        pad_token_id=tok.pad_token_id,
                    )
                    R = out_ids.size(1) - P
                    if R > 0:
                        rollouts.append((out_ids, R))
            student.train()
            if not rollouts:
                continue

            # 6b. PEC filter + reward weighting (paper Eq. 2). If the filter
            #     is off we treat every rollout as EQUIVALENT (weight 1).
            kept = []
            if args.enable_pec_filter and pec_call is not None and ref_sva:
                for out_ids, R in rollouts:
                    cand = _decode_response(out_ids, P)
                    try:
                        r = pec_call(cand, ref_sva, rtl_ctx,
                                     depth=args.pec_depth,
                                     timeout=args.pec_timeout)
                        verdict = r.verdict
                    except Exception:
                        verdict = "UNSUPPORTED"
                    w = _rollout_weight(verdict)
                    if w > 0:
                        kept.append((out_ids, R, w, verdict))
            else:
                for out_ids, R in rollouts:
                    kept.append((out_ids, R, 1.0, "OPD"))

            # 6c. Paper Eq. 3: if no rollout survives, contribute no gradient.
            if not kept:
                n_empty_set += 1
                continue

            # 6d. Reward-weighted forward-KL: L = Σ w_i · L_OPD(y_i) / Σ w_i.
            total_w = sum(w for _, _, w, _ in kept)
            loss = None
            for out_ids, R, w, _verdict in kept:
                li = _opd_loss_for_rollout(out_ids, P, R) * (w / total_w)
                loss = li if loss is None else loss + li
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optim.step(); sched.step(); optim.zero_grad()
            step += 1
            R = kept[0][1]  # for logging only

            if step % args.log_every_steps == 0:
                dt = time.time() - t0
                kept_summary = ",".join(f"{v[:3]}({w:.1f})"
                                         for _, _, w, v in kept[:4])
                log(f"  step {step}/{args.max_steps}  loss={loss.item():.4f}  "
                    f"lr={sched.get_last_lr()[0]:.2e}  R={R}  "
                    f"K={args.k_rollouts}/kept={len(kept)} [{kept_summary}]  "
                    f"{step / max(dt, 1):.2f} step/s  "
                    f"eta={(args.max_steps - step) / max(step / max(dt,1), 1e-3):.0f}s  "
                    f"truncated={n_truncated} empty_S={n_empty_set}")

            if step % args.save_every_steps == 0:
                ck = out_dir / f"checkpoint-{step}"
                student.save_pretrained(str(ck))
                tok.save_pretrained(str(ck))
                log(f"  [ckpt] saved {ck}")

    # final save
    final = out_dir / "final"
    student.save_pretrained(str(final))
    tok.save_pretrained(str(final))
    log(f"[done] saved {final}  total_time={time.time() - t0:.0f}s  "
        f"truncated={n_truncated}  empty_S={n_empty_set}")


if __name__ == "__main__":
    main()
