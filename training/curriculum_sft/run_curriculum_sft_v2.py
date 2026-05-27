#!/usr/bin/env python3
"""
run_curriculum_sft_v2.py — Component-2 curriculum fine-tuning on the unified
SFT pool (data/train/sft/codev_sft_unified.jsonl with temporal_class field; auto-buckets to C1/C2/C3), evaluated on the held-out
NL2SVA-Human test set (data/test/nl2sva_human.jsonl).

Differences from run_curriculum_sft.py:
  - Reads new JSONL schema (fields: nl, reference_sva, expected_tcl, hash)
  - Quality-filters to samples with genuine NL (drops empty + placeholder NL
    like "[ASSERT] ...")
  - Scales to arbitrary model size; supports --bf16 and --gradient-checkpointing
  - 3-stage curriculum (C1 → C2 → C3) collapsed from the 5-level TCL
    via {1→C1, 2/3/4→C2, 5→C3}. Same convention as
    data/test/manifest_tc_split.json. Older 5-stage L1..L5 mode is gone.
  - Evaluates on NL2SVA-Human at the end of every stage

Usage:
  python scripts/run_curriculum_sft_v2.py \\
      --model ${STUDENT_MODEL} \\
      --device cuda:0 \\
      --output-dir results/sft_qwen_coder_7b
"""
import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer


# --- DDP helpers --------------------------------------------------------
def _ddp_env():
    """Return (is_ddp, local_rank, world_size, rank)."""
    if "LOCAL_RANK" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        return (True,
                int(os.environ["LOCAL_RANK"]),
                int(os.environ["WORLD_SIZE"]),
                int(os.environ.get("RANK", os.environ["LOCAL_RANK"])))
    return (False, 0, 1, 0)


def _is_main(rank: int) -> bool:
    return rank == 0


def _rprint(rank: int, *a, **kw):
    if _is_main(rank):
        print(*a, **kw)

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from src.tcl import classify_tcl
from src.mock_verifier import syntax_check
from src.temporal_loss import TEMPORAL_OPS

DEFAULT_SFT_DIR = REPO_ROOT / "data" / "train" / "sft"
TEST_JSONL = REPO_ROOT / "data" / "test" / "nl2sva_human.jsonl"
VLLM_EVAL_SCRIPT = REPO_ROOT / "eval" / "run_eval_nl2sva_human_vllm.py"
FUNCATK_SCRIPT = REPO_ROOT / "eval" / "run_funcatk_eval.py"

SYSTEM_PROMPT = (
    "You are an expert in SystemVerilog Assertions (SVA). Given a "
    "natural-language description of a design property, output ONE "
    "syntactically correct SVA assertion. Emit ONLY the SVA — no explanation "
    "or markdown fences. Match temporal complexity to the spec: bare `##N` "
    "for fixed delays, `##[a:b]` for ranged, `|->`/`|=>` only when antecedent-"
    "consequent, `s_eventually`/`s_until` for liveness."
)

# Single source of truth for the fveval prompt — import from run_funcatk_eval
# so train and test cannot drift. Both `format_example` (training) and
# `_run_funcatk_eval` (eval subprocess) end up rendering byte-identical
# prompts on the same (nl, rtl) pair.
import sys as _sys
_sys.path.insert(0, str(REPO_ROOT / "eval"))
from run_funcatk_eval import (
    FVEVAL_SYSTEM_PROMPT,
    build_fveval_user_prompt as _ev_build_fveval_user_prompt,
)


def build_fveval_user_prompt(nl: str, rtl_context: str,
                              rtl_cap: int = 6000) -> str:
    """Wrapper that caps rtl_context length before delegating to the
    canonical builder in run_funcatk_eval.py. The eval-side function does
    not cap, so we pre-truncate here for training-time prompt budget."""
    rtl = (rtl_context or "")
    if len(rtl) > rtl_cap:
        rtl = rtl[:rtl_cap]
    return _ev_build_fveval_user_prompt(nl, rtl)

PLACEHOLDER_RE = re.compile(r"^\s*\[(?:ASSERT|ASSUME|COVER)[^\]]*\]\s*$",
                            re.IGNORECASE)


def is_real_nl(nl: str) -> bool:
    nl = (nl or "").strip()
    if len(nl) < 5:
        return False
    if PLACEHOLDER_RE.match(nl):
        return False
    return True


# Map fine-grained TCL (1..5) to the 3-class curriculum used here:
#   C1 = combinational  (L1)
#   C2 = bounded temporal (L2 + L3 + L4)
#   C3 = liveness  (L5)
# Same convention as data/test/manifest_tc_split.json.
TCL_TO_CLASS = {1: "C1", 2: "C2", 3: "C2", 4: "C2", 5: "C3"}
CURRICULUM_STAGES = ("C1", "C2", "C3")


def _find_class_file(sft_dir: Path, cls: str) -> Path | None:
    """Look for a per-class shard if one exists; else fall back to the
    single unified file (which we'll bucket in memory)."""
    direct = sft_dir / f"codev_sft_unified_{cls}.jsonl"
    if direct.exists():
        return direct
    matches = sorted(sft_dir.glob(f"*_{cls}.jsonl"))
    return matches[0] if matches else None


def load_sft_by_class(sft_dir: Path, explicit_jsonl: Path | None = None):
    """Return dict {"C1","C2","C3" -> list of {nl, sva, tcl}} filtered
    to real-NL only. If `explicit_jsonl` is given, bucket from that
    single file; otherwise read per-class shards or fall back to a
    well-known filename in `sft_dir`."""
    by_cls = defaultdict(list)
    total = kept = placeholder = empty = 0

    if explicit_jsonl is not None and explicit_jsonl.exists():
        print(f"[data] loading from explicit file: {explicit_jsonl}")
        for line in open(explicit_jsonl):
            r = json.loads(line)
            total += 1
            nl = r.get("nl", "")
            if not nl or len(nl.strip()) < 5:
                empty += 1; continue
            if PLACEHOLDER_RE.match(nl):
                placeholder += 1; continue
            tcl = r.get("expected_tcl")
            cls = r.get("temporal_class") or TCL_TO_CLASS.get(tcl)
            if cls not in CURRICULUM_STAGES:
                continue
            by_cls[cls].append({
                "nl": nl, "sva": r["reference_sva"],
                "tcl": tcl, "id": r.get("id", ""),
                "rtl_context": r.get("rtl_context", "") or "",
            })
            kept += 1
        print(f"[data] loaded SFT: kept={kept}/{total}  "
              f"(dropped empty={empty} placeholder={placeholder})")
        for cls in CURRICULUM_STAGES:
            print(f"  {cls}: {len(by_cls.get(cls, []))}")
        return by_cls

    # Per-class shards exist? Use them.
    shard_paths = {c: _find_class_file(sft_dir, c) for c in CURRICULUM_STAGES}
    if all(p is not None for p in shard_paths.values()):
        for cls, path in shard_paths.items():
            for line in open(path):
                r = json.loads(line)
                total += 1
                nl = r.get("nl", "")
                if not nl or len(nl.strip()) < 5:
                    empty += 1; continue
                if PLACEHOLDER_RE.match(nl):
                    placeholder += 1; continue
                by_cls[cls].append({
                    "nl": nl, "sva": r["reference_sva"],
                    "tcl": r["expected_tcl"], "id": r.get("id", ""),
                    "rtl_context": r.get("rtl_context", "") or "",
                })
                kept += 1
    else:
        # Fall back to a single unified jsonl. Bucket via expected_tcl.
        candidates = [
            sft_dir / "codev_sft_unified.jsonl",
            sft_dir / "sft_train.jsonl",
            sft_dir / "master_sft.jsonl",
        ]
        unified = next((p for p in candidates if p.exists()), None)
        if unified is None:
            print(f"[data] no recognised SFT file in {sft_dir}")
            return by_cls
        print(f"[data] loading from unified file: {unified.name}")
        for line in open(unified):
            r = json.loads(line)
            total += 1
            nl = r.get("nl", "")
            if not nl or len(nl.strip()) < 5:
                empty += 1; continue
            if PLACEHOLDER_RE.match(nl):
                placeholder += 1; continue
            tcl = r.get("expected_tcl")
            cls = r.get("temporal_class") or TCL_TO_CLASS.get(tcl)
            if cls not in CURRICULUM_STAGES:
                continue
            by_cls[cls].append({
                "nl": nl, "sva": r["reference_sva"],
                "tcl": tcl, "id": r.get("id", ""),
                "rtl_context": r.get("rtl_context", "") or "",
            })
            kept += 1

    print(f"[data] loaded SFT: kept={kept}/{total}  "
          f"(dropped empty={empty} placeholder={placeholder})")
    for cls in CURRICULUM_STAGES:
        print(f"  {cls}: {len(by_cls.get(cls, []))}")
    return by_cls


def load_test_set():
    tasks = []
    with open(TEST_JSONL) as f:
        for line in f:
            r = json.loads(line)
            tasks.append({
                "id": r.get("id", ""),
                "nl": r.get("nl", ""),
                "expected_tcl": r.get("expected_tcl", 0),
                "reference_sva": r.get("reference_sva", ""),
            })
    print(f"[test] NL2SVA-Human: {len(tasks)} tasks")
    return tasks


def bucket_tasks_by_class(tasks):
    """Bucket eval tasks into the same C1/C2/C3 buckets used by the
    training curriculum, via expected_tcl → TCL_TO_CLASS."""
    by_cls = defaultdict(list)
    for t in tasks:
        cls = TCL_TO_CLASS.get(int(t.get("expected_tcl", 0)))
        if cls:
            by_cls[cls].append(t)
    return by_cls


# --- training -----------------------------------------------------------

def format_example(tok, nl, sva, max_len=4096, rtl_context: str = "",
                   rtl_cap: int = 6000):
    """Build a (prompt, response) pair using the SAME fveval template as
    run_funcatk_eval.py — keeps train and test I/O schemas aligned."""
    msgs = [
        {"role": "system", "content": FVEVAL_SYSTEM_PROMPT},
        {"role": "user",
         "content": build_fveval_user_prompt(nl, rtl_context, rtl_cap)},
    ]
    prompt = tok.apply_chat_template(msgs, tokenize=False,
                                     add_generation_prompt=True)
    # Wrap response in the expected ```systemverilog ... ``` fence so the
    # model learns to produce that exact form at test time.
    response = f"```systemverilog\n{sva.strip()}\n```" + tok.eos_token
    pids = tok(prompt, add_special_tokens=False)["input_ids"]
    rids = tok(response, add_special_tokens=False)["input_ids"]
    ids = pids + rids
    lbl = [-100] * len(pids) + rids[:]
    if len(ids) > max_len:
        # If oversize, prefer to truncate the prompt's RTL block from the
        # right (already capped to rtl_cap above) — here we hard-cap the
        # whole sequence as a safety net.
        ids = ids[:max_len]; lbl = lbl[:max_len]
    return ids, lbl


def temporal_weights(tok, labels, alpha=3.0):
    w = [1.0] * len(labels)
    for i, tid in enumerate(labels):
        if tid == -100:
            w[i] = 0.0; continue
        s = tok.decode([tid])
        if any(op in s for op in TEMPORAL_OPS):
            w[i] = alpha
    return w


class SVADataset(Dataset):
    def __init__(self, tok, examples, max_len, alpha):
        self.tok, self.ex, self.max_len, self.alpha = tok, examples, max_len, alpha

    def __len__(self): return len(self.ex)

    def __getitem__(self, i):
        e = self.ex[i]
        ids, lbl = format_example(
            self.tok, e["nl"], e["sva"], self.max_len,
            rtl_context=e.get("rtl_context", "") or "",
        )
        w = temporal_weights(self.tok, lbl, self.alpha)
        return {"input_ids": ids, "labels": lbl, "weights": w}


def collate(batch, pad):
    mx = max(len(b["input_ids"]) for b in batch)
    ids, lbl, w, am = [], [], [], []
    for b in batch:
        n = mx - len(b["input_ids"])
        ids.append(b["input_ids"] + [pad] * n)
        lbl.append(b["labels"] + [-100] * n)
        w.append(b["weights"] + [0.0] * n)
        am.append([1] * len(b["input_ids"]) + [0] * n)
    return {"input_ids": torch.tensor(ids), "labels": torch.tensor(lbl),
            "weights": torch.tensor(w, dtype=torch.float32),
            "attention_mask": torch.tensor(am)}


def temporal_weighted_loss(logits, labels, weights):
    """Paper §4.2 TT-CE: L = (1/T) Σ w_t · CE_t over response tokens, where
    T is the number of valid (non-ignore-index) tokens. Normalizing by the
    token count rather than by Σ w_t preserves α as the actual per-token
    upweight factor; dividing by Σ w_t would silently cancel α out."""
    sl = logits[:, :-1, :].contiguous()
    slab = labels[:, 1:].contiguous()
    sw = weights[:, 1:].contiguous()
    pt = F.cross_entropy(sl.reshape(-1, sl.size(-1)), slab.reshape(-1),
                          reduction="none", ignore_index=-100).view(slab.shape)
    valid = (slab != -100).float()
    n_tokens = valid.sum().clamp(min=1.0)
    return (pt * sw * valid).sum() / n_tokens


# --- eval ---------------------------------------------------------------

def extract_sva(text):
    text = text.strip()
    m = re.search(r"```(?:systemverilog|sv|verilog)?\s*(.+?)```", text, re.DOTALL)
    if m: text = m.group(1).strip()
    m = re.search(r"(assert\s+property\s*\(.*?\)\s*;)", text, re.DOTALL | re.IGNORECASE)
    if m: return m.group(1).strip()
    return text.splitlines()[0].strip() if text else ""


@torch.no_grad()
def eval_on_test(tag, tok, model, tasks, device, max_new=256):
    model.eval()
    per_tcl = defaultdict(lambda: {"total": 0, "syn": 0, "match": 0})
    rows = []
    for t in tasks:
        # Use the SAME fveval template as run_funcatk_eval.py and
        # format_example so train/eval are consistent.
        msgs = [{"role": "system", "content": FVEVAL_SYSTEM_PROMPT},
                {"role": "user",
                 "content": build_fveval_user_prompt(
                     t["nl"], t.get("rtl_context", "") or "")}]
        prompt = tok.apply_chat_template(msgs, tokenize=False,
                                          add_generation_prompt=True)
        inp = tok(prompt, return_tensors="pt").to(device)
        out = model.generate(**inp, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        raw = tok.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True)
        sva = extract_sva(raw)
        syn = syntax_check(sva)
        gen_tcl = None
        if syn["ok"]:
            cls = classify_tcl(sva)
            gen_tcl = cls[0] if isinstance(cls, tuple) else cls
        match = gen_tcl == t["expected_tcl"] if gen_tcl is not None else False
        per_tcl[t["expected_tcl"]]["total"] += 1
        if syn["ok"]: per_tcl[t["expected_tcl"]]["syn"] += 1
        if match:     per_tcl[t["expected_tcl"]]["match"] += 1
        rows.append({"id": t["id"], "sva": sva, "gen_tcl": gen_tcl,
                     "match": match, "syn_ok": syn["ok"]})
    total = len(tasks)
    syn = sum(v["syn"] for v in per_tcl.values())
    mat = sum(v["match"] for v in per_tcl.values())
    print(f"[{tag}] syn {syn}/{total}={100*syn/total:.1f}%  "
          f"match {mat}/{total}={100*mat/total:.1f}%  "
          f"per-TCL: " + " ".join(
              f"L{lv}={per_tcl[lv]['match']}/{per_tcl[lv]['total']}"
              for lv in sorted(per_tcl)))
    return {"tag": tag, "syntax": syn, "match": mat, "total": total,
            "per_tcl": {str(k): v for k, v in per_tcl.items()}, "rows": rows}


def stage_metric(report, target_cls):
    """Aggregate per-TCL eval stats into the C-class bucket the current
    stage was trained on."""
    target_lvs = [str(lv) for lv, c in TCL_TO_CLASS.items() if c == target_cls]
    total = match = 0
    for lv in target_lvs:
        stats = report.get("per_tcl", {}).get(lv, {})
        total += int(stats.get("total", 0))
        match += int(stats.get("match", 0))
    return (match / total) if total else 0.0


def _run_one_vllm_eval(tag, model_dir, tasks_jsonl: Path, args) -> dict:
    """Single vLLM subprocess call against one tasks.jsonl. Returns the
    raw eval report (per_tcl + total/match).

    LoRA mode: pass the original base model via --model and the saved
    adapter directory via --adapter. Full-FT mode: pass the saved
    checkpoint directly via --model.
    """
    if not args.vllm_eval_gpu:
        raise RuntimeError("vLLM eval requested but --vllm-eval-gpu is empty")
    with tempfile.TemporaryDirectory(prefix="sft_eval_") as td:
        out_path = Path(td) / "eval.json"
        if getattr(args, "use_lora", False):
            cmd = [
                sys.executable, str(VLLM_EVAL_SCRIPT),
                "--model", str(args.model),
                "--adapter", str(model_dir),
                "--tasks", str(tasks_jsonl),
                "--prompt-format", args.eval_prompt_format,
                "--max-new-tokens", str(args.eval_max_new),
                "--gpu-memory-utilization", str(args.vllm_gpu_memory_utilization),
                "--tensor-parallel-size", str(args.vllm_tensor_parallel_size),
                "--tag", tag,
                "--output", str(out_path),
            ]
        else:
            cmd = [
                sys.executable, str(VLLM_EVAL_SCRIPT),
                "--model", str(model_dir),
                "--tasks", str(tasks_jsonl),
                "--prompt-format", args.eval_prompt_format,
                "--max-new-tokens", str(args.eval_max_new),
                "--gpu-memory-utilization", str(args.vllm_gpu_memory_utilization),
                "--tensor-parallel-size", str(args.vllm_tensor_parallel_size),
                "--tag", tag,
                "--output", str(out_path),
            ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = args.vllm_eval_gpu
        env["PYTHONPATH"] = (str(REPO_ROOT) + ":"
                             + env.get("PYTHONPATH", ""))
        subprocess.run(cmd, check=True, env=env)
        with open(out_path) as f:
            return json.load(f)


def run_vllm_eval(tag, model_dir, _unused_tasks, args):
    """Run vLLM eval on BOTH nl2sva_human and nl2sva_machine. Returns a
    combined report:

      {
        "by_test": {
            "human":   {<full per_tcl eval JSON>},
            "machine": {<full per_tcl eval JSON>},
        },
        # back-compat: aggregated across both test sets
        "total": <human.total + machine.total>,
        "match": <sum of matches>,
        "per_tcl": {
            "1": {"total": ..., "match": ...},
            ...
        },
      }

    The `_unused_tasks` argument is kept for back-compat with the older
    per-stage filtered eval; we now always run on the full test sets.
    """
    eval_paths = {
        "human": Path(args.eval_tasks_human),
        "machine": Path(args.eval_tasks_machine),
    }
    by_test = {}
    combined_per_tcl = {}
    total = match = 0
    for name, path in eval_paths.items():
        if not path.exists():
            print(f"  [eval-skip] {name}: missing tasks file {path}")
            continue
        rep = _run_one_vllm_eval(f"{tag}__{name}", model_dir, path, args)
        by_test[name] = rep
        total += int(rep.get("total", 0))
        match += int(rep.get("match", 0))
        for k, v in (rep.get("per_tcl") or {}).items():
            slot = combined_per_tcl.setdefault(k, {"total": 0, "match": 0})
            slot["total"] += int(v.get("total", 0))
            slot["match"] += int(v.get("match", 0))
    return {
        "by_test": by_test,
        "total": total,
        "match": match,
        "per_tcl": combined_per_tcl,
    }


def _run_funcatk_eval(tag, model_dir, args) -> float:
    """Run scripts/run_funcatk_eval.py against nl2sva_human and return
    Func@1 as a fraction in [0, 1]. The eval subprocess runs vLLM on
    --funcatk-eval-gpu (shared with rank 0 — rank 0 is paused at the
    DDP barrier during eval, holding memory but not computing)."""
    if not args.funcatk_eval_gpu:
        raise RuntimeError("--funcatk-eval-gpu required when --eval-each-stage is set")
    with tempfile.TemporaryDirectory(prefix="sft_funcatk_") as td:
        out_path = Path(td) / "eval.json"
        cmd = [sys.executable, str(FUNCATK_SCRIPT)]
        if getattr(args, "use_lora", False):
            cmd += ["--model", str(args.model),
                    "--adapter", str(model_dir)]
        else:
            cmd += ["--model", str(model_dir)]
        cmd += [
            "--tasks", str(args.eval_tasks_human),
            "--prompt-format", "fveval",
            "--num-samples", str(args.funcatk_num_samples),
            "--ks", "1",
            "--temperature", "0.8",
            "--top-p", "0.95",
            "--max-new-tokens", "1024",
            "--gpu-memory-utilization", str(args.funcatk_gpu_mem_util),
            "--max-model-len", str(args.funcatk_max_model_len),
            "--workers", "16",
            "--skip-coverage-check",
            "--skip-greedy-diagnostic",
            "--tag", tag,
            "--output", str(out_path),
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = args.funcatk_eval_gpu
        env["PYTHONPATH"] = (str(REPO_ROOT) + ":"
                             + env.get("PYTHONPATH", ""))
        # Scrub torchrun/DDP env vars — vLLM's EngineCore tries to
        # init_process_group when these are set, then deadlocks against
        # the outer torchrun's rendezvous. Funcatk eval is single-GPU,
        # we want a fresh local distributed env.
        for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE",
                  "GROUP_RANK", "ROLE_RANK", "ROLE_WORLD_SIZE",
                  "MASTER_ADDR", "MASTER_PORT",
                  "TORCHELASTIC_RUN_ID", "TORCHELASTIC_USE_AGENT_STORE",
                  "TORCHELASTIC_RESTART_COUNT", "TORCHELASTIC_MAX_RESTARTS",
                  "TORCHELASTIC_ERROR_FILE", "OMP_NUM_THREADS"):
            env.pop(k, None)
        try:
            subprocess.run(cmd, check=True, env=env)
            with open(out_path) as f:
                rep = json.load(f)
            return float(rep["overall"]["func@1"]) / 100.0
        except subprocess.CalledProcessError as e:
            print(f"  [funcatk-eval-FAIL] tag={tag} returncode={e.returncode}")
            return -1.0


# --- curriculum --------------------------------------------------------

def _parse_stage_floats(s: str, default_len: int = 3) -> list:
    out = [float(x) for x in s.split(",") if x.strip()]
    if len(out) != default_len:
        raise SystemExit(f"--stage-* expects {default_len} comma-separated "
                         f"values, got {len(out)}: {s!r}")
    return out


def curriculum(model, tok, by_cls, eval_by_cls, device, args, out_dir,
               *, is_ddp=False, world_size=1, rank=0,
               baseline_func1: float = -1.0):
    # Paper §4.2 / App. D: per-stage epochs, lr, replay, and Func@1 val gates.
    stage_epochs = [int(x) for x in _parse_stage_floats(args.stage_epochs)]
    stage_lrs    = _parse_stage_floats(args.stage_lr)
    stage_replay = _parse_stage_floats(args.stage_replay)
    stage_gates  = _parse_stage_floats(args.stage_val_gate)
    stage_cfg = dict(zip(CURRICULUM_STAGES,
                          zip(stage_epochs, stage_lrs,
                              stage_replay, stage_gates)))
    _rprint(rank, f"[curriculum] per-stage cfg = {stage_cfg}")

    # `model` here may be a DDP wrapper; the underlying PEFT/HF module is
    # `model.module`. Optimizer is built over the wrapper's trainable
    # params either way (DDP exposes them through `.parameters()`); LR is
    # updated per stage by mutating the AdamW param_groups in place.
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=stage_lrs[0], weight_decay=0.01,
    )
    seen = []
    log = []
    step = 0
    # Best Func@1 seen so far — initialized from baseline (computed
    # pre-training on rank 0) so we can detect regression vs. starting
    # point, not just vs. earlier stage.
    best_func1 = baseline_func1 if baseline_func1 > 0 else 0.0
    abort_training = False
    start_stage = getattr(args, "start_stage", "C1")
    started = False
    for cls in CURRICULUM_STAGES:
        ep_n, lr_n, replay_n, gate_n = stage_cfg[cls]
        # Update LR for this stage.
        for pg in opt.param_groups:
            pg["lr"] = lr_n
        current = list(by_cls.get(cls, []))
        if not current:
            _rprint(rank, f"[stage {cls}] no data, skipping")
            continue
        if cls == start_stage:
            started = True
        if not started:
            _rprint(rank, f"[stage {cls}] before start-stage={start_stage}; "
                          f"adding {len(current)} examples to replay buffer "
                          f"and skipping training")
            seen.extend(current)
            continue
        replay_size = int(replay_n * len(current)) if seen else 0
        replay = random.sample(seen, min(len(seen),
                               max(1, replay_size))) if seen else []
        stage = current + replay
        random.shuffle(stage)
        _rprint(rank, f"\n=== Stage {cls}: examples={len(current)} "
                f"replay={len(replay)} (ratio={replay_n}) "
                f"epochs={ep_n} lr={lr_n} val_gate={gate_n} "
                f"total={len(stage)} ===")

        ds = SVADataset(tok, stage, args.max_len, args.alpha)
        if is_ddp:
            sampler = DistributedSampler(
                ds, num_replicas=world_size, rank=rank,
                shuffle=True, seed=args.seed, drop_last=True,
            )
            loader = DataLoader(
                ds, batch_size=args.batch_size, sampler=sampler,
                collate_fn=lambda b: collate(b, tok.pad_token_id or tok.eos_token_id),
            )
        else:
            sampler = None
            loader = DataLoader(
                ds, batch_size=args.batch_size, shuffle=True,
                collate_fn=lambda b: collate(b, tok.pad_token_id or tok.eos_token_id),
            )
        eval_tasks = eval_by_cls.get(cls, [])
        best_metric = -1.0
        bad_evals = 0
        stop_stage = False
        eval_ckpt_dir = out_dir / f"_stage{cls}_eval_ckpt"
        model.train()
        for epoch in range(ep_n):
            if sampler is not None:
                sampler.set_epoch(epoch + step)  # different shuffle per epoch
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(input_ids=batch["input_ids"],
                             attention_mask=batch["attention_mask"])
                loss = temporal_weighted_loss(out.logits.float(),
                                                batch["labels"], batch["weights"])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                log.append({"step": step, "stage": cls, "epoch": epoch,
                             "loss": float(loss.item())})
                if (step % 25 == 0 or step == 1) and _is_main(rank):
                    print(f"  step {step}  stage={cls}  epoch={epoch}  "
                          f"loss={loss.item():.4f}")
                if (args.eval_every_steps > 0 and eval_tasks
                        and step % args.eval_every_steps == 0
                        and _is_main(rank)):
                    _save_for_eval(model, tok, eval_ckpt_dir)
                    report = run_vllm_eval(f"stage{cls}_step{step}", eval_ckpt_dir,
                                           eval_tasks, args)
                    metric = stage_metric(report, cls)
                    improved = metric > (best_metric + args.early_stop_min_delta)
                    if improved:
                        best_metric = metric
                        bad_evals = 0
                    else:
                        bad_evals += 1
                    log.append({
                        "type": "eval",
                        "step": step,
                        "stage": cls,
                        "epoch": epoch,
                        "target_class": cls,
                        "metric_name": "tcl_match_rate",
                        "metric_value": metric,
                        "best_metric": best_metric,
                        "bad_evals": bad_evals,
                        "total": report["total"],
                        "match": report["match"],
                        "syntax": report["syntax"],
                    })
                    print(f"  [eval] stage={cls} step={step} "
                          f"match={metric:.3f} best={best_metric:.3f} "
                          f"gate={gate_n:.3f} bad={bad_evals}/{args.patience}")
                    model.train()
                    # Paper §4.2 val gate: stop as soon as the stage clears
                    # its per-class Func@1 threshold (0.85/0.65/0.50).
                    if gate_n > 0 and metric >= gate_n:
                        print(f"  [val-gate] stage {cls} cleared "
                              f"{gate_n:.2f}; moving on")
                        stop_stage = True
                        break
                    if args.patience > 0 and bad_evals >= args.patience:
                        print(f"  [early-stop] stage {cls} hit patience={args.patience}; "
                              f"stopping current stage")
                        stop_stage = True
                        break
            if stop_stage:
                break
        if is_ddp:
            dist.barrier()
        # Stage-end Func@1 eval (rank 0). Sync the resulting func@1 +
        # auto-stop signal across ranks so DDP gradients don't deadlock.
        stage_func1 = -1.0
        stop_signal = 0
        if args.eval_each_stage and _is_main(rank):
            _save_for_eval(model, tok, eval_ckpt_dir)
            stage_func1 = _run_funcatk_eval(f"stage{cls}_end", eval_ckpt_dir, args)
            improved = stage_func1 > best_func1 + 1e-6
            print(f"  [stage-end-funcatk] {cls}: func@1 = "
                  f"{100*max(stage_func1,0):.2f}%  best = {100*best_func1:.2f}%  "
                  f"{'NEW BEST' if improved else 'no improve'}")
            log.append({
                "type": "stage_end_funcatk",
                "step": step,
                "stage": cls,
                "func1": stage_func1,
                "best_so_far": best_func1,
            })
            if improved:
                # Save best LoRA adapter snapshot
                best_dir = out_dir / "best_func1"
                if best_dir.exists():
                    shutil.rmtree(best_dir, ignore_errors=True)
                shutil.copytree(eval_ckpt_dir, best_dir)
                best_func1 = stage_func1
                print(f"  [best] saved best-func1 ckpt to {best_dir}")
            elif (best_func1 > 0
                  and stage_func1 >= 0
                  and (best_func1 - stage_func1) > args.early_stop_min_delta):
                drop_pp = 100 * (best_func1 - stage_func1)
                print(f"  [early-stop] func@1 dropped {drop_pp:.2f}pp vs best; "
                      f"will stop after {cls}")
                stop_signal = 1
        # Broadcast Func@1 + stop signal so non-rank-0 ranks know to exit.
        if is_ddp:
            t = torch.tensor([stop_signal, int(stage_func1 * 1e6)],
                             device=args.device, dtype=torch.long)
            dist.broadcast(t, src=0)
            stop_signal = int(t[0].item())
            stage_func1 = float(t[1].item()) / 1e6
            dist.barrier()
        if stop_signal:
            abort_training = True
        if _is_main(rank):
            shutil.rmtree(eval_ckpt_dir, ignore_errors=True)
        seen.extend(current)
        if abort_training:
            break
    return log


def _save_for_eval(model_or_ddp, tok, ckpt_dir: Path):
    """Unwrap DDP, save adapter (or full model) + tokenizer."""
    inner = model_or_ddp.module if isinstance(model_or_ddp, DDP) else model_or_ddp
    inner.save_pretrained(ckpt_dir)
    tok.save_pretrained(ckpt_dir)


# --- main ---------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--output-dir", default="results/sft_v2")
    ap.add_argument("--sft-dir", default=str(DEFAULT_SFT_DIR))
    ap.add_argument("--sft-jsonl", default="",
                    help="explicit unified jsonl (overrides --sft-dir lookup)")
    ap.add_argument("--batch-size", type=int, default=4)
    # Paper §4.2 / App. D specifies a per-stage curriculum. The three
    # --stage-* flags below override the single --lr / --epochs-per-stage /
    # --replay-ratio when set. Paper defaults:
    #   C1: 3 epochs, lr 2e-5, replay 0,   val gate 0.85
    #   C2: 5 epochs, lr 1e-5, replay 0.5, val gate 0.65
    #   C3: 6 epochs, lr 8e-6, replay 0.5, val gate 0.50
    ap.add_argument("--epochs-per-stage", type=int, default=3,
                    help="legacy single-stage default; overridden by --stage-epochs")
    ap.add_argument("--lr", type=float, default=2e-5,
                    help="legacy single-stage default; overridden by --stage-lr")
    ap.add_argument("--stage-epochs", default="3,5,6",
                    help="per-stage epoch counts (C1,C2,C3). Paper §4.2 default: 3,5,6")
    ap.add_argument("--stage-lr", default="2e-5,1e-5,8e-6",
                    help="per-stage learning rates (C1,C2,C3). "
                         "Paper §4.2 default: 2e-5,1e-5,8e-6")
    ap.add_argument("--stage-replay", default="0.0,0.5,0.5",
                    help="per-stage replay ratios (C1,C2,C3). "
                         "Paper §4.2 default: 0,0.5,0.5")
    ap.add_argument("--stage-val-gate", default="0.85,0.65,0.50",
                    help="per-stage Func@1 thresholds; stage stops early "
                         "once met. Paper §4.2 default: 0.85,0.65,0.50")
    ap.add_argument("--alpha", type=float, default=3.0)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-each-stage", action="store_true")
    ap.add_argument("--eval-every-steps", type=int, default=500,
                    help="run stage-specific eval every N optimizer steps; 0 disables")
    ap.add_argument("--patience", type=int, default=3,
                    help="stop current stage after this many non-improving evals")
    ap.add_argument("--early-stop-min-delta", type=float, default=0.02,
                    help="Func@1 drop (vs best-so-far) that triggers early "
                         "stop. 0.02 = 2pp drop tolerated, anything beyond stops.")
    ap.add_argument("--eval-max-new", type=int, default=256,
                    help="generation budget for periodic eval")
    ap.add_argument("--eval-prompt-format", default="simple",
                    choices=["simple", "fveval"])
    ap.add_argument("--eval-tasks-human",
                    default=str(REPO_ROOT / "data" / "test" / "nl2sva_human.jsonl"),
                    help="full nl2sva_human jsonl path (used for periodic eval)")
    ap.add_argument("--eval-tasks-machine",
                    default=str(REPO_ROOT / "data" / "test" / "nl2sva_machine.jsonl"),
                    help="full nl2sva_machine jsonl path (used for periodic eval)")
    ap.add_argument("--vllm-eval-gpu", default="",
                    help="CUDA_VISIBLE_DEVICES value for external vLLM eval")
    ap.add_argument("--vllm-tensor-parallel-size", type=int, default=1)
    ap.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.80)
    # Stage-end Func@1 gate (driven by scripts/run_funcatk_eval.py)
    ap.add_argument("--funcatk-eval-gpu", default="0",
                    help="CUDA_VISIBLE_DEVICES for the stage-end Func@1 eval "
                         "subprocess. Shares GPU with rank 0 (rank 0 is paused "
                         "at the DDP barrier during eval).")
    ap.add_argument("--funcatk-num-samples", type=int, default=8,
                    help="vLLM samples per task for Func@1 (8 = stable signal)")
    ap.add_argument("--funcatk-max-model-len", type=int, default=4096,
                    help="vLLM max_model_len for stage-end Func@1 eval; "
                         "must exceed longest fveval prompt (~2.1k tokens)")
    ap.add_argument("--funcatk-gpu-mem-util", type=float, default=0.15,
                    help="vLLM gpu_memory_utilization for the eval subprocess "
                         "when sharing a GPU with rank 0 (~15%% of 140GB ≈ 21GB)")
    ap.add_argument("--gradient-checkpointing", action="store_true")
    # LoRA fine-tune (PEFT) — for memory-constrained "polish" SFT
    ap.add_argument("--use-lora", action="store_true",
                    help="wrap the model with a LoRA adapter (PEFT) instead "
                         "of full-parameter fine-tune. Cuts VRAM ~3x.")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--replay-ratio", type=float, default=0.2,
                    help="replay ratio per stage (0.2 = proposal default, 0.5 = aggressive)")
    ap.add_argument("--start-stage", default="C1",
                    choices=list(CURRICULUM_STAGES),
                    help="skip earlier stages but still feed their examples "
                         "into the replay buffer; train only from this stage on")
    ap.add_argument("--resume-adapter", default="",
                    help="path to a saved LoRA adapter to load instead of "
                         "fresh init; requires --use-lora")
    args = ap.parse_args()

    is_ddp, local_rank, world_size, rank = _ddp_env()
    if is_ddp:
        # 30-min timeout — funcatk eval (vLLM init + 79*N gens + PEC) on
        # rank 0 keeps non-rank-0 ranks at the next collective. Default
        # 10-min cap kills DDP before eval can finish.
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
        torch.cuda.set_device(local_rank)
        args.device = f"cuda:{local_rank}"
        # In-step (every-N) vLLM eval is hard to coordinate across DDP
        # ranks (rank 0 stalls while doing eval; others hit allreduce
        # timeouts). Force in-step eval off; keep stage-end Func@1 eval
        # which uses an explicit barrier+broadcast.
        if args.eval_every_steps != 0:
            _rprint(rank, "[ddp] forcing --eval-every-steps 0 "
                          "(stage-end Func@1 eval still active)")
        args.eval_every_steps = 0

    random.seed(args.seed + rank); torch.manual_seed(args.seed + rank)
    out = Path(args.output_dir)
    if _is_main(rank):
        out.mkdir(parents=True, exist_ok=True)
    if is_ddp:
        dist.barrier()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    explicit = Path(args.sft_jsonl) if args.sft_jsonl else None
    by_cls = load_sft_by_class(Path(args.sft_dir), explicit_jsonl=explicit)
    tasks = load_test_set()
    eval_by_cls = bucket_tasks_by_class(tasks)

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
             "fp32": torch.float32}[args.dtype]
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(args.device)
    if args.gradient_checkpointing:
        # use_reentrant=False is DDP-compatible (the reentrant variant
        # double-marks frozen LoRA-base params and explodes under DDP).
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    if args.use_lora:
        from peft import LoraConfig, get_peft_model, PeftModel
        if args.resume_adapter:
            model = PeftModel.from_pretrained(
                model, args.resume_adapter, is_trainable=True,
            )
            if _is_main(rank):
                print(f"[lora] resumed adapter from {args.resume_adapter}")
                model.print_trainable_parameters()
        else:
            lora_cfg = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
            )
            model = get_peft_model(model, lora_cfg)
            if _is_main(rank):
                model.print_trainable_parameters()
                print(f"[lora] r={args.lora_r}, alpha={args.lora_alpha}, "
                      f"dropout={args.lora_dropout}")
    _rprint(rank, f"Model loaded: {args.model}")

    if is_ddp:
        # PEFT freezes base params, so DDP needs find_unused_parameters=True.
        model = DDP(model, device_ids=[local_rank],
                    output_device=local_rank,
                    find_unused_parameters=True)
        _rprint(rank, f"[ddp] wrapped model on cuda:{local_rank}, "
                      f"world_size={world_size}")

    if _is_main(rank):
        print("\n--- Baseline eval (pre-SFT) ---")
        baseline_model = model.module if isinstance(model, DDP) else model
        baseline = eval_on_test("pre", tok, baseline_model, tasks, args.device)
    else:
        baseline = None
    if is_ddp:
        dist.barrier()

    # Optional pre-training Func@1 baseline (rank 0 only, broadcast).
    baseline_func1 = -1.0
    if args.eval_each_stage and _is_main(rank):
        print("\n--- Baseline Func@1 eval ---")
        # Save base/adapter snapshot to a temp ckpt for the eval subprocess
        baseline_ckpt = out / "_baseline_eval_ckpt"
        _save_for_eval(model, tok, baseline_ckpt)
        baseline_func1 = _run_funcatk_eval("baseline", baseline_ckpt, args)
        print(f"[baseline] Func@1 = {100*max(baseline_func1,0):.2f}%")
        shutil.rmtree(baseline_ckpt, ignore_errors=True)
    if is_ddp:
        t = torch.tensor([int(baseline_func1 * 1e6)],
                         device=args.device, dtype=torch.long)
        dist.broadcast(t, src=0)
        baseline_func1 = float(t[0].item()) / 1e6
        dist.barrier()

    log = curriculum(model, tok, by_cls, eval_by_cls, args.device, args, out,
                     is_ddp=is_ddp, world_size=world_size, rank=rank,
                     baseline_func1=baseline_func1)

    if is_ddp:
        dist.barrier()
    if _is_main(rank):
        print("\n--- Post-SFT eval ---")
        final_model = model.module if isinstance(model, DDP) else model
        final = eval_on_test("post", tok, final_model, tasks, args.device)

        # Save
        report = {"args": vars(args), "baseline": baseline, "final": final,
                  "train_loss": log}
        with open(out / f"sft_{ts}.json", "w") as f:
            json.dump(report, f, indent=2, default=str)
        save_dir = out / f"checkpoint_{ts}"
        inner = model.module if isinstance(model, DDP) else model
        inner.save_pretrained(save_dir)
        tok.save_pretrained(save_dir)
        print(f"\nSaved: {out}/sft_{ts}.json  +  {save_dir}")

    if is_ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
