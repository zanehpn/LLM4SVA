#!/usr/bin/env python3
"""
fill_nl_with_vllm.py — vLLM version of fill_nl_with_llm.py.

Used for models (e.g. DeepSeek-Coder-V2) whose custom modeling code is
incompatible with the latest transformers HF generate path. vLLM has an
independent inference engine that bypasses the issue.

Usage:
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python3 scripts/fill_nl_with_vllm.py \\
      --model /ssd2/junyi/.../DeepSeek-Coder-V2-Lite-Instruct/snapshots/<sha>/ \\
      --in-path data/train/sft/wave3_new_nl_fill.jsonl \\
      --out data/train/sft/wave3_new_nl_filled_dscoder.jsonl
"""
import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent
SFT_DIR = EXPERIMENTS_DIR / "data" / "train" / "sft"

PLACEHOLDER_RE = re.compile(r"^\s*\[(?:ASSERT|ASSUME|COVER)[^\]]*\]\s*$",
                            re.IGNORECASE)


def needs_nl(nl: str) -> bool:
    nl = (nl or "").strip()
    return len(nl) < 5 or PLACEHOLDER_RE.match(nl)


SYSTEM_PROMPT = (
    "You are an expert SystemVerilog hardware verification engineer. "
    "Given a SystemVerilog module and a SystemVerilog Assertion (SVA) over its "
    "signals, write a one-sentence natural-language description of what the "
    "assertion CHECKS, written as a fragment in the style of:\n"
    "    'that the counter does not overflow. Use the signals X, Y, Z.'\n"
    "    'when req is asserted, gnt must follow within 3 cycles. Use signals "
    "req, gnt.'\n"
    "Output ONLY the fragment — no preamble, no markdown, no explanation, no "
    "quoting. Start with 'that' or 'when' or 'whenever'. End with one sentence "
    "naming the relevant signals. Pay special attention:\n"
    "  - 'disable iff (X)' means: STOP evaluating when X is true. Phrase as "
    "'unless X' or 'when X is false', NOT 'X is disabled'.\n"
    "  - 'cover property' means: at least one execution should reach this "
    "state. Phrase as 'cover the case where ...' not 'X implies Y'.\n"
    "  - 'A ##N B' means: A is true at time t AND B is true at time t+N. "
    "It is NOT 'A implies B'."
)


def build_user_prompt(rtl_context: str, reference_sva: str,
                      max_rtl_chars: int = 2000) -> str:
    rtl_context = (rtl_context or "").strip()
    sva = reference_sva.strip()[:6000]   # also cap SVA in case it's massive
    if rtl_context:
        rtl_context = rtl_context[:max_rtl_chars]
        return (
            "Module:\n"
            f"```systemverilog\n{rtl_context}\n```\n\n"
            "Assertion:\n"
            f"```systemverilog\n{sva}\n```\n\n"
            "Write the one-sentence NL description fragment now."
        )
    return (
        "Assertion (no module context available):\n"
        f"```systemverilog\n{sva}\n```\n\n"
        "Write the one-sentence NL description fragment now."
    )


def cleanup(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"^(?:NL|Description|Answer|Output)\s*[:：]\s*", "", text,
                  flags=re.IGNORECASE)
    text = text.strip().strip('"').strip("'").strip("`").strip()
    text = text.split("\n")[0].strip()
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--in-path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=64,
                    help="vLLM internal batch size for parallelism")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    with open(args.in_path) as f:
        for line in f:
            records.append(json.loads(line))
    n_total = len(records)
    missing_idx = [i for i, r in enumerate(records) if needs_nl(r.get("nl", ""))]
    if args.limit:
        missing_idx = missing_idx[:args.limit]
    print(f"[fill-nl-vllm] total={n_total}  missing={len(missing_idx)}")

    print(f"[fill-nl-vllm] loading model {args.model}")
    from vllm import LLM, SamplingParams
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    tok = llm.get_tokenizer()
    print("[fill-nl-vllm] model loaded")

    sp = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_new_tokens,
    )

    # Build prompts in batch
    prompts = []
    for idx in missing_idx:
        r = records[idx]
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",
             "content": build_user_prompt(r.get("rtl_context", ""),
                                          r.get("reference_sva", ""))},
        ]
        try:
            prompt = tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt = tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)
    print(f"[fill-nl-vllm] prompts built: {len(prompts)}")

    t0 = time.time()
    outputs = llm.generate(prompts, sp)
    elapsed = time.time() - t0
    print(f"[fill-nl-vllm] inference done in {elapsed:.0f}s "
          f"({elapsed/len(prompts):.2f}s/sample)")

    by_source = Counter()
    for k, idx in enumerate(missing_idx):
        nl = cleanup(outputs[k].outputs[0].text)
        records[idx]["nl"] = nl
        records[idx]["nl_filled_by"] = Path(args.model).name
        by_source[records[idx].get("source", "?")] += 1

    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"\n[fill-nl-vllm] wrote {n_total} records → {out_path}")
    print(f"[fill-nl-vllm] by_source={dict(by_source)}")


if __name__ == "__main__":
    main()
