"""split_codev_83k_with_reasoning.py
Parse CodeV-SVA-dataset-training-83K.jsonl (chat-format, with <think>
reasoning + SVA in the assistant turn) into the same schema as
data/train/{sft,grpo}/ and split body-hash-disjoint between SFT and GRPO.

Output:
  data/train/sft_with_reasoning/sft_train.jsonl
  data/train/grpo_with_reasoning/grpo_pool.jsonl
  data/train/{sft,grpo}_with_reasoning/manifest.json

Schema per row:
  id, source, nl, rtl_context, reference_sva, expected_tcl, hash,
  reasoning, assistant_full   (← new: full <think>...</think>SVA block)
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Reuse TCL classifier
import importlib.util
_g_spec = importlib.util.spec_from_file_location(
    "_g", str(ROOT / "src" / "ast_checker.py"))
try:
    _g = importlib.util.module_from_spec(_g_spec); _g_spec.loader.exec_module(_g)
    _classify = getattr(_g, "classify_tcl", None) or getattr(_g, "tcl_of", None)
except Exception:
    _classify = None


# ------- TCL fallback classifier (regex on operator presence) ----------
_LIVENESS_RE = re.compile(r"\b(s_eventually|s_until|s_always|until_with|nexttime)\b")
_TEMPORAL_RE = re.compile(r"\b(throughout|within|intersect)\b|\#\#|\[\*|\|->|\|=>")
def fallback_classify(sva: str) -> int:
    if _LIVENESS_RE.search(sva or ""):
        return 5
    if _TEMPORAL_RE.search(sva or ""):
        # Distinguish L4 (|->) vs L2 (##N) vs L3 (##[a:b])
        if "|->" in sva or "|=>" in sva:
            return 4
        if re.search(r"\#\#\s*\[", sva):
            return 3
        return 2
    return 1


def tcl_classify(sva: str) -> int:
    if _classify is not None:
        try:
            return int(_classify(sva))
        except Exception:
            pass
    return fallback_classify(sva)


# ------- parsing helpers -----------------------------------------------
_SVA_BLOCK_RE = re.compile(r"```(?:systemverilog|sv|verilog)?\s*\n?(.+?)```",
                            re.DOTALL | re.IGNORECASE)
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_RTL_RE = re.compile(r"Here is the testbench[^\n]*\n(.*?)\nQuestion:",
                      re.DOTALL)
_QUESTION_RE = re.compile(r"Question:\s*(.*?)(?:\nYou should use|\Z)",
                           re.DOTALL)


def parse_row(row: dict, idx: int) -> dict | None:
    """Extract structured fields from a chat-format row."""
    msgs = row.get("messages", [])
    user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
    assistant_msg = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
    if not user_msg or not assistant_msg:
        return None

    # Extract RTL context (between "Here is the testbench" and "Question:")
    rtl_m = _RTL_RE.search(user_msg)
    rtl_context = rtl_m.group(1).strip() if rtl_m else ""

    # Extract NL (after "Question:" up to "You should use" or end)
    nl_m = _QUESTION_RE.search(user_msg)
    nl = nl_m.group(1).strip() if nl_m else ""

    # Extract SVA from assistant's ```systemverilog block
    sva_m = _SVA_BLOCK_RE.search(assistant_msg)
    if not sva_m:
        return None
    sva = sva_m.group(1).strip()

    # Extract <think> reasoning
    think_m = _THINK_RE.search(assistant_msg)
    reasoning = think_m.group(1).strip() if think_m else ""

    # Body hash: normalize whitespace and lowercase keywords for dedup
    body_norm = re.sub(r"\s+", " ", sva.lower()).strip()
    body_hash = hashlib.sha256(body_norm.encode()).hexdigest()[:16]

    return {
        "id": f"codev_83k_{idx}",
        "source": "codev_sva_83k",
        "nl": nl,
        "rtl_context": rtl_context,
        "reference_sva": sva,
        "expected_tcl": tcl_classify(sva),
        "hash": body_hash,
        "reasoning": reasoning,
        "assistant_full": assistant_msg,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="path to CodeV-SVA-dataset-training-83K.jsonl")
    ap.add_argument("--out-sft", required=True)
    ap.add_argument("--out-grpo", required=True)
    ap.add_argument("--out-manifest", required=True)
    ap.add_argument("--sft-frac", type=float, default=0.18,
                    help="fraction of data to use for SFT (default 0.18, "
                         "matches data/train manifest's ~18%)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-dedup", action="store_true",
                    help="skip body-hash dedup (keep all rows including "
                         "exact duplicates)")
    args = ap.parse_args()

    # 1. Parse all rows
    rows = []
    skipped_no_sva = 0
    with open(args.input) as f:
        for i, ln in enumerate(f):
            try:
                r = json.loads(ln)
            except Exception:
                continue
            parsed = parse_row(r, i)
            if parsed is None:
                skipped_no_sva += 1
                continue
            rows.append(parsed)
    print(f"[parse] {len(rows)} rows parsed  (skipped {skipped_no_sva} no-SVA)")

    # 2. Dedup by body hash (optional)
    if args.no_dedup:
        deduped = rows
        print(f"[dedup] SKIPPED (--no-dedup); keeping all {len(rows)} rows")
    else:
        seen = set()
        deduped = []
        for r in rows:
            if r["hash"] in seen:
                continue
            seen.add(r["hash"])
            deduped.append(r)
        print(f"[dedup] {len(deduped)} unique by body hash  "
              f"(removed {len(rows) - len(deduped)} duplicates)")

    # 3. Body-hash disjoint split: shuffle, take first sft_frac as SFT
    random.seed(args.seed)
    random.shuffle(deduped)
    n_total = len(deduped)
    n_sft = int(n_total * args.sft_frac)
    sft_pool = deduped[:n_sft]
    grpo_pool = deduped[n_sft:]
    for r in sft_pool: r["split"] = "sft"
    for r in grpo_pool: r["split"] = "grpo"

    # 4. Sanity: hashes disjoint only when dedup was applied
    if not args.no_dedup:
        sft_hashes = {r["hash"] for r in sft_pool}
        grpo_hashes = {r["hash"] for r in grpo_pool}
        assert sft_hashes.isdisjoint(grpo_hashes), "split overlap!"

    # 5. Per-TCL counts
    sft_tcl = Counter(r["expected_tcl"] for r in sft_pool)
    grpo_tcl = Counter(r["expected_tcl"] for r in grpo_pool)
    print(f"[split] sft={len(sft_pool)} grpo={len(grpo_pool)}")
    print(f"  sft per_tcl: {dict(sorted(sft_tcl.items()))}")
    print(f"  grpo per_tcl: {dict(sorted(grpo_tcl.items()))}")

    # 6. Avg reasoning length
    avg_reason_chars = sum(len(r["reasoning"]) for r in deduped) / max(len(deduped), 1)
    avg_reason_words = sum(len(r["reasoning"].split()) for r in deduped) / max(len(deduped), 1)
    print(f"[reasoning] avg chars={avg_reason_chars:.0f}  avg words={avg_reason_words:.0f}")

    # 7. Write
    Path(args.out_sft).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_grpo).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_sft, "w") as f:
        for r in sft_pool:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(args.out_grpo, "w") as f:
        for r in grpo_pool:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    manifest = {
        "policy": "Body-hash disjoint split of CodeV-SVA-83K (with <think> "
                  "reasoning preserved). SFT and GRPO pools share no SVA "
                  "by body hash. Train models that emit <think> reasoning "
                  "before SVA to match the data format.",
        "input": args.input,
        "input_rows": len(rows),
        "deduped_rows": len(deduped),
        "skipped_no_sva": skipped_no_sva,
        "sft_size": len(sft_pool),
        "grpo_size": len(grpo_pool),
        "sft_per_tcl": dict(sorted(sft_tcl.items())),
        "grpo_per_tcl": dict(sorted(grpo_tcl.items())),
        "avg_reasoning_chars": round(avg_reason_chars, 1),
        "avg_reasoning_words": round(avg_reason_words, 1),
        "sft_path": args.out_sft,
        "grpo_path": args.out_grpo,
        "intersection_check":
            "PASS (body hashes disjoint by construction)" if not args.no_dedup
            else "SKIPPED (--no-dedup; duplicates preserved)",
    }
    Path(args.out_manifest).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_manifest, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[write] {args.out_sft}")
    print(f"[write] {args.out_grpo}")
    print(f"[write] {args.out_manifest}")


if __name__ == "__main__":
    main()
