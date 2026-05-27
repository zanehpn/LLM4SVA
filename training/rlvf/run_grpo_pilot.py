#!/usr/bin/env python3
"""
run_grpo_pilot.py — GRPO-RLVF pilot on our Qwen-Coder-7B SFT checkpoint.

Goal: validate Component 3 can correct the L1 overshoot the pure SFT run
introduced (post-SFT L1 match dropped from 79% → 18%).

Reward function (simplified for pilot; formal+vacuity deferred):
  R = w_syn * syntax_ok(sva)
    + w_sim * normalized_ast_sim(sva, ref_sva)
  with w_syn = 0.40, w_sim = 0.60 (re-scaled from proposal's 0.15+0.25=0.40).

Pilot hyperparameters (smaller than proposal Table):
  G = 4 rollouts / prompt (proposal: 8)
  total_steps = 200   (proposal: 8000)
  clip_range ε = 0.2  (standard)
  kl_coeff β = 0.04   (proposal)
  lr = 1e-6           (proposal)

Post-GRPO eval runs on NL2SVA-Human 79 same as SFT eval.

Usage:
  CUDA_VISIBLE_DEVICES=3 python scripts/run_grpo_pilot.py \\
      --policy results/sft_qwen_coder_7b/checkpoint_20260420_201925 \\
      --output-dir results/grpo_pilot_1
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from src.mock_verifier import syntax_check

VLLM_EVAL_SCRIPT = REPO_ROOT / "eval" / "run_eval_nl2sva_human_vllm.py"
FUNCATK_EVAL_SCRIPT = REPO_ROOT / "eval" / "run_funcatk_eval.py"
GRPO_POOL_DEFAULT = REPO_ROOT / "data" / "train" / "grpo" / "grpo_pool_tiered.jsonl"
GRPO_POOL_FROM_SFT = REPO_ROOT / "data" / "train" / "grpo" / "grpo_pool_from_sft.jsonl"
GRPO_POOL_PHASE2 = REPO_ROOT / "data" / "train" / "grpo" / "grpo_pool_phase2.jsonl"

SYSTEM_PROMPT = (
    "You are an expert in SystemVerilog Assertions (SVA). Given a natural-"
    "language description of a design property, output ONE syntactically "
    "correct SVA assertion. Emit ONLY the SVA — no explanation. Match "
    "temporal complexity: bare `##N` for fixed delays, `##[a:b]` for ranged, "
    "`|->`/`|=>` only when antecedent-consequent, `s_eventually`/`s_until` "
    "for liveness."
)

# Eval-matched (FVEval-style) prompt: must mirror run_eval_nl2sva_human.py so
# GRPO trains on the same surface form the model sees at test time. Closes
# the train/eval prompt-format distribution gap that left earlier GRPO pilots
# stuck at the SFT ceiling.
FVEVAL_SYSTEM_PROMPT = (
    "You are an AI assistant tasked with formal verification of register "
    "transfer level (RTL) designs. Your job is to translate a description "
    "of an assertion into a concrete SystemVerilog Assertion (SVA) "
    "implementation. Match temporal complexity: bare `##N` for fixed "
    "delays, `##[a:b]` for ranged delays, `|->` / `|=>` for "
    "antecedent-consequent implication, `s_eventually` / `s_until` for "
    "liveness."
)
FVEVAL_USER_POSTAMBLE = (
    "Do not add code to output an error message string.\n"
    "Enclose your SVA code with ```systemverilog and ```. "
    "Only output the code snippet and do NOT output anything else.\n\n"
    "For example,\n"
    "```systemverilog\n"
    "asrt: assert property (@(posedge clk) disable iff (tb_reset)\n"
    "    (a && b) != 1'b1\n"
    ");\n"
    "```\n"
    "Answer:"
)


FVEVAL_RTL_CAP_CHARS = 4500  # Set to match the longest RTL in the
                              # CodeV-curated grpo pool (max 4171 chars);
                              # the cap is a guard against accidentally
                              # using pools with industrial-RTL outliers
                              # that blow up SDPA. The CodeV pool's distri-
                              # bution matches NL2SVA-Machine eval, so no
                              # row gets truncated when running on it.
                              # Run_funcatk_eval has no cap, so train and
                              # eval prompts are now identical for any
                              # pool with max RTL <= this value.


def build_fveval_user_prompt(nl: str, rtl_context: str,
                             rtl_cap: int = FVEVAL_RTL_CAP_CHARS) -> str:
    parts = []
    rtl = (rtl_context or "").strip()
    if rtl:
        if rtl_cap and len(rtl) > rtl_cap:
            rtl = rtl[:rtl_cap] + "\n  // ... [truncated]"
        parts.append(
            "Here is the testbench to perform your translation:\n"
            f"{rtl}"
        )
    nl = nl.strip()
    if not re.match(
        r"(?i)^\s*(create|generate|write)\s+(an?\s+|the\s+)?sva\b", nl
    ):
        nl = f"Create a SVA assertion that checks: {nl}"
    parts.append(f"Question: {nl}")
    parts.append(FVEVAL_USER_POSTAMBLE)
    return "\n\n".join(parts)


# --- reward components ------------------------------------------------------

def extract_sva(text: str) -> str:
    text = (text or "").strip()
    # strip <think>...</think>
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "<think>" in text:
        text = text.split("<think>")[0]
    m = re.search(r"```(?:systemverilog|sv|verilog)?\s*(.+?)```",
                  text, re.DOTALL)
    if m: text = m.group(1).strip()
    m = re.search(r"(assert\s+property\s*\(.*?\)\s*;)",
                  text, re.DOTALL | re.IGNORECASE)
    if m: return m.group(1).strip()
    return text.splitlines()[0].strip() if text else ""


def _tokenize_sva(sva: str):
    """Simple SV-ish tokenizer: split on non-word but keep operators together."""
    sva = re.sub(r"\s+", " ", sva).strip()
    return re.findall(r"[A-Za-z_]\w*|##\[[^\]]*\]|##\d+|\|->|\|=>|[()\[\];,.]|\S", sva)


def _levenshtein(a, b):
    """Token-level edit distance."""
    n, m = len(a), len(b)
    if n == 0: return m
    if m == 0: return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            tmp = dp[j]
            dp[j] = prev if a[i - 1] == b[j - 1] else 1 + min(dp[j], dp[j - 1], prev)
            prev = tmp
    return dp[m]


def ast_sim(gen_sva: str, ref_sva: str) -> float:
    if not gen_sva or not ref_sva:
        return 0.0
    a = _tokenize_sva(gen_sva); b = _tokenize_sva(ref_sva)
    if not a or not b: return 0.0
    d = _levenshtein(a, b)
    return max(0.0, 1.0 - d / max(len(a), len(b)))


def compute_reward_astdiff(completion: str, reference_sva: str,
                           rtl_context: str = "") -> float:
    """Original reward: weighted syntax + AST edit similarity."""
    sva = extract_sva(completion)
    syn = 1.0 if syntax_check(sva)["ok"] else 0.0
    sim = ast_sim(sva, reference_sva)
    return 0.4 * syn + 0.6 * sim


# PEC reward — lazy import so non-PEC pipelines don't pay startup cost.
_PEC = None
_INFER_RESET = None
_FREE_RTL_CACHE = {}

# Cadence-aligned canonicalization for the training reward. Without these,
# the reward side treats `disable iff (tb_reset)` asymmetry between ref
# and generated candidate as a genuine directional implication, zeroing
# out most of the training signal on Machine-style prompts (40% of
# candidates would be IMPLIES_REF_TO_LM under strict LRM; eval side
# already applies the same canonicalization).
# Paper §4.3 / App. E: open PEC abstains (UNSUPPORTED) on liveness so the
# filter stays sound. Setting this to a positive cycle bound trades soundness
# for coverage and was used in the v6 reward ablation only — see App. F.
PEC_REWARD_LIVENESS_BOUND = None


def _free_input_rtl(reference_sva: str) -> str:
    """Build a minimal synthetic module declaring every identifier in the
    reference SVA as a free input. Cached by SVA hash to avoid recomputation."""
    key = hash(reference_sva)
    if key in _FREE_RTL_CACHE:
        return _FREE_RTL_CACHE[key]
    # Extract identifiers — exclude SV keywords/operators/numbers
    KEYWORDS = {
        "assert", "property", "posedge", "negedge", "disable", "iff", "if",
        "else", "begin", "end", "always", "always_ff", "always_comb", "logic",
        "wire", "reg", "module", "endmodule", "input", "output", "and", "or",
        "not", "throughout", "within", "intersect", "first_match", "until",
        "until_with", "s_until", "s_eventually", "s_always", "nexttime",
        "strong", "weak", "1", "0", "1'b0", "1'b1", "0'b1", "0'b0",
        "rose", "fell", "stable", "past", "changed", "sampled", "onehot",
        "onehot0", "countones", "isunknown", "isknown",
    }
    ids = set()
    for tok in re.findall(r"[A-Za-z_]\w*", reference_sva):
        if tok.lower() in KEYWORDS:
            continue
        if tok in ("clk", "tb_reset"):
            continue   # we'll declare these explicitly
        ids.add(tok)
    decls = ["    input logic clk", "    input logic tb_reset"]
    for ident in sorted(ids):
        decls.append(f"    input logic [31:0] {ident}")
    rtl = "module pec_top (\n" + ",\n".join(decls) + "\n);\nendmodule\n"
    _FREE_RTL_CACHE[key] = rtl
    return rtl


def compute_reward_pec(completion: str, reference_sva: str,
                       rtl_context: str = "") -> float:
    """PEC equivalence reward — paper Eq. 4.

    Reward table (§4.4 / Eq. 4):
      EQUIVALENT              → 1.00
      IMPLIES_REF_TO_LM       → 0.60
      IMPLIES_LM_TO_REF       → 0.40
      UNSUPPORTED + syntax_ok → 0.15
      everything else         → 0.00

    The IMPLIES_REF_TO_LM > IMPLIES_LM_TO_REF asymmetry follows paper §4.4:
    a strictly stricter LM is closer to specification intent than a strictly
    more permissive one. The 0.15 syntax-floor lets liveness rollouts that
    the open PEC cannot decide (UNSUPPORTED) still receive a tiny credit so
    GRPO advantage estimation does not see identically-zero rewards across a
    C3 group; it sits well below any verified equivalence.

    Historical alternative shapes (v3/v4 asymmetric, v5 binary, v6 symmetric
    0.5/0.5) remain selectable via the `--reward-shape` CLI flag for the
    Appendix F sensitivity study.
    """
    global _PEC, _INFER_RESET
    if _PEC is None:
        from src.pec_yosys import prop_equivalence, infer_reset_expr
        _PEC = prop_equivalence
        _INFER_RESET = infer_reset_expr
    sva = extract_sva(completion)
    if not sva:
        return 0.0
    if not syntax_check(sva)["ok"]:
        return 0.0
    rtl = rtl_context.strip() or _free_input_rtl(reference_sva)
    reset_expr = _INFER_RESET(rtl, reference_sva, sva)
    try:
        r = _PEC(sva, reference_sva, rtl, depth=10, timeout=15,
                 reset_expr=reset_expr,
                 liveness_bound=PEC_REWARD_LIVENESS_BOUND)
    except Exception:
        return 0.05
    # Paper Eq. 4 — asymmetric IMPLIES weights with a small UNSUPPORTED
    # syntax-floor so liveness-only groups still have non-zero advantage
    # signal. The open PEC collapses PARSE_ERROR / TIMEOUT / EXTRACT_ERROR /
    # UNKNOWN onto UNSUPPORTED (paper §4.3), so r.verdict here is one of the
    # five paper-facing verdicts.
    syntax_ok = syntax_check(sva)["ok"] if sva else False
    return {
        "EQUIVALENT":         1.00,
        "IMPLIES_REF_TO_LM":  0.60,
        "IMPLIES_LM_TO_REF":  0.40,
        "NOT_EQUIVALENT":     0.00,
        "UNSUPPORTED":        0.15 if syntax_ok else 0.0,
    }.get(r.verdict, 0.0)


def compute_reward_hybrid(completion: str, reference_sva: str,
                          rtl_context: str = "") -> float:
    """Hybrid: 0.20 * syntax + 0.30 * AST sim + 0.50 * PEC verdict."""
    sva = extract_sva(completion)
    syn = 1.0 if syntax_check(sva)["ok"] else 0.0
    sim = ast_sim(sva, reference_sva)
    pec = compute_reward_pec(completion, reference_sva, rtl_context)
    return 0.20 * syn + 0.30 * sim + 0.50 * pec


def compute_reward_pec_multiref(completion: str, ref_svas, rtl_context: str = ""):
    """Phase 2 reward: max(PEC(gen, ref_i)) over a list of equivalent
    references. Each ref_i was already PEC-verified equivalent (or
    LM-implies-REF, i.e. stronger) to the canonical ref at pool-build
    time, so awarding 1.0 if gen matches any of them is monotonic in
    correctness — gen must be at least as strong as the canonical ref.

    Verdict mapping mirrors v5 (no IMPLIES partial credit). Once any ref
    yields 1.0 we short-circuit; otherwise return the max (which is 0.0
    in v5 if no ref matched)."""
    global _PEC, _INFER_RESET
    if _PEC is None:
        from src.pec_yosys import prop_equivalence, infer_reset_expr
        _PEC = prop_equivalence
        _INFER_RESET = infer_reset_expr
    sva = extract_sva(completion)
    if not sva:
        return 0.0
    if not syntax_check(sva)["ok"]:
        return 0.0
    if isinstance(ref_svas, str):
        ref_svas = [ref_svas]
    if not ref_svas:
        return 0.0
    rtl = rtl_context.strip() or _free_input_rtl(ref_svas[0])
    reset_expr = _INFER_RESET(rtl, ref_svas[0], sva)
    best = 0.0
    for ref in ref_svas:
        try:
            r = _PEC(sva, ref, rtl, depth=10, timeout=15,
                     reset_expr=reset_expr,
                     liveness_bound=PEC_REWARD_LIVENESS_BOUND)
        except Exception:
            continue
        score = {
            # Paper Eq. 4 — see compute_reward_pec docstring above.
            "EQUIVALENT":         1.00,
            "IMPLIES_REF_TO_LM":  0.60,
            "IMPLIES_LM_TO_REF":  0.40,
            "NOT_EQUIVALENT":     0.00,
            "UNSUPPORTED":        0.15 if syntax_check(sva)["ok"] else 0.0,
        }.get(r.verdict, 0.0)
        if score >= 1.0:
            return score
        if score > best:
            best = score
    return best


# ---------------------------------------------------------------------------
# Verilator + PEC tiered reward
# ---------------------------------------------------------------------------
# Verilator (`verilator --lint-only --assert`) is much more permissive than
# yosys-slang on real industrial RTL (handles `bind`, hierarchical paths,
# packages, macros via `+define+` / `-I`). Using it as a *gate* before PEC
# achieves two things:
#   1. Rewards are non-zero on industrial SVAs that PEC can't even parse.
#   2. PEC time isn't wasted on syntactically broken candidates.
_SV_KEYWORDS_FOR_FREE_RTL = {
    "assert", "assume", "cover", "property", "endproperty", "sequence",
    "endsequence", "posedge", "negedge", "disable", "iff", "if", "else",
    "begin", "end", "always", "always_ff", "always_comb", "always_latch",
    "logic", "wire", "reg", "module", "endmodule", "input", "output",
    "inout", "bind", "and", "or", "not", "xor", "nand", "nor", "xnor",
    "throughout", "within", "intersect", "first_match", "until",
    "until_with", "s_until", "s_until_with", "s_eventually", "s_always",
    "nexttime", "strong", "weak", "rose", "fell", "stable", "past",
    "changed", "sampled", "onehot", "onehot0", "countones", "isunknown",
    "isknown", "signed", "unsigned",
}


_SVA_LABEL_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*:\s*", re.MULTILINE)


def _strip_sva_label(sva: str) -> tuple[str, str]:
    """Remove a leading `<label>:` from `assert property (...)` style SVAs
    and return (stripped_sva, label_or_empty)."""
    s = (sva or "").lstrip()
    m = re.match(r"([A-Za-z_]\w*)\s*:\s*(?=(?:assert|assume|cover|property))",
                 s, re.IGNORECASE)
    if m:
        return s[m.end():].lstrip(), m.group(1)
    return s, ""


def _identifiers_in_sva(sva: str, exclude: set | None = None) -> set:
    """Collect identifiers in an SVA, excluding SV keywords, numeric
    literals, the label (if any), and a custom exclude set (e.g. clk,
    tb_reset which we always declare explicitly)."""
    if not sva:
        return set()
    body, _label = _strip_sva_label(sva)
    out = set()
    for tok in re.findall(r"[A-Za-z_]\w*", body):
        if tok.lower() in _SV_KEYWORDS_FOR_FREE_RTL:
            continue
        if exclude and tok in exclude:
            continue
        out.add(tok)
    return out


_FUNC_CALL_RE = re.compile(r"\b([A-Za-z_]\w*)\s*\(")
# Clock identifiers used in `@(posedge X)` / `@(negedge X)` / `@(edge X)`
# / `@(X)` — must stay as wires, not parameters.
_CLOCK_ID_RE = re.compile(
    r"@\s*\(\s*(?:posedge|negedge|edge)?\s*([A-Za-z_]\w*)")
# Identifiers used as a temporal delay count: `##IDENT`. Must be a
# constant (parameter), not a wire — fixes Error-[SVA-INCE].
_DELAY_ID_RE = re.compile(r"##\s*([A-Za-z_]\w*)")
# Multi-dimensional array access: `IDENT[a][b]` or `IDENT[a][b][c]`.
# VCS Error-[IBMDA] fires when wrapper declares as packed `[31:0]` only.
_MDA_RE = re.compile(r"\b([A-Za-z_]\w*)\s*\[[^\]]+\]\s*\[[^\]]+\]")
_SV_BUILTIN_FUNCS = {
    # Sampled-value functions
    "past", "rose", "fell", "stable", "changed", "sampled",
    # Boolean / count functions
    "onehot", "onehot0", "countones", "isunknown", "isknown",
    "signed", "unsigned", "bits", "left", "right", "high", "low",
    "size", "increment", "dimensions", "unpacked_dimensions",
    "clog2", "ln", "log10", "exp", "sqrt", "pow", "floor", "ceil",
    "abs", "min", "max",
    # Display / control
    "display", "write", "monitor", "error", "warning", "info", "fatal",
    "time", "stime", "realtime", "random", "urandom", "urandom_range",
    # SVA helpers
    "assert", "assume", "cover", "property", "sequence",
    "posedge", "negedge", "if", "else",
}


def _classify_identifiers(ids: set, sva_text: str) -> tuple[set, set, set]:
    """Bucket identifiers into (params, funcs, ports):
      params  — ALL_CAPS (≥3 chars), used as compile-time constants
                (RADIX, MULT_DLY, C_AXI_ADDR_WIDTH); fixes NCE / IRIPS
      funcs   — appear as `IDENT(` and aren't SV builtins; declared as
                stub functions returning 32'd0 (fixes INF / INT)
      ports   — everything else; declared as `input logic [31:0]`
    """
    # Function-call identifiers
    funcs = set()
    for m in _FUNC_CALL_RE.finditer(sva_text or ""):
        name = m.group(1)
        if name.lower() in _SV_KEYWORDS_FOR_FREE_RTL:
            continue
        if name.lower() in _SV_BUILTIN_FUNCS:
            continue
        if name in ids:
            funcs.add(name)
    # Clock-signal identifiers — must stay wires (parameter as a clock is
    # `Error-[ICE] Invalid clocking expression`).
    clocks = set()
    for m in _CLOCK_ID_RE.finditer(sva_text or ""):
        name = m.group(1)
        if name in ids:
            clocks.add(name)
    # `##IDENT` delay-count identifiers — must be parameter (constant).
    delays = set()
    for m in _DELAY_ID_RE.finditer(sva_text or ""):
        name = m.group(1)
        if name in ids and name not in clocks:
            delays.add(name)
    # MDA accesses (IDENT[a][b]+) — port must be 2D packed array.
    mdas = set()
    for m in _MDA_RE.finditer(sva_text or ""):
        name = m.group(1)
        if name in ids and name not in clocks and name not in funcs:
            mdas.add(name)
    # ALL_CAPS-style + delay-bound parameters (not function/clock/MDA)
    params = {x for x in ids
              if (x in delays
                  or (len(x) >= 3 and x.isupper()
                      and x not in funcs and x not in clocks))}
    params -= mdas    # MDA wins over param if both
    ports = ids - params - funcs - mdas
    return params, funcs, ports, mdas


def _build_compileable_unit(sva: str, reference_sva: str = "",
                             rtl_context: str = "") -> str:
    """Wrap an SVA in a single self-contained module that Verilator/VCS
    can lint against. Strategy:

    Synthesize a fresh module declaring every identifier referenced by
    either the candidate SVA or the reference as a free input — except:
      * ALL_CAPS identifiers ≥3 chars → `parameter` (32'd0)
      * Identifiers that appear as `name(` → stub functions returning 0
    These two buckets fix VCS Error-[NCE]/[IRIPS]/[INF]/[INT] families
    where the lint engine needs a constant or a callable, not a wire.

    `rtl_context` is intentionally NOT used — industrial RTL has
    macro / package / hierarchical-path dependencies the linter can't
    resolve without the full project flist.
    """
    exclude = {"clk", "tb_reset"}
    sva_text = (sva or "") + " " + (reference_sva or "")
    ids = _identifiers_in_sva(sva, exclude) | _identifiers_in_sva(reference_sva, exclude)
    params, funcs, ports, mdas = _classify_identifiers(ids, sva_text)

    port_decls = ["    input logic clk", "    input logic tb_reset"]
    for ident in sorted(ports):
        port_decls.append(f"    input logic [31:0] {ident}")
    # 2D packed array for MDA accesses — covers `x[i][j]` and `x[i][j][k]`
    # patterns (32 elements each dim, 32-bit data).
    for ident in sorted(mdas):
        port_decls.append(f"    input logic [31:0][31:0][31:0] {ident}")

    body_decls = []
    for ident in sorted(params):
        # Default 32'd4: large enough that `[*PARAM-1]` and `[PARAM-1:0]`
        # don't degenerate to empty, small enough not to overflow indices.
        body_decls.append(f"  parameter logic [31:0] {ident} = 32'd4;")
    for ident in sorted(funcs):
        # Default-arg stub accepts 0-8 args (covers all observed call sites)
        # so the SVA can call `f()` / `f(a)` / `f(a,b,c)` / etc. uniformly.
        body_decls.append(
            f"  function automatic logic [31:0] {ident}(\n"
            f"    input logic [31:0] a0 = 32'd0, input logic [31:0] a1 = 32'd0,\n"
            f"    input logic [31:0] a2 = 32'd0, input logic [31:0] a3 = 32'd0,\n"
            f"    input logic [31:0] a4 = 32'd0, input logic [31:0] a5 = 32'd0,\n"
            f"    input logic [31:0] a6 = 32'd0, input logic [31:0] a7 = 32'd0);\n"
            f"    return 32'd0;\n"
            f"  endfunction"
        )

    return (
        "module sva_check (\n"
        + ",\n".join(port_decls)
        + "\n);\n"
        + ("\n".join(body_decls) + "\n" if body_decls else "")
        + "  " + (sva or "").strip() + "\n"
        "endmodule\n"
    )


def verilator_compile_check(sva: str, reference_sva: str = "",
                             rtl_context: str = "",
                             timeout: int = 8) -> bool:
    """Return True iff Verilator can `--lint-only --assert` parse the
    candidate SVA inside a synthetic free-input wrapper."""
    if not sva:
        return False
    code = _build_compileable_unit(sva, reference_sva, rtl_context)
    try:
        with tempfile.TemporaryDirectory(prefix="vrlr_chk_") as td:
            f = Path(td) / "check.sv"
            f.write_text(code)
            r = subprocess.run(
                ["verilator", "--lint-only", "--assert", "-Wno-fatal",
                 "-Wno-DECLFILENAME", "-Wno-MULTITOP", str(f)],
                capture_output=True, timeout=timeout,
            )
            return r.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False
    except Exception:
        return False


_VCS_COMPILE_CHECK = None
_VCS_DYNAMIC_CHECK = None


def _get_vcs_compile_check():
    """Lazy-import vcs_compile_check so the module loads even when the docker
    harness is unavailable (e.g. for unit tests that only exercise PEC)."""
    global _VCS_COMPILE_CHECK
    if _VCS_COMPILE_CHECK is None:
        import importlib.util as _ilu
        spec = _ilu.spec_from_file_location(
            "_vcs_check",
            str(Path(__file__).resolve().parent / "vcs_compile_check.py"))
        mod = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _VCS_COMPILE_CHECK = mod.vcs_compile_check
    return _VCS_COMPILE_CHECK


def _get_vcs_dynamic_check():
    """Lazy-import vcs_dynamic_check (differential simulation)."""
    global _VCS_DYNAMIC_CHECK
    if _VCS_DYNAMIC_CHECK is None:
        import importlib.util as _ilu
        spec = _ilu.spec_from_file_location(
            "_vcs_dyn",
            str(Path(__file__).resolve().parent / "vcs_dynamic_check.py"))
        mod = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _VCS_DYNAMIC_CHECK = mod.vcs_dynamic_check
    return _VCS_DYNAMIC_CHECK


def compute_reward_vcs_then_pec(completion: str, reference_sva: str,
                                rtl_context: str = "") -> float:
    """Tiered reward (VCS lint gate → PEC).

    VCS L-2016.06 with `-assert svaext` accepts the full SVA spec --- including
    `s_eventually` / `##[a:b]` / `intersect` / `throughout` / recursive
    properties --- which Verilator 5.020 rejects. Replacing the Verilator gate
    with VCS lifts the C3 (liveness) reward signal from 0% to ~15% on the
    GRPO pool at the cost of ~0.7s of extra latency per rollout.

    Reward table:
      no SVA / syntax fail   → 0.00   (bare regex / mock-verifier check)
      VCS lint fails         → 0.05   (extracted SVA but can't compile)
      VCS OK + PEC EQUIV     → 1.00
      VCS OK + PEC IMPLIES_* → 0.50
      VCS OK + PEC NOT_EQ    → 0.20
      VCS OK + PEC error     → 0.15   (UNSUPPORTED / PARSE / TIMEOUT --- we
                                       know the SVA is compilable, so give
                                       partial credit)
    """
    global _PEC, _INFER_RESET
    if _PEC is None:
        from src.pec_yosys import prop_equivalence, infer_reset_expr
        _PEC = prop_equivalence
        _INFER_RESET = infer_reset_expr
    sva = extract_sva(completion)
    if not sva:
        return 0.0
    if not syntax_check(sva)["ok"]:
        return 0.0
    # Stage 1: VCS lint --- ~0.9s/call, accepts the full SVA spec.
    vcs_compile_check = _get_vcs_compile_check()
    if not vcs_compile_check(sva, reference_sva, rtl_context):
        return 0.05
    # Stage 2: PEC --- only on syntactically-clean, compilable SVAs.
    rtl = rtl_context.strip() or _free_input_rtl(reference_sva)
    reset_expr = _INFER_RESET(rtl, reference_sva, sva)
    try:
        r = _PEC(sva, reference_sva, rtl, depth=10, timeout=15,
                 reset_expr=reset_expr,
                 liveness_bound=PEC_REWARD_LIVENESS_BOUND)
        verdict = r.verdict
    except Exception:
        verdict = "EXTRACT_ERROR"
    # Stage 3 fallback: when PEC can't decide (UNSUPPORTED / TIMEOUT /
    # PARSE_ERROR / UNKNOWN / EXTRACT_ERROR) replace the constant 0.10-0.15
    # floor with a continuous differential-simulation agreement score in
    # [0.15, 0.65]. This restores reward variance for the ~50% of rollouts
    # where sby's BMC bails on liveness / sequence operators it can't lower.
    if verdict in ("UNSUPPORTED", "TIMEOUT", "PARSE_ERROR",
                   "UNKNOWN", "EXTRACT_ERROR"):
        try:
            agreement = _get_vcs_dynamic_check()(
                sva, reference_sva, rtl_context, n_cycles=100, timeout=15)
        except Exception:
            agreement = None
        if agreement is not None:
            return 0.15 + 0.50 * agreement
        return 0.15
    return {
        "EQUIVALENT":         1.00,
        "IMPLIES_REF_TO_LM":  0.50,
        "IMPLIES_LM_TO_REF":  0.50,
        "NOT_EQUIVALENT":     0.20,
    }.get(verdict, 0.10)


def compute_reward_vcs_dynamic(completion: str, reference_sva: str,
                                rtl_context: str = "") -> float:
    """All-dynamic reward: VCS lint as a cheap pre-gate, then differential
    simulation gives a continuous [0, 1] agreement score against the
    reference SVA. Skips the formal PEC stage entirely.

    Reward table:
      no SVA / syntax fail     -> 0.00
      VCS lint fails           -> 0.05   (extracted but uncompilable)
      VCS OK + dynamic sim OK  -> agreement in [0, 1]   (continuous)
      VCS OK + dynamic fails   -> 0.10   (sim crashed / timed out)

    Compared to vcs_then_pec, this trades formal guarantees for:
      (a) maximum reward variance per GRPO group  (no 0.15 binary floor)
      (b) coverage of liveness / ranged-delay / intersect that sby rejects
      (c) lower per-call latency (~2s vs ~3-15s for sby on hard SVAs)

    Caveats:
      - Multi-bit literal equality (`x == 32'd5`) loses informativeness
        because we drive each port as $urandom_range(0, 1).
      - Train-eval mismatch: Func@1 evaluation still uses sby PEC, so
        the model may learn properties that simulate well but sby rejects.
      - 100 random cycles is a coarse oracle; equivalent properties that
        only diverge on rare states won't be distinguished.
    """
    sva = extract_sva(completion)
    if not sva:
        return 0.0
    if not syntax_check(sva)["ok"]:
        return 0.0
    vcs_compile_check = _get_vcs_compile_check()
    if not vcs_compile_check(sva, reference_sva, rtl_context):
        return 0.05
    try:
        agreement = _get_vcs_dynamic_check()(
            sva, reference_sva, rtl_context, n_cycles=100, timeout=15)
    except Exception:
        agreement = None
    if agreement is None:
        return 0.10
    return float(agreement)


REWARD_MODES = {
    "astdiff": compute_reward_astdiff,
    "pec_equiv": compute_reward_pec,
    "hybrid": compute_reward_hybrid,
    "pec_multiref": compute_reward_pec_multiref,
    "vcs_then_pec": compute_reward_vcs_then_pec,
    "vcs_dynamic": compute_reward_vcs_dynamic,
}


def latest_checkpoint(out_dir: Path) -> Path | None:
    cks = sorted(out_dir.glob("checkpoint-*"))
    return cks[-1] if cks else None


def eval_metric(report: dict, eval_tcls: set[int], eval_mode: str = "tcl_match",
                funcatk_k: int = 1) -> float:
    """Compute a scalar early-stop signal from an eval report.

    `eval_mode`:
      - "tcl_match": original greedy TCL-match rate. Reads per_tcl[lv].tcl_match.
      - "funcatk":  reads overall.func@K (K = funcatk_k) produced by
                     run_funcatk_eval.py. Respects eval_tcls by averaging
                     per_tcl[lv].func@K when the filter is non-empty.
    """
    if eval_mode == "funcatk":
        key = f"func@{funcatk_k}"
        if not eval_tcls:
            return float(report.get("overall", {}).get(key, 0.0)) / 100.0
        # Weighted by each TCL's evaluable count
        total_n = 0
        weighted = 0.0
        for lv in eval_tcls:
            stats = report.get("per_tcl", {}).get(str(lv), {})
            n = int(stats.get("evaluable", 0))
            if n > 0:
                weighted += float(stats.get(key, 0.0)) * n
                total_n += n
        return (weighted / total_n / 100.0) if total_n else 0.0
    # default: tcl_match
    if not eval_tcls:
        total = int(report.get("total", 0))
        match = int(report.get("match", 0))
        return (match / total) if total else 0.0
    total = 0
    match = 0
    for lv in eval_tcls:
        stats = report.get("per_tcl", {}).get(str(lv), {})
        total += int(stats.get("total", 0))
        match += int(stats.get("tcl_match", stats.get("match", 0)))
    return (match / total) if total else 0.0


def _run_one_grpo_eval(base_model: str, adapter: str,
                       tasks_path: Path, out_json: Path,
                       tag: str, args) -> dict:
    """One vLLM subprocess call against a single tasks.jsonl. Returns the
    raw eval JSON (funcatk-style or tcl_match-style depending on
    --eval-mode)."""
    if args.eval_mode == "funcatk":
        cmd = [
            sys.executable, str(FUNCATK_EVAL_SCRIPT),
            "--model", base_model,
            "--tasks", str(tasks_path),
            "--prompt-format", args.eval_prompt_format,
            "--num-samples", str(args.eval_num_samples),
            "--ks", ",".join(str(k) for k in sorted(
                {1, args.funcatk_k} | {int(x) for x in args.eval_ks.split(",") if x.strip()}
            )),
            "--max-new-tokens", str(args.eval_max_new),
            "--gpu-memory-utilization", str(args.vllm_gpu_memory_utilization),
            "--tensor-parallel-size", str(args.vllm_tensor_parallel_size),
            "--workers", str(args.eval_pec_workers),
            "--depth", str(args.eval_pec_depth),
            "--timeout", str(args.eval_pec_timeout),
            "--pec-reset-mode", args.eval_pec_reset_mode,
            "--liveness-bound", str(args.eval_liveness_bound),
            "--skip-coverage-check",
            "--skip-greedy-diagnostic",
            "--tag", tag,
            "--output", str(out_json),
        ]
    else:
        cmd = [
            sys.executable, str(VLLM_EVAL_SCRIPT),
            "--model", base_model,
            "--tasks", str(tasks_path),
            "--prompt-format", args.eval_prompt_format,
            "--max-new-tokens", str(args.eval_max_new),
            "--gpu-memory-utilization", str(args.vllm_gpu_memory_utilization),
            "--tensor-parallel-size", str(args.vllm_tensor_parallel_size),
            "--tag", tag,
            "--output", str(out_json),
        ]
    if adapter:
        cmd.extend(["--adapter", adapter])
    if args.eval_tcls:
        cmd.extend(["--filter-tcls", args.eval_tcls])
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.vllm_eval_gpu
    env["PATH"] = "${OSS_CAD_SUITE}/bin:" + env.get("PATH", "")
    env["PYTHONPATH"] = (str(REPO_ROOT) + ":"
                         + env.get("PYTHONPATH", ""))
    subprocess.run(cmd, check=True, env=env)
    with open(out_json) as f:
        return json.load(f)


def run_vllm_eval(base_model: str, adapter: str, args) -> dict:
    """Run vLLM eval on BOTH nl2sva_human and nl2sva_machine. Returns a
    combined report mirroring the SFT runner's schema:

      {
        "by_test": {"human": <full report>, "machine": <full report>},
        # aggregated for early-stop signal
        "total":  <sum>,
        "match":  <sum>,
        "per_tcl": {"1": {"total":, "match":}, ...},
        "overall": {"func@K": <weighted>, ...},     # funcatk mode only
        "per_tcl_funcatk": {...},                   # funcatk mode only
      }
    """
    if not args.vllm_eval_gpu:
        raise RuntimeError("vLLM eval requested but --vllm-eval-gpu is empty")

    eval_paths = {
        "human": Path(args.eval_tasks_human),
        "machine": Path(args.eval_tasks_machine),
    }
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_test = {}
    for name, path in eval_paths.items():
        if not path.exists():
            print(f"  [eval-skip] {name}: missing tasks file {path}")
            continue
        out_json = out_dir / f"_eval_step_{args._eval_target_step}__{name}.json"
        rep = _run_one_grpo_eval(
            base_model, adapter, path, out_json,
            f"grpo_step_{args._eval_target_step}__{name}", args,
        )
        by_test[name] = rep

    combined = {"by_test": by_test, "total": 0, "match": 0, "per_tcl": {}}
    if args.eval_mode == "funcatk":
        # Aggregate funcatk fields too: weighted by evaluable_tasks
        agg_eval = 0
        agg_strict = {k: 0.0 for k in (args.eval_ks.split(",") if args.eval_ks else ["1"])}
        agg_relaxed = {k: 0.0 for k in agg_strict}
        per_tcl_fa: dict = {}
        for name, rep in by_test.items():
            ev = int(rep.get("evaluable_tasks", 0) or 0)
            agg_eval += ev
            for k in list(agg_strict):
                key_s = f"func@{k}"
                key_r = f"func_relaxed@{k}"
                v_s = float(rep.get("overall", {}).get(key_s, 0.0))
                v_r = float(rep.get("overall", {}).get(key_r, 0.0))
                agg_strict[k] += v_s * ev
                agg_relaxed[k] += v_r * ev
            for tcl_k, tcl_v in (rep.get("per_tcl") or {}).items():
                slot = per_tcl_fa.setdefault(tcl_k, {"evaluable": 0,
                                                    "func@1_w": 0.0,
                                                    "func@16_w": 0.0})
                ev_t = int(tcl_v.get("evaluable", 0))
                slot["evaluable"] += ev_t
                for k in ("1", "16"):
                    fk = f"func@{k}"
                    if fk in tcl_v:
                        slot[f"func@{k}_w"] += float(tcl_v.get(fk, 0.0)) * ev_t
        # finalise
        combined["overall"] = {
            **{f"func@{k}": (agg_strict[k] / max(agg_eval, 1)) for k in agg_strict},
            **{f"func_relaxed@{k}": (agg_relaxed[k] / max(agg_eval, 1)) for k in agg_relaxed},
        }
        for k, slot in per_tcl_fa.items():
            ev = max(slot.pop("evaluable"), 1)
            slot["func@1"] = slot.pop("func@1_w") / ev
            slot["func@16"] = slot.pop("func@16_w") / ev
        combined["per_tcl"] = per_tcl_fa
        combined["evaluable_tasks"] = agg_eval
    else:
        # tcl_match mode: aggregate total / match across human + machine
        for name, rep in by_test.items():
            combined["total"] += int(rep.get("total", 0))
            combined["match"] += int(rep.get("match", 0))
            for tcl_k, tcl_v in (rep.get("per_tcl") or {}).items():
                slot = combined["per_tcl"].setdefault(tcl_k, {"total": 0, "match": 0})
                slot["total"] += int(tcl_v.get("total", 0))
                slot["match"] += int(tcl_v.get("match", 0))
    return combined


# --- TRL GRPOTrainer wiring --------------------------------------------------

def build_dataset(pool_path: Path, max_samples: int | None = None,
                  prompt_format: str = "simple"):
    """Build GRPO prompt dataset.

    Supports three pool schemas:
      - tiered pool: fields {sva, nl, expected_tcl, tier, ...}
      - SFT-derived pool: fields {reference_sva, nl, expected_tcl, rtl_context, ...}
      - phase2 pool: adds `ref_svas` (list of PEC-verified equivalent refs)

    `prompt_format`:
      - "simple": original `Generate an SVA assertion for:\\n{nl}` user msg
      - "fveval": match run_eval_nl2sva_human.py exactly — testbench RTL +
        "Question: Create a SVA assertion that checks: {nl}" + answer template
    """
    from datasets import Dataset
    rows = []
    with open(pool_path) as f:
        for line in f:
            r = json.loads(line)
            ref = r.get("reference_sva") or r.get("sva", "")
            ref_svas = r.get("ref_svas") or [ref]
            nl = (r.get("nl") or "").strip()
            rtl_context = r.get("rtl_context", "")
            if not nl or re.match(r"^\s*\[(?:ASSERT|ASSUME|COVER)", nl, re.I):
                mod = r.get("module_name", "")
                nl = (f"Write an SVA for module {mod} "
                      f"(TCL level {r.get('expected_tcl', '?')})")
            if prompt_format == "fveval":
                prompt_msgs = [
                    {"role": "system", "content": FVEVAL_SYSTEM_PROMPT},
                    {"role": "user",
                     "content": build_fveval_user_prompt(nl, rtl_context)},
                ]
            else:
                prompt_msgs = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",
                     "content": f"Generate an SVA assertion for:\n{nl}"},
                ]
            rows.append({
                "prompt": prompt_msgs,
                "reference_sva": ref,
                "ref_svas": ref_svas,
                "rtl_context": rtl_context,
                "expected_tcl": r.get("expected_tcl", 0),
                "tier": r.get("tier", 2),
            })
    if max_samples:
        rows = rows[:max_samples]
    return Dataset.from_list(rows)


def make_reward_fn(mode: str = "astdiff"):
    """Return a TRL-compatible reward function for the given mode.

    TRL GRPOTrainer calls reward_fn(prompts, completions, **kwargs) where
    kwargs includes the original dataset columns (reference_sva, rtl_context,
    expected_tcl). Must return a list[float] — one reward per completion.
    """
    if mode not in REWARD_MODES:
        raise ValueError(f"unknown reward mode {mode!r} "
                         f"(choices: {list(REWARD_MODES)})")
    rfn = REWARD_MODES[mode]
    is_multiref = (mode == "pec_multiref")

    def reward_fn(completions, **kwargs):
        refs = kwargs.get("reference_sva")
        ref_lists = kwargs.get("ref_svas")
        rtls = kwargs.get("rtl_context")
        out = []
        if isinstance(completions, str):
            completions = [completions]
        for i, c in enumerate(completions):
            if isinstance(c, list):   # chat-format: list of messages
                c = c[-1].get("content", "") if c else ""
            rtl = rtls[i] if isinstance(rtls, list) else (rtls or "")
            if is_multiref:
                refs_i = (ref_lists[i] if isinstance(ref_lists, list)
                          else ref_lists)
                if not refs_i:
                    refs_i = [refs[i] if isinstance(refs, list) else refs]
                out.append(rfn(c, refs_i, rtl or ""))
            else:
                ref = refs[i] if isinstance(refs, list) else refs
                out.append(rfn(c, ref or "", rtl or ""))
        return out
    return reward_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True,
                    help="path to SFT checkpoint (starting policy)")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num-generations", type=int, default=4,
                    help="G: rollouts per prompt")
    ap.add_argument("--batch-size", type=int, default=1,
                    help="per-device prompt batch; effective rollouts = batch*G")
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--beta", type=float, default=0.04,
                    help="KL coefficient")
    ap.add_argument("--max-prompts", type=int, default=0,
                    help="debug: cap dataset size")
    ap.add_argument("--max-completion-length", type=int, default=192)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lora", action="store_true", default=True,
                    help="use LoRA (required to fit 7B on 1 GPU)")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--reward-mode", default="astdiff",
                    choices=list(REWARD_MODES),
                    help="astdiff = original syntax+AST; "
                         "pec_equiv = SymbiYosys property-equivalence to ref "
                         "(strong but slow); hybrid = mixed; "
                         "pec_multiref = max(PEC) over multi-ref pool (Phase 2)")
    ap.add_argument("--pool", default="",
                    help="pool jsonl path; default depends on --reward-mode "
                         "(astdiff → tiered, pec_equiv/hybrid → from_sft, "
                         "pec_multiref → phase2)")
    ap.add_argument("--prompt-format", default="simple",
                    choices=["simple", "fveval"],
                    help="simple = bare 'Generate an SVA assertion for:\\n{nl}' "
                         "(legacy); fveval = match run_eval_nl2sva_human.py "
                         "verbatim (testbench RTL + Question + answer template) "
                         "to close the train/eval distribution gap.")
    ap.add_argument("--eval-every-steps", type=int, default=50)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--early-stop-min-delta", type=float, default=0.0)
    ap.add_argument("--eval-prompt-format", default="fveval",
                    choices=["simple", "fveval"])
    ap.add_argument("--eval-max-new", type=int, default=256)
    ap.add_argument("--eval-tcls", default="", help="comma-separated TCLs for early-stop metric")
    ap.add_argument("--eval-mode", default="tcl_match",
                    choices=["tcl_match", "funcatk"],
                    help="tcl_match (default) = greedy TCL match via "
                    "run_eval_nl2sva_human_vllm; funcatk = full func@k with "
                    "PEC (Cadence-aligned + bounded-liveness) via "
                    "run_funcatk_eval.")
    ap.add_argument("--eval-tasks",
                    default="",
                    help="DEPRECATED single-set path. If non-empty, treated as "
                         "an alias for --eval-tasks-human (back-compat).")
    ap.add_argument("--eval-tasks-human",
                    default=str(REPO_ROOT / "data" / "test" / "nl2sva_human.jsonl"),
                    help="full nl2sva_human jsonl path")
    ap.add_argument("--eval-tasks-machine",
                    default=str(REPO_ROOT / "data" / "test" / "nl2sva_machine.jsonl"),
                    help="full nl2sva_machine jsonl path")
    ap.add_argument("--eval-num-samples", type=int, default=16,
                    help="funcatk mode: samples per task")
    ap.add_argument("--eval-ks", default="1,16",
                    help="funcatk mode: comma-separated k values to report")
    ap.add_argument("--funcatk-k", type=int, default=1,
                    help="funcatk mode: which k is the early-stop signal")
    ap.add_argument("--eval-pec-workers", type=int, default=8)
    ap.add_argument("--eval-pec-depth", type=int, default=15)
    ap.add_argument("--eval-pec-timeout", type=int, default=30)
    ap.add_argument("--eval-pec-reset-mode", default="auto",
                    help="Cadence reset canonicalization mode for eval PEC")
    ap.add_argument("--eval-liveness-bound", type=int, default=15,
                    help="bounded-liveness rewrite bound for eval PEC")
    ap.add_argument("--vllm-eval-gpu", default="")
    ap.add_argument("--vllm-tensor-parallel-size", type=int, default=1)
    ap.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.80)
    args = ap.parse_args()
    # Back-compat: if user passed --eval-tasks, redirect to --eval-tasks-human
    if args.eval_tasks:
        args.eval_tasks_human = args.eval_tasks
    if not args.pool:
        if args.reward_mode == "pec_multiref":
            args.pool = str(GRPO_POOL_PHASE2)
        elif args.reward_mode in ("pec_equiv", "hybrid"):
            args.pool = str(GRPO_POOL_FROM_SFT)
        else:
            args.pool = str(GRPO_POOL_DEFAULT)

    from trl import GRPOConfig, GRPOTrainer
    from peft import LoraConfig

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    print(f"[reward] mode = {args.reward_mode}")
    print(f"[pool]   path = {args.pool}")

    dataset = build_dataset(Path(args.pool), args.max_prompts or None,
                            prompt_format=args.prompt_format)
    print(f"[prompt] format = {args.prompt_format}")
    print(f"[data] {len(dataset)} prompts from GRPO pool")
    if "tier" in dataset.column_names:
        print(f"       tiers: "
              f"T1={sum(1 for r in dataset if r['tier']==1)}, "
              f"T2={sum(1 for r in dataset if r['tier']==2)}")

    peft_cfg = None
    if args.lora:
        peft_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=2 * args.lora_r,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        print(f"[lora] r={args.lora_r}, alpha={2*args.lora_r}")

    eval_tcls = {int(x) for x in args.eval_tcls.split(",") if x.strip()}
    best_metric = -1.0
    bad_evals = 0
    current_step = 0
    resume_ckpt = None
    trainer = None
    t0 = time.time()
    while current_step < args.max_steps:
        target_step = min(args.max_steps, current_step + args.eval_every_steps)
        args._eval_target_step = target_step
        # TRL 0.12+ requires generation_batch_size divisible by num_generations.
        # With per_device_batch=1 and 1 GPU, bump gradient_accumulation to
        # num_generations so the effective generation batch matches.
        grad_accum = max(1, args.num_generations // max(1, args.batch_size))
        cfg = GRPOConfig(
            output_dir=str(out),
            learning_rate=args.lr,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=grad_accum,
            num_generations=args.num_generations,
            max_steps=target_step,
            max_completion_length=args.max_completion_length,
            max_prompt_length=1024,
            logging_steps=1,
            save_steps=max(1, target_step),
            save_total_limit=8,   # keep all 4-8 checkpoints so best isn't auto-deleted
            beta=args.beta,
            epsilon=0.2,
            temperature=1.1,
            bf16=True,
            gradient_checkpointing=True,
            report_to=[],
            seed=args.seed,
        )
        trainer = GRPOTrainer(
            model=args.policy,
            reward_funcs=make_reward_fn(args.reward_mode),
            args=cfg,
            train_dataset=dataset,
            peft_config=peft_cfg,
        )
        trainer.train(resume_from_checkpoint=str(resume_ckpt) if resume_ckpt else None)
        current_step = target_step
        ckpt = latest_checkpoint(out)
        adapter_path = str(ckpt) if (args.lora and ckpt is not None) else ""
        base_model = args.policy if args.lora else (str(ckpt) if ckpt is not None else args.policy)
        # Free the trainer before vLLM eval spawns — they contend for the
        # same GPU. vLLM 0.18 asserts that free memory is non-increasing
        # during its profiling step; if PyTorch's caching allocator releases
        # memory mid-profile, the assertion fires. Synchronize fully and
        # sleep long enough for the driver to reconcile cudaFree before
        # spawning the eval subprocess.
        del trainer
        trainer = None
        import gc
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # repeat once more — Adam states for LoRA optimizer release lazily
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        time.sleep(15)
        report = run_vllm_eval(base_model, adapter_path, args)
        metric = eval_metric(report, eval_tcls, eval_mode=args.eval_mode,
                             funcatk_k=args.funcatk_k)
        improved = metric > (best_metric + args.early_stop_min_delta)
        if improved:
            best_metric = metric
            bad_evals = 0
        else:
            bad_evals += 1
        print(f"[eval] step={current_step} match={metric:.3f} best={best_metric:.3f} bad={bad_evals}/{args.patience}")
        if args.patience > 0 and bad_evals >= args.patience:
            print(f"[early-stop] GRPO hit patience={args.patience} at step={current_step}")
            resume_ckpt = ckpt
            break
        resume_ckpt = ckpt

    print(f"[grpo] training done in {time.time()-t0:.1f}s")
    # Note: trainer is cleared before each eval to reduce VRAM pressure on
    # single-GPU setups; latest checkpoint is already saved via save_steps.
    last_ckpt = latest_checkpoint(out)
    if last_ckpt is not None:
        print(f"[grpo] latest checkpoint: {last_ckpt}")
    elif trainer is not None:
        trainer.save_model(str(out / "final"))
        print(f"[grpo] saved to {out / 'final'}")


if __name__ == "__main__":
    main()
