#!/usr/bin/env python3
"""
run_rlvf.py — RLVF (GRPO) training skeleton.

SCALE-BLOCKED: Full training cannot run in the current environment.
Requirements:
  - GPU: 4× A100 80GB (estimated ~180 GPU-hours, no critic)
  - Formal verifier: SymbiYosys or JasperGold
  - Fine-tuned base model: TemporalSVA-FT (from curriculum fine-tuning)
  - Linux OS (SymbiYosys only runs on Linux)

This file documents the intended implementation per FINAL_PROPOSAL.md
Component 3. The GRPO *advantage math* is implementable without the
verifier, and is unit-tested below (`--demo` mode).

Paper RLVF reward (unchanged when switching PPO → GRPO):
  R = 0.15·syntax + 0.40·formal_verify + 0.20·(1-vacuity) + 0.25·AST_sim

Why GRPO instead of PPO (per FINAL_PROPOSAL.md):
  - w_fv (40% of reward) is a hard verifier pass/fail → verifiable-reward
    regime where GRPO empirically matches/exceeds PPO (DeepSeek-R1,
    Qwen2.5-Math, DAPO).
  - No critic/value model → ~40% memory saving.
  - Group-relative baseline naturally separates "syntactically valid but
    vacuous" from "syntactically wrong" rollouts under the same prompt.

SoTA baselines for final comparison (IMPORTANT — experiments MUST report
against these; see FINAL_PROPOSAL.md Evaluation Sketch):
  - CodeV-SVA-14B (reported 75.8% NL2SVA-Human)  ← primary SoTA
  - AssertLLM
  - Hybrid-NL2SVA
  - GPT-5, DeepSeek-R1 (general-purpose SoTA)
  - AssertFix (repair baseline, for WaveformLens comparison)
"""

import sys
import os
import math
import statistics
from typing import List, Dict, Optional


# -----------------------------------------------------------------------------
# Prerequisite check
# -----------------------------------------------------------------------------
def check_prerequisites():
    """Raise if prerequisites for real training are not met."""
    issues = []

    try:
        import torch
        if not torch.cuda.is_available():
            issues.append("No CUDA GPU detected (need 4×A100)")
    except ImportError:
        issues.append("PyTorch not installed")

    import shutil
    if shutil.which("sby") is None:
        issues.append("SymbiYosys ('sby') not found in PATH")

    if issues:
        raise EnvironmentError(
            "SCALE-BLOCKED: GRPO-RLVF training cannot run.\n" +
            "\n".join(f"  - {i}" for i in issues)
        )


# -----------------------------------------------------------------------------
# Reward function (structure per proposal; formal/vacuity/sim are SCALE-BLOCKED)
# -----------------------------------------------------------------------------
def compute_reward(sva: str, rtl: str, ground_truth_sva: str) -> dict:
    """
    RLVF reward function (FINAL_PROPOSAL.md Section: Reward Function).

    R(SVA, RTL) = w_syn·syntax_ok + w_fv·formal_verify
                + w_vac·(1-vacuity) + w_sim·AST_edit_sim

    Weights: w_syn=0.15, w_fv=0.40, w_vac=0.20, w_sim=0.25
    """
    raise NotImplementedError(
        "SCALE-BLOCKED: compute_reward needs SymbiYosys for formal verification.\n"
        "  - syntax (0.15): implementable via src/mock_verifier.py\n"
        "  - formal_verify (0.40): requires SymbiYosys\n"
        "  - vacuity (0.20): requires SymbiYosys\n"
        "  - AST_similarity (0.25): requires Verible parser\n"
        "Docker-based SymbiYosys stub: yosyshq/sby:latest"
    )


# -----------------------------------------------------------------------------
# GRPO advantage computation (IMPLEMENTABLE — no GPU or verifier needed)
# -----------------------------------------------------------------------------
def group_relative_advantages(rewards: List[float], eps: float = 1e-8) -> List[float]:
    """
    GRPO advantage estimation (FINAL_PROPOSAL.md Component 3):

        Â_i = (R_i - mean({R_j})) / (std({R_j}) + eps)

    This is the core of GRPO — critic-free advantage from group statistics.

    Args:
        rewards: list of scalar rewards for the G rollouts of a single prompt
        eps:     numerical stabilizer (paper: 1e-8)

    Returns:
        list of advantages, same length as rewards
    """
    if len(rewards) == 0:
        return []
    if len(rewards) == 1:
        # Degenerate group — advantage is 0 (no relative signal)
        return [0.0]
    mu = sum(rewards) / len(rewards)
    var = sum((r - mu) ** 2 for r in rewards) / len(rewards)
    sigma = math.sqrt(var)
    return [(r - mu) / (sigma + eps) for r in rewards]


def detect_variance_collapse(rewards: List[float], tol: float = 1e-6) -> bool:
    """
    GRPO failure mode (per FINAL_PROPOSAL.md Failure Modes table):
    when all G rollouts receive the same reward, group std ≈ 0 → Â=0,
    so there is no learning signal. Mitigation: resample at higher
    temperature or fall back to REINFORCE-style centering.
    """
    if len(rewards) < 2:
        return True
    return statistics.pstdev(rewards) < tol


# -----------------------------------------------------------------------------
# GRPO training loop (SCALE-BLOCKED)
# -----------------------------------------------------------------------------
def run_grpo_training(
    model_path: str,
    train_data_path: str,
    output_dir: str,
    group_size: int = 8,
    clip_range: float = 0.2,
    kl_coeff: float = 0.04,
    learning_rate: float = 1e-6,
    n_steps: int = 8000,
):
    """
    GRPO training loop (FINAL_PROPOSAL.md Component 3 hyperparameters).

    Hyperparameters (default values per proposal):
      - group_size G:      8 samples per prompt (DeepSeekMath default)
      - clip_range ε:      0.2
      - kl_coeff β:        0.04 (adaptive: ↑ if KL>10, ↓ if KL<5)
      - learning_rate:     1e-6 (10× smaller than fine-tuning)
      - n_steps:           8K updates → ~64K rollouts (8K × G=8)
      - reward_norm:       group-wise standardization (intrinsic to GRPO)

    Loss (token-level ratio ρ_{i,t}):
      L = -E[1/|o_i| · Σ_t min(ρ_{i,t}·Â_i, clip(ρ_{i,t},1-ε,1+ε)·Â_i)]
          + β·KL(π || π_ref)

    SCALE-BLOCKED: Requires 4×A100, SymbiYosys, ~180 GPU-hours.
    """
    raise NotImplementedError(
        "SCALE-BLOCKED: GRPO training requires GPU cluster.\n"
        f"  model:       {model_path}\n"
        f"  data:        {train_data_path}\n"
        f"  group_size:  {group_size}\n"
        f"  clip_range:  {clip_range}\n"
        f"  kl_coeff:    {kl_coeff}\n"
        f"  lr:          {learning_rate}\n"
        f"  n_steps:     {n_steps}\n"
        "  Hardware: 4×A100 80GB\n"
        "  Estimated: 180 GPU-hours (GRPO, no critic) "
        "+ 250 CPU-hours (verifier)\n"
        "  Reference impl: TRL `GRPOTrainer` or verl framework\n"
        "  Paper ref: FINAL_PROPOSAL.md Component 3\n"
        "\n"
        "  IMPORTANT — evaluation MUST report SoTA comparison:\n"
        "    - CodeV-SVA-14B (primary, 75.8% NL2SVA-Human)\n"
        "    - AssertLLM, Hybrid-NL2SVA\n"
        "    - GPT-5, DeepSeek-R1 (general-purpose)\n"
        "    - AssertFix (repair baseline)\n"
        "  Per FINAL_PROPOSAL.md success condition (1):\n"
        "    ≥30pp improvement on L4/L5 vs CodeV-SVA-14B."
    )


# -----------------------------------------------------------------------------
# Mock demo — exercises the advantage math against mock rewards
# -----------------------------------------------------------------------------
def mock_rlvf_demo():
    """
    Demo: simulate G=8 rollouts for a prompt, score each with the mock
    (syntax-only) reward, then compute GRPO group-relative advantages.

    This exercises the real GRPO advantage math — only the reward component
    is mocked. In full training, rewards come from SymbiYosys.
    """
    import json
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.mock_verifier import MockVerifier

    mv = MockVerifier()

    # One prompt, 8 candidate rollouts (simulating group sample)
    # Mix of TCL-1..5 quality + a couple of broken ones
    group = [
        ("assert property (@(posedge clk) req |-> ##1 gnt);",        "good TCL-4"),
        ("assert property (@(posedge clk) req |-> s_eventually gnt);", "good TCL-5"),
        ("assert property (@(posedge clk) valid);",                  "TCL-1 comb"),
        ("assert property (@(posedge clk) req |-> ##[1:3] gnt);",    "good TCL-3"),
        ("req |-> gnt",                                              "bad: no keyword"),
        ("assert property (@(posedge clk) req |-> gnt",              "bad: no semicolon"),
        ("assert property (@(posedge clk) a ##1 b ##1 c);",          "good TCL-2"),
        ("assert property (@(posedge clk) req |-> ##2 ack);",        "good TCL-4"),
    ]

    print("=" * 66)
    print("GRPO Mock Demo — 1 prompt, G=8 rollouts")
    print("=" * 66)
    print("Reward component: syntax-only (mock). formal/vacuity/sim BLOCKED.")
    print()

    rewards = []
    labels = []
    for sva, label in group:
        r = mv.mock_reward(sva)
        rewards.append(r["total_mock"])
        labels.append(label)

    # GRPO advantage math — fully implemented, no GPU needed
    advantages = group_relative_advantages(rewards)
    collapsed = detect_variance_collapse(rewards)

    print(f"{'idx':>3} | {'label':<20} | {'reward':>7} | {'advantage':>9}")
    print("-" * 66)
    for i, (lab, r, a) in enumerate(zip(labels, rewards, advantages)):
        print(f"{i:>3} | {lab:<20} | {r:>7.3f} | {a:>+9.3f}")
    print()

    print(f"Group mean reward : {sum(rewards)/len(rewards):.3f}")
    print(f"Group std  reward : {statistics.pstdev(rewards):.3f}")
    print(f"Variance collapse : {collapsed}")
    print()

    print("SCALE-BLOCKED COMPONENTS (full training):")
    print("  - formal verify reward (0.40): needs SymbiYosys")
    print("  - vacuity reward (0.20):       needs SymbiYosys")
    print("  - AST similarity reward (0.25): needs Verible parser")
    print("  - GRPO policy update:          needs 4×A100 GPU")
    print()

    print("IMPORTANT — SoTA comparison (per user instruction):")
    print("  Full eval must report GRPO-TemporalSVA vs:")
    print("    - CodeV-SVA-14B       (primary SoTA, 75.8% NL2SVA-Human)")
    print("    - AssertLLM, Hybrid-NL2SVA")
    print("    - GPT-5, DeepSeek-R1  (general-purpose)")
    print("    - AssertFix           (repair baseline for WaveformLens)")
    print("  Success condition: ≥30pp gain on L4/L5 vs CodeV-SVA-14B.")
    print()

    print("To run real GRPO training (on a GPU box):")
    print("  docker pull yosyshq/sby:latest")
    print("  # then: pip install trl>=0.11  # GRPOTrainer")
    print("  python scripts/run_rlvf.py <ft_model> <data_dir>")


# -----------------------------------------------------------------------------
# Self-tests for the advantage math (run unconditionally before training)
# -----------------------------------------------------------------------------
def _self_test():
    # Zero-variance group → all advantages 0
    adv = group_relative_advantages([0.5, 0.5, 0.5, 0.5])
    assert all(abs(a) < 1e-3 for a in adv), f"zero-var advantages should be ~0, got {adv}"
    assert detect_variance_collapse([0.5, 0.5, 0.5])

    # Non-degenerate group → zero-mean, unit-ish std
    adv = group_relative_advantages([0.0, 1.0, 0.5, 0.5])
    assert abs(sum(adv) / len(adv)) < 1e-6, f"advantages should be zero-mean, got {adv}"
    assert not detect_variance_collapse([0.0, 1.0, 0.5, 0.5])

    # Singleton / empty
    assert group_relative_advantages([]) == []
    assert group_relative_advantages([0.7]) == [0.0]
    print("[self-test] GRPO advantage math: OK")


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 66)
    print("GRPO-RLVF Training Script (TemporalSVA Component 3)")
    print("=" * 66)

    _self_test()

    if "--demo" in sys.argv:
        mock_rlvf_demo()
        sys.exit(0)

    try:
        check_prerequisites()
    except EnvironmentError as e:
        print(f"\n{e}")
        print("\nRunning mock demo instead (--demo mode)...")
        print("Use: python scripts/run_rlvf.py --demo\n")
        mock_rlvf_demo()
        sys.exit(1)

    run_grpo_training(
        model_path=sys.argv[1] if len(sys.argv) > 1 else "temporalsva-ft",
        train_data_path=sys.argv[2] if len(sys.argv) > 2 else "data/train",
        output_dir="results/rlvf",
    )
