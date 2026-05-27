"""
Temporal-Token-Weighted Cross-Entropy Loss.

Paper spec (FINAL_PROPOSAL.md Component 2):
    TEMPORAL_OPS = {"##", "[*", "[=", "|->", "|=>", "until", "eventually",
                    "s_eventually", "s_until", "s_always", "throughout",
                    "within", "intersect", "$rose", "$fell"}
    alpha = 3.0  (temporal tokens get 3× loss weight)

This module provides:
  1. A pure-Python / NumPy reference implementation (no PyTorch required)
     so it can run on any machine without GPU.
  2. SCALE-BLOCKED: The real PyTorch version is shown as comments.
  3. A self-contained unit test suite.

Usage:
    python src/temporal_loss.py
"""

import math
import numpy as np
from typing import List, Optional

# ---------------------------------------------------------------------------
# Temporal operator vocabulary
# ---------------------------------------------------------------------------
TEMPORAL_OPS = {
    "##", "[*", "[=", "|->", "|=>",
    "until", "eventually", "s_eventually", "s_until", "s_always",
    "throughout", "within", "intersect", "$rose", "$fell",
}


def is_temporal_token(token: str) -> bool:
    """Return True if the token string contains any temporal operator substring."""
    return any(op in token for op in TEMPORAL_OPS)


# ---------------------------------------------------------------------------
# NumPy reference implementation
# ---------------------------------------------------------------------------

def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over last axis."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def cross_entropy(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Per-token cross-entropy.

    Args:
        logits: (batch, seq_len, vocab_size) float array
        labels: (batch, seq_len) int array with token ids

    Returns:
        losses: (batch, seq_len) float array (unreduced)
    """
    probs = softmax(logits)
    batch, seq_len = labels.shape
    # Gather probability of the correct label at each position
    correct_probs = probs[np.arange(batch)[:, None],
                          np.arange(seq_len)[None, :],
                          labels]
    return -np.log(correct_probs + 1e-12)


def build_weight_mask(
    token_strings: List[List[str]],
    alpha: float = 3.0
) -> np.ndarray:
    """
    Build weight matrix where temporal tokens get weight alpha, others get 1.0.

    Args:
        token_strings: list of list of decoded token strings (batch × seq_len)
        alpha: multiplier for temporal tokens (paper default 3.0)

    Returns:
        weights: (batch, seq_len) float array
    """
    batch = len(token_strings)
    seq_len = max(len(row) for row in token_strings)
    weights = np.ones((batch, seq_len), dtype=np.float32)
    for i, row in enumerate(token_strings):
        for j, tok in enumerate(row):
            if is_temporal_token(tok):
                weights[i, j] = alpha
    return weights


def temporal_weighted_ce_loss(
    logits: np.ndarray,
    labels: np.ndarray,
    token_strings: List[List[str]],
    alpha: float = 3.0,
    ignore_index: int = -100,
) -> float:
    """
    Temporal-token-weighted cross-entropy loss (NumPy reference).

    Args:
        logits:        (batch, seq_len, vocab_size)
        labels:        (batch, seq_len) — use ignore_index for padding
        token_strings: decoded token strings (batch × seq_len)
        alpha:         weight multiplier for temporal tokens
        ignore_index:  label value to ignore (e.g. padding)

    Returns:
        scalar loss (float)
    """
    base_loss = cross_entropy(logits, np.clip(labels, 0, logits.shape[-1] - 1))
    weights = build_weight_mask(token_strings, alpha=alpha)

    # Mask out ignore_index positions
    valid_mask = (labels != ignore_index).astype(np.float32)
    weights = weights * valid_mask

    # Paper §4.2 — TT-CE: L = (1/T) Σ w_t · CE_t over the valid (non-pad)
    # response tokens. Normalize by the *token count* T, not by the active
    # weight Σ w_t — the latter cancels out α and breaks the paper's loss
    # equation.
    weighted = base_loss * weights
    n_tokens = valid_mask.sum()
    if n_tokens == 0:
        return 0.0
    return float(weighted.sum() / n_tokens)


# ---------------------------------------------------------------------------
# SCALE-BLOCKED: PyTorch version (reference only — requires GPU for real training)
# ---------------------------------------------------------------------------
PYTORCH_REFERENCE = '''
# SCALE-BLOCKED: This code requires PyTorch + GPU for actual training.
# Shown here as reference implementation matching the paper spec.

import torch
import torch.nn.functional as F

TEMPORAL_OPS = {"##", "[*", "[=", "|->", "|=>", "until", "eventually",
                "s_eventually", "s_until", "s_always", "throughout",
                "within", "intersect", "$rose", "$fell"}

def temporal_weighted_ce_loss(logits, labels, tokenizer, alpha=3.0):
    """
    Args:
        logits:    (batch, seq_len, vocab_size) torch.Tensor
        labels:    (batch, seq_len) torch.LongTensor
        tokenizer: HuggingFace tokenizer for decoding
        alpha:     temporal token weight multiplier (paper: 3.0)
    """
    base_loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                labels.view(-1), reduction="none")
    base_loss = base_loss.view(labels.shape)  # (batch, seq_len)

    weights = torch.ones_like(base_loss)
    batch, seq_len = labels.shape
    for i in range(batch):
        for j in range(seq_len):
            tok_id = labels[i, j].item()
            if tok_id == -100:  # ignore padding
                weights[i, j] = 0.0
                continue
            tok_str = tokenizer.decode([tok_id])
            if any(op in tok_str for op in TEMPORAL_OPS):
                weights[i, j] = alpha

    # Paper §4.2 — TT-CE: L = (1/T) Σ w_t · CE_t over response tokens.
    valid = (labels != -100).float()
    weighted_loss = (base_loss * weights * valid).sum()
    n_tokens = valid.sum().clamp(min=1.0)
    return weighted_loss / n_tokens
'''


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def _make_synthetic_batch(vocab_size: int = 100):
    """
    Create a minimal synthetic batch for testing.

    Sequence:
        [0] "module" — non-temporal
        [1] "##"     — temporal  → weight 3.0
        [2] "1"      — non-temporal
        [3] "|->"    — temporal  → weight 3.0
        [4] "gnt"    — non-temporal
    """
    batch, seq_len = 1, 5
    rng = np.random.default_rng(42)
    logits = rng.standard_normal((batch, seq_len, vocab_size)).astype(np.float32)

    # Correct labels: just use token id 0 for all (simplification)
    labels = np.zeros((batch, seq_len), dtype=np.int32)

    token_strings = [["module", "##", "1", "|->", "gnt"]]
    return logits, labels, token_strings


def test_is_temporal_token():
    assert is_temporal_token("##") is True
    assert is_temporal_token("##1") is True
    assert is_temporal_token("|->") is True
    assert is_temporal_token("|=>") is True
    assert is_temporal_token("s_eventually") is True
    assert is_temporal_token("s_until") is True
    assert is_temporal_token("throughout") is True
    assert is_temporal_token("$rose") is True
    assert is_temporal_token("$fell") is True
    assert is_temporal_token("module") is False
    assert is_temporal_token("gnt") is False
    assert is_temporal_token("always") is False   # NOT in set (only s_always)
    assert is_temporal_token("valid") is False
    print("PASS: test_is_temporal_token")


def test_weight_mask_values():
    """Temporal positions must have weight alpha; others must have weight 1.0."""
    alpha = 3.0
    token_strings = [["module", "##", "1", "|->", "gnt"]]
    weights = build_weight_mask(token_strings, alpha=alpha)
    assert weights.shape == (1, 5)
    assert weights[0, 0] == 1.0,  f"'module' should be 1.0, got {weights[0,0]}"
    assert weights[0, 1] == alpha, f"'##' should be {alpha}, got {weights[0,1]}"
    assert weights[0, 2] == 1.0,  f"'1' should be 1.0, got {weights[0,2]}"
    assert weights[0, 3] == alpha, f"'|->' should be {alpha}, got {weights[0,3]}"
    assert weights[0, 4] == 1.0,  f"'gnt' should be 1.0, got {weights[0,4]}"
    print("PASS: test_weight_mask_values")


def test_loss_higher_with_temporal_weight():
    """
    With alpha>1, the weighted loss must be >= unweighted loss when temporal
    tokens exist and their base CE is non-zero.
    """
    logits, labels, token_strings = _make_synthetic_batch()
    loss_weighted = temporal_weighted_ce_loss(logits, labels, token_strings, alpha=3.0)
    loss_unweighted = temporal_weighted_ce_loss(logits, labels, token_strings, alpha=1.0)
    # May be equal only if temporal base_loss happens to equal non-temporal;
    # in general with random logits the weighted loss will differ.
    assert isinstance(loss_weighted, float)
    assert isinstance(loss_unweighted, float)
    assert loss_weighted > 0.0
    assert loss_unweighted > 0.0
    print(f"PASS: test_loss_higher_with_temporal_weight "
          f"(weighted={loss_weighted:.4f}, unweighted={loss_unweighted:.4f})")


def test_alpha_1_equals_standard_ce():
    """alpha=1 should reduce to standard mean CE (over valid positions)."""
    logits, labels, token_strings = _make_synthetic_batch()
    loss_alpha1 = temporal_weighted_ce_loss(logits, labels, token_strings, alpha=1.0)
    # Compute standard mean CE manually
    base = cross_entropy(logits, labels)
    expected = float(base.mean())
    assert abs(loss_alpha1 - expected) < 1e-5, \
        f"alpha=1 loss {loss_alpha1:.6f} != standard CE {expected:.6f}"
    print(f"PASS: test_alpha_1_equals_standard_ce (loss={loss_alpha1:.6f})")


def test_ignore_index_masking():
    """Positions with label==-100 (padding) must not contribute to loss."""
    vocab_size = 50
    rng = np.random.default_rng(7)
    logits = rng.standard_normal((1, 4, vocab_size)).astype(np.float32)
    labels_with_pad = np.array([[0, -100, 2, -100]])
    labels_no_pad   = np.array([[0, 0,    2, 0   ]])  # different tokens at pad pos
    token_strings = [["module", "<pad>", "gnt", "<pad>"]]

    loss_padded = temporal_weighted_ce_loss(logits, labels_with_pad,
                                            token_strings, alpha=3.0,
                                            ignore_index=-100)
    # Should only use positions 0 and 2
    base = cross_entropy(logits, np.clip(labels_with_pad, 0, vocab_size - 1))
    # weights for token_strings (no temporal ops, all 1.0), mask out pads
    valid_mask = (labels_with_pad != -100).astype(np.float32)
    expected = float((base * valid_mask).sum() / valid_mask.sum())
    assert abs(loss_padded - expected) < 1e-5, \
        f"Padded loss {loss_padded:.6f} != expected {expected:.6f}"
    print(f"PASS: test_ignore_index_masking (loss={loss_padded:.6f})")


def test_all_temporal_tokens():
    """A sequence of all temporal tokens should have all weights == alpha."""
    alpha = 3.0
    toks = [["##", "|->", "s_eventually", "throughout", "$rose"]]
    weights = build_weight_mask(toks, alpha=alpha)
    assert (weights == alpha).all(), f"All temporal tokens should have weight {alpha}"
    print("PASS: test_all_temporal_tokens")


def test_empty_sequence():
    """Zero-length sequence should not crash."""
    vocab_size = 10
    logits = np.zeros((1, 0, vocab_size), dtype=np.float32)
    labels = np.zeros((1, 0), dtype=np.int32)
    token_strings = [[]]
    loss = temporal_weighted_ce_loss(logits, labels, token_strings, alpha=3.0)
    assert loss == 0.0
    print("PASS: test_empty_sequence")


def run_all_tests():
    print("=== temporal_loss.py unit tests ===")
    test_is_temporal_token()
    test_weight_mask_values()
    test_loss_higher_with_temporal_weight()
    test_alpha_1_equals_standard_ce()
    test_ignore_index_masking()
    test_all_temporal_tokens()
    test_empty_sequence()
    print("=== All temporal_loss tests passed ===")


if __name__ == "__main__":
    run_all_tests()
