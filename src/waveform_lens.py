"""
WaveformLens: Cycle-precise temporal constraint extraction from waveform pairs.

Paper spec (FINAL_PROPOSAL.md Component 4):
    extract_temporal_constraints(expected, actual, signals_of_interest)
    → list of constraint dicts with:
        {signal, expected_cycle, actual_cycle, expected_type, violation, delta}

Inputs:
    expected / actual: dict mapping signal_name → list of (cycle, value) or
                       dict mapping signal_name → list of int values
                       (index = cycle number starting from 0)

    A "transition" is any change in signal value between consecutive cycles.
"""

from typing import List, Dict, Any, Optional, Tuple


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def find_transitions(signal_trace: List[int]) -> List[Tuple[int, str]]:
    """
    Find all value transitions in a signal trace.

    Args:
        signal_trace: list of int values, index = cycle

    Returns:
        list of (cycle, transition_type) where transition_type is:
            "rising"  — 0 → 1 (or low → high)
            "falling" — 1 → 0 (or high → low)
            "change"  — any other value change
    """
    transitions = []
    for i in range(1, len(signal_trace)):
        prev, curr = signal_trace[i - 1], signal_trace[i]
        if prev == curr:
            continue
        if prev == 0 and curr != 0:
            ttype = "rising"
        elif prev != 0 and curr == 0:
            ttype = "falling"
        else:
            ttype = "change"
        transitions.append((i, ttype))
    return transitions


def find_closest(
    act_transitions: List[Tuple[int, str]],
    exp_cycle: int,
    exp_type: str,
    window: int = 10,
) -> Optional[Tuple[int, str]]:
    """
    Find the closest actual transition of matching type to the expected one.
    Search within ±window cycles.
    """
    candidates = [
        (abs(act_cycle - exp_cycle), act_cycle, act_type)
        for act_cycle, act_type in act_transitions
        if act_type == exp_type and abs(act_cycle - exp_cycle) <= window
    ]
    if not candidates:
        return None
    candidates.sort()
    _, best_cycle, best_type = candidates[0]
    return (best_cycle, best_type)


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------

def extract_temporal_constraints(
    expected: Dict[str, List[int]],
    actual: Dict[str, List[int]],
    signals_of_interest: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Compare expected and actual waveforms, extract temporal constraint violations.

    Args:
        expected:             {signal: [value_at_cycle_0, value_at_cycle_1, ...]}
        actual:               same format
        signals_of_interest:  which signals to check; if None, checks all in expected

    Returns:
        List of constraint dicts. Each dict has:
            signal          — signal name
            expected_cycle  — cycle where transition was expected
            expected_type   — "rising" / "falling" / "change"
            violation       — "missing" / "delayed" / "early"
            actual_cycle    — (if not missing) cycle where transition actually occurred
            delta           — (if not missing) actual_cycle - expected_cycle
    """
    if signals_of_interest is None:
        signals_of_interest = list(expected.keys())

    constraints = []

    for signal in signals_of_interest:
        exp_trace = expected.get(signal, [])
        act_trace = actual.get(signal, [])

        if not exp_trace:
            continue

        exp_transitions = find_transitions(exp_trace)
        act_transitions = find_transitions(act_trace) if act_trace else []

        for exp_cycle, exp_type in exp_transitions:
            matched = find_closest(act_transitions, exp_cycle, exp_type)

            if matched is None:
                constraints.append({
                    "signal": signal,
                    "expected_cycle": exp_cycle,
                    "expected_type": exp_type,
                    "violation": "missing",
                    "actual_cycle": None,
                    "delta": None,
                })
            elif matched[0] != exp_cycle:
                delta = matched[0] - exp_cycle
                constraints.append({
                    "signal": signal,
                    "expected_cycle": exp_cycle,
                    "expected_type": exp_type,
                    "violation": "delayed" if delta > 0 else "early",
                    "actual_cycle": matched[0],
                    "delta": delta,
                })
            # else: perfect match — no constraint generated

    return constraints


def format_constraints_for_prompt(constraints: List[Dict[str, Any]]) -> str:
    """Format constraint list as bullet points for the WaveformLens repair prompt."""
    if not constraints:
        return "No temporal violations detected."
    lines = []
    for c in constraints:
        sig = c["signal"]
        vtype = c["violation"]
        exp_c = c["expected_cycle"]
        exp_t = c["expected_type"]
        if vtype == "missing":
            lines.append(
                f"  - Signal '{sig}': expected {exp_t} transition at cycle {exp_c}, "
                f"but no matching transition found in actual waveform."
            )
        elif vtype == "delayed":
            lines.append(
                f"  - Signal '{sig}': {exp_t} transition expected at cycle {exp_c}, "
                f"occurred {c['delta']} cycle(s) LATE (actual cycle {c['actual_cycle']})."
            )
        elif vtype == "early":
            lines.append(
                f"  - Signal '{sig}': {exp_t} transition expected at cycle {exp_c}, "
                f"occurred {abs(c['delta'])} cycle(s) EARLY (actual cycle {c['actual_cycle']})."
            )
    return "\n".join(lines)


def build_repair_prompt(sva_draft: str, constraints: List[Dict[str, Any]],
                        signal_defs: str = "") -> str:
    """Build the WaveformLens LLM repair prompt from paper spec."""
    constraint_text = format_constraints_for_prompt(constraints)
    prompt = f"""The following SVA failed formal verification:
{sva_draft}

Counterexample analysis reveals these temporal violations:
{constraint_text}
"""
    if signal_defs:
        prompt += f"\nSignal definitions in RTL:\n{signal_defs}\n"
    prompt += (
        "\nPlease rewrite the SVA to be consistent with the temporal constraints above.\n"
        "Pay special attention to clock-cycle delays (## N) and sequence operators."
    )
    return prompt


# ---------------------------------------------------------------------------
# Unit tests / demo
# ---------------------------------------------------------------------------

def _run_demo():
    # Waveform pair 1: req→gnt with 2-cycle delay
    expected_1 = {
        "req": [0, 1, 1, 1, 1],
        "gnt": [0, 0, 1, 1, 0],  # gnt rises at cycle 2
    }
    actual_1 = {
        "req": [0, 1, 1, 1, 1],
        "gnt": [0, 0, 0, 0, 1],  # gnt rises at cycle 4 (2 cycles late)
    }
    c1 = extract_temporal_constraints(expected_1, actual_1, ["req", "gnt"])
    print("=== Waveform pair 1: gnt delayed ===")
    print(format_constraints_for_prompt(c1))
    print()

    # Waveform pair 2: missing transition entirely
    expected_2 = {"valid": [0, 1, 0, 0, 0]}
    actual_2   = {"valid": [0, 0, 0, 0, 0]}  # valid never goes high
    c2 = extract_temporal_constraints(expected_2, actual_2, ["valid"])
    print("=== Waveform pair 2: valid missing ===")
    print(format_constraints_for_prompt(c2))
    print()

    # Waveform pair 3: early transition
    expected_3 = {"done": [0, 0, 0, 1, 0]}
    actual_3   = {"done": [0, 0, 1, 0, 0]}  # done rises 1 cycle early
    c3 = extract_temporal_constraints(expected_3, actual_3, ["done"])
    print("=== Waveform pair 3: done early ===")
    print(format_constraints_for_prompt(c3))


if __name__ == "__main__":
    _run_demo()
