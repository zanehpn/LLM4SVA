"""
Mock Formal Verifier for TemporalSVA pilot experiments.

IMPORTANT: This is a MOCK — it does NOT perform real formal verification.
Real verification requires SymbiYosys or JasperGold, which are not available
in this environment. This mock performs:
  1. Syntax checks (regex-based SVA structure validation)
  2. Returns structured pass/fail/cex stubs
  3. Documents exactly what is and is not checked

SCALE-BLOCKED: Real RLVF training requires a real verifier.
"""

import re
import hashlib
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class VerificationResult:
    status: str          # "PASS" | "FAIL_SYNTAX" | "FAIL_VERIFICATION" | "TIMEOUT"
    is_mock: bool = True
    confidence: str = "LOW"          # Mock always LOW
    syntax_ok: bool = False
    counterexample: Optional[Dict] = None
    vacuous: Optional[bool] = None   # Cannot determine without real verifier
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "is_mock": self.is_mock,
            "confidence": self.confidence,
            "syntax_ok": self.syntax_ok,
            "counterexample": self.counterexample,
            "vacuous": self.vacuous,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Syntax checks (heuristic, not a real parser)
# ---------------------------------------------------------------------------

_REQUIRED_KEYWORDS_ANY = [
    re.compile(r'\bassert\s+property\b', re.I),
    re.compile(r'\bcover\s+property\b', re.I),
    re.compile(r'\bassume\s+property\b', re.I),
    re.compile(r'\bproperty\b', re.I),
]

_CLOCK_PATTERN = re.compile(r'@\s*\(\s*(?:posedge|negedge)\s+\w+', re.I)
_SEMICOLON_END = re.compile(r';\s*$', re.M)
_UNBALANCED_PARENS = None  # checked procedurally


def _check_balanced_parens(s: str) -> bool:
    depth = 0
    for ch in s:
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def syntax_check(sva: str) -> Dict[str, Any]:
    """
    Heuristic syntax check on SVA string.

    Returns dict with:
        ok: bool
        issues: list of str (detected problems)
        warnings: list of str
    """
    issues = []
    warnings = []

    # Strip comments
    clean = re.sub(r'//[^\n]*', ' ', sva)
    clean = re.sub(r'/\*.*?\*/', ' ', clean, flags=re.DOTALL)

    # Must have assert/cover/assume property OR bare property keyword
    has_keyword = any(p.search(clean) for p in _REQUIRED_KEYWORDS_ANY)
    if not has_keyword:
        issues.append("Missing 'assert property', 'cover property', 'assume property', or 'property' keyword")

    # Should have a clock
    if not _CLOCK_PATTERN.search(clean):
        warnings.append("No clock expression (@posedge/@negedge) detected")

    # Should end with semicolon
    if not _SEMICOLON_END.search(clean):
        issues.append("Missing trailing semicolon")

    # Balanced parentheses
    if not _check_balanced_parens(clean):
        issues.append("Unbalanced parentheses")

    # Operator sanity: |-> or |=> must have something on both sides
    for op in ['|->', '|=>']:
        if op in clean:
            parts = clean.split(op)
            if len(parts) < 2 or not parts[0].strip() or not parts[1].strip():
                issues.append(f"Operator '{op}' appears to be missing antecedent or consequent")

    # Warn on common typos
    if re.search(r'#\s*#', clean):
        warnings.append("Possible typo: '# #' (space in ##)")
    if re.search(r'\|\s*->', clean):
        warnings.append("Possible typo: '| ->' (space in |->)")

    return {"ok": len(issues) == 0, "issues": issues, "warnings": warnings}


# ---------------------------------------------------------------------------
# Mock verifier
# ---------------------------------------------------------------------------

class MockVerifier:
    """
    Mock verifier that only checks syntax.

    SCALE-BLOCKED:
        Real RLVF training reward = 0.15×syntax + 0.40×formal_verify +
        0.20×(1-vacuity) + 0.25×sim_score
        This mock can only compute the syntax component (0.15).
        Used as a stand-in reward source for the GRPO group-advantage demo
        in scripts/run_rlvf.py --demo (see FINAL_PROPOSAL.md Component 3).
    """

    def __init__(self):
        self._cache: Dict[str, VerificationResult] = {}
        self.call_count = 0

    def _cache_key(self, sva: str, rtl: Optional[str] = None) -> str:
        content = (sva or "") + "|||" + (rtl or "")
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def verify(
        self,
        sva: str,
        rtl: Optional[str] = None,
        timeout_s: int = 30,
    ) -> VerificationResult:
        """
        Mock verification. Only syntax is checked; formal result is stubbed.

        Args:
            sva:       SVA string to verify
            rtl:       RTL context (ignored in mock)
            timeout_s: timeout (ignored in mock)
        """
        key = self._cache_key(sva, rtl)
        if key in self._cache:
            return self._cache[key]

        self.call_count += 1
        syn = syntax_check(sva)

        if not syn["ok"]:
            result = VerificationResult(
                status="FAIL_SYNTAX",
                syntax_ok=False,
                counterexample=None,
                vacuous=None,
                notes=(
                    "MOCK: Syntax check failed. Issues: " +
                    "; ".join(syn["issues"]) +
                    ". NOTE: Real formal verification not performed (SymbiYosys unavailable)."
                ),
            )
        else:
            # Syntax passed — stub the formal result
            # We cannot actually verify without SymbiYosys/JasperGold.
            result = VerificationResult(
                status="PASS",
                syntax_ok=True,
                counterexample=None,
                vacuous=None,
                notes=(
                    "MOCK: Syntax check passed. "
                    "SCALE-BLOCKED: Real formal verification requires SymbiYosys or JasperGold. "
                    "RLVF training (Component 3) is blocked without a real verifier. "
                    "This stub always returns PASS for syntactically valid SVAs."
                ),
            )
            if syn["warnings"]:
                result.notes += " Warnings: " + "; ".join(syn["warnings"])

        self._cache[key] = result
        return result

    def mock_reward(self, sva: str, rtl: Optional[str] = None) -> Dict[str, float]:
        """
        Compute partial RLVF reward (syntax component only).

        Paper reward: 0.15×syntax + 0.40×formal + 0.20×(1-vacuity) + 0.25×sim
        Mock:         0.15×syntax + 0.00×formal + 0.00×vacuity + 0.00×sim
                      (formal/vacuity/sim all blocked)
        """
        res = self.verify(sva, rtl)
        w_syn = 0.15
        r_syn = 1.0 if res.syntax_ok else 0.0
        return {
            "syntax":     r_syn,
            "formal":     None,   # SCALE-BLOCKED
            "vacuity":    None,   # SCALE-BLOCKED
            "similarity": None,   # SCALE-BLOCKED
            "total_mock": w_syn * r_syn,
            "total_max":  w_syn,  # max achievable in mock mode
            "blocked_components": ["formal_verify", "vacuity_check", "similarity"],
        }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    mv = MockVerifier()

    test_cases = [
        ("assert property (@(posedge clk) req |-> ##1 gnt);", "good"),
        ("req |-> gnt",                                        "missing keyword + clock"),
        ("assert property (@(posedge clk) req |-> gnt",        "missing semicolon"),
        ("assert property (@(posedge clk) req |-> ##[1:3] gnt);", "good with range"),
        ("assert property posedge clk req |-> gnt);",          "unbalanced parens"),
    ]

    for sva, label in test_cases:
        res = mv.verify(sva)
        reward = mv.mock_reward(sva)
        print(f"[{label}]")
        print(f"  Status: {res.status}, syntax_ok={res.syntax_ok}")
        print(f"  Reward (mock): {reward['total_mock']:.2f}/{reward['total_max']:.2f}")
        print(f"  Notes: {res.notes[:100]}...")
        print()
