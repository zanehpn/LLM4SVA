"""
Lightweight SVA parser (regex-based).

Extracts structural fields from a SystemVerilog assertion string:
  - property_name
  - clock_expr
  - disable_iff
  - body (the property expression)
  - temporal_operators (list of detected op strings)
  - bound (assert / cover / assume)

No tree-sitter dependency; pure regex over SVA text.
"""

import re
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ParsedSVA:
    raw: str
    property_name: Optional[str] = None
    clock_expr: Optional[str] = None
    disable_iff: Optional[str] = None
    body: Optional[str] = None
    bound: str = "assert"
    temporal_operators: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "property_name": self.property_name,
            "clock_expr": self.clock_expr,
            "disable_iff": self.disable_iff,
            "body": self.body,
            "bound": self.bound,
            "temporal_operators": self.temporal_operators,
        }


# Patterns for temporal operator extraction
_TEMPORAL_PATTERNS = [
    (re.compile(r'\bs_eventually\b', re.I), 's_eventually'),
    (re.compile(r'\bs_until\b', re.I), 's_until'),
    (re.compile(r'\bs_always\b', re.I), 's_always'),
    (re.compile(r'\buntil_with\b', re.I), 'until_with'),
    (re.compile(r'\bstrong\s*\(', re.I), 'strong'),
    (re.compile(r'\|->'), '|->'),
    (re.compile(r'\|=>'), '|=>'),
    (re.compile(r'\bthroughout\b', re.I), 'throughout'),
    (re.compile(r'\bwithin\b', re.I), 'within'),
    (re.compile(r'\bintersect\b', re.I), 'intersect'),
    (re.compile(r'\bfirst_match\b', re.I), 'first_match'),
    (re.compile(r'##\s*\[\s*\d+\s*:\s*(?:\d+|\$)\s*\]'), '##[a:b]'),
    (re.compile(r'\[\s*\*\s*\d+\s*:\s*(?:\d+|\$)\s*\]'), '[*a:b]'),
    (re.compile(r'\[\s*=\s*\d+\s*:\s*(?:\d+|\$)\s*\]'), '[=a:b]'),
    (re.compile(r'\[\s*-\s*>\s*\d+\s*:\s*(?:\d+|\$)\s*\]'), '[->a:b]'),
    (re.compile(r'##\s*\d+'), '##N'),
    (re.compile(r'\[\s*\*\s*\d+\s*\]'), '[*N]'),
    (re.compile(r'\[\s*=\s*\d+\s*\]'), '[=N]'),
    (re.compile(r'\$rose\b'), '$rose'),
    (re.compile(r'\$fell\b'), '$fell'),
    (re.compile(r'\$stable\b'), '$stable'),
    (re.compile(r'\$past\b'), '$past'),
]


def _strip_comments(s: str) -> str:
    s = re.sub(r'//[^\n]*', ' ', s)
    s = re.sub(r'/\*.*?\*/', ' ', s, flags=re.DOTALL)
    return s


def parse_sva(sva_text: str) -> ParsedSVA:
    """Parse a single SVA string into a ParsedSVA object."""
    result = ParsedSVA(raw=sva_text)

    clean = _strip_comments(sva_text)

    # Bound type
    if re.search(r'\bcover\b', clean, re.I):
        result.bound = "cover"
    elif re.search(r'\bassume\b', clean, re.I):
        result.bound = "assume"
    else:
        result.bound = "assert"

    # Property name — pattern: property <name>
    m = re.search(r'\bproperty\s+(\w+)\s*\(', clean, re.I)
    if m:
        result.property_name = m.group(1)

    # Clock expression — @(posedge/negedge clk)
    m = re.search(r'@\s*\(\s*((?:posedge|negedge)\s+\w+)\s*\)', clean, re.I)
    if m:
        result.clock_expr = m.group(1).strip()

    # disable iff (...)
    m = re.search(r'disable\s+iff\s*\(([^)]+)\)', clean, re.I)
    if m:
        result.disable_iff = m.group(1).strip()

    # Body: text inside the outermost assert/cover property (...) after clock
    # Heuristic: take everything after the clock expression up to the final );
    body_match = re.search(r'@\s*\([^)]+\)\s*(.*?)\s*;?\s*$', clean, re.DOTALL)
    if body_match:
        result.body = body_match.group(1).strip().rstrip(');').strip()

    # Extract temporal operators (unique, in order of specificity)
    seen = set()
    for pat, name in _TEMPORAL_PATTERNS:
        if pat.search(clean) and name not in seen:
            result.temporal_operators.append(name)
            seen.add(name)

    return result


def parse_batch(sva_list: list) -> list:
    return [parse_sva(s) for s in sva_list]


if __name__ == "__main__":
    tests = [
        "assert property (@(posedge clk) req |-> ##[1:3] gnt);",
        "assert property p_req_gnt (@(posedge clk) disable iff (rst) req |-> s_eventually gnt);",
        "cover property (@(negedge clk) a ##1 b ##1 c);",
    ]
    for t in tests:
        p = parse_sva(t)
        print(p.to_dict())
        print()
