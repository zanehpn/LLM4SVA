#!/usr/bin/env python3
"""
Heuristically fill missing/placeholder `nl` fields in a jsonl file containing
SystemVerilog assertion records.

The goal is not perfect paraphrase quality. The goal is to produce a stable,
readable one-sentence description for records whose `nl` is empty, too short,
or an obvious placeholder, without requiring a GPU model.
"""

import argparse
import json
import re
from pathlib import Path


PLACEHOLDER_RE = re.compile(r"^\s*\[(?:ASSERT|ASSUME|COVER)[^\]]*\]\s*$",
                            re.IGNORECASE)
IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_$]*")


def needs_nl(nl: str) -> bool:
    nl = (nl or "").strip()
    return len(nl) < 5 or bool(PLACEHOLDER_RE.match(nl))


def clean_ws(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def split_args(text: str) -> list[str]:
    parts = []
    start = 0
    depth = 0
    for i, ch in enumerate(text):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif ch == "," and depth == 0:
            parts.append(text[start:i].strip())
            start = i + 1
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def extract_balanced(text: str, start_idx: int) -> tuple[str, int] | tuple[None, None]:
    if start_idx < 0 or start_idx >= len(text) or text[start_idx] != "(":
        return None, None
    depth = 0
    in_str = False
    i = start_idx
    while i < len(text):
        ch = text[i]
        if in_str:
            if ch == "\\":
                i += 2
                continue
            if ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    return text[start_idx + 1:i], i
        i += 1
    return None, None


def split_top_level(expr: str, token: str) -> tuple[str, str] | None:
    depth = 0
    i = 0
    while i <= len(expr) - len(token):
        ch = expr[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif depth == 0 and expr[i:i + len(token)] == token:
            return expr[:i].strip(), expr[i + len(token):].strip()
        i += 1
    return None


def prettify_name(name: str) -> str:
    name = re.sub(r"\([^)]*\)", "", name)
    name = re.sub(r"(^|_)(assert|assume|cover|property|prop|check|checker|hold|holds)(?=_|$)", " ", name, flags=re.I)
    name = re.sub(r"(^|_)(p|a|ap|cp)(?=_|$)", " ", name, flags=re.I)
    name = name.replace("$", " ")
    name = re.sub(r"[_]+", " ", name)
    name = re.sub(r"\b\d+\b", lambda m: m.group(0), name)
    name = clean_ws(name)
    return name.lower()


def extract_disable_condition(expr: str) -> tuple[str | None, str]:
    m = re.match(r"\s*disable\s+iff\s*\(", expr, flags=re.I)
    if not m:
        return None, expr.strip()
    inside, end = extract_balanced(expr, m.end() - 1)
    if inside is None:
        return None, expr.strip()
    return inside.strip(), expr[end + 1:].strip()


def describe_signal_expr(expr: str) -> str:
    expr = clean_ws(expr)
    if not expr:
        return "the condition holds"

    expr = re.sub(r"@\s*\([^)]*\)\s*", "", expr)
    expr = re.sub(r"//.*$", "", expr).strip()
    expr = re.sub(r"^\((.*)\)$", r"\1", expr)

    replacements = [
        (r"\$rose\s*\(([^)]+)\)", r"\1 rises"),
        (r"\$fell\s*\(([^)]+)\)", r"\1 falls"),
        (r"\$stable\s*\(([^)]+)\)", r"\1 remains stable"),
        (r"\$changed\s*\(([^)]+)\)", r"\1 changes"),
        (r"!\s*\$past\s*\(([^,()]+)\s*,\s*(\d+)\s*\)", r"\1 was low \2 cycles earlier"),
        (r"\$past\s*\(([^,()]+)\s*,\s*(\d+)\s*\)", r"\1 \2 cycles earlier"),
        (r"\$past\s*\(([^)]+)\)", r"the previous value of \1"),
        (r"!\s*([A-Za-z_][A-Za-z0-9_.$\[\]]*)", r"\1 is low"),
    ]
    for pat, repl in replacements:
        expr = re.sub(pat, repl, expr)

    expr = expr.replace("&&", " and ")
    expr = expr.replace("||", " or ")
    expr = re.sub(r"\bthroughout\b", " throughout ", expr, flags=re.I)
    expr = re.sub(r"\buntil_with\b", " until ", expr, flags=re.I)
    expr = re.sub(r"\bs_until\b", " until ", expr, flags=re.I)
    expr = re.sub(r"\bs_eventually\b", " eventually ", expr, flags=re.I)
    expr = re.sub(r"==", " equals ", expr)
    expr = re.sub(r"!=", " does not equal ", expr)
    expr = re.sub(r">=", " is at least ", expr)
    expr = re.sub(r"<=", " is at most ", expr)
    expr = re.sub(r">", " is greater than ", expr)
    expr = re.sub(r"<", " is less than ", expr)
    expr = clean_ws(expr)
    return expr


def extract_signal_names(text: str) -> list[str]:
    text = re.sub(r'"[^"]*"', " ", text or "")
    names = []
    for name in IDENT_RE.findall(text or ""):
        low = name.lower()
        if low in {
            "assert", "property", "assume", "cover", "posedge", "negedge",
            "disable", "iff", "strong", "weak", "first_match", "intersect",
            "throughout", "until", "until_with", "s_until", "s_eventually",
            "if", "else", "begin", "end", "or", "and", "not", "module",
            "display", "assertion", "failed", "check", "stable", "changed",
            "rose", "fell"
        }:
            continue
        if name.startswith(("d", "h", "b")) and re.fullmatch(r"[dhb][0-9a-fA-F_xzXZ']+", name):
            continue
        if name not in names:
            names.append(name)
        if len(names) >= 5:
            break
    return names


def describe_property_call(expr: str) -> str | None:
    m = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_$]*)\s*(?:\((.*)\))?", clean_ws(expr))
    if not m:
        return None
    name = prettify_name(m.group(1))
    if not name:
        return None
    args = split_args(m.group(2) or "")
    if args:
        sigs = ", ".join(clean_ws(a) for a in args[:3])
        return f"that the {name} condition holds for {sigs}."
    return f"that the {name} condition holds."


def describe_assertion_body(expr: str) -> str:
    disable_cond, core = extract_disable_condition(expr)

    prop_desc = describe_property_call(core)
    if prop_desc:
        if disable_cond:
            return prop_desc[:-1] + f" unless {describe_signal_expr(disable_cond)}."
        return prop_desc

    for arrow, timing in (("|=>", "in the next cycle"), ("|->", "in the same cycle")):
        parts = split_top_level(core, arrow)
        if not parts:
            continue
        lhs, rhs = parts
        rhs_delay = re.search(r"##\[(\d+):\$\]", rhs)
        rhs_fixed = re.search(r"##\s*(\d+)", rhs)
        if rhs_delay:
            when = f"within at least {rhs_delay.group(1)} cycles and eventually"
        elif rhs_fixed:
            when = f"after {rhs_fixed.group(1)} cycles"
        else:
            when = timing
        desc = (f"when {describe_signal_expr(lhs)}, "
                f"{describe_signal_expr(rhs)} must hold {when}.")
        if disable_cond:
            desc += f" Evaluation stops when {describe_signal_expr(disable_cond)}."
        return clean_ws(desc)

    if "##[" in core:
        desc = f"that the sequence {describe_signal_expr(core)} occurs with a ranged delay."
    elif "##" in core:
        desc = f"that the sequence {describe_signal_expr(core)} occurs with a cycle delay."
    else:
        desc = f"that {describe_signal_expr(core)}."
    if disable_cond:
        desc += f" Evaluation stops when {describe_signal_expr(disable_cond)}."
    return clean_ws(desc)


def fill_one(record: dict) -> str:
    sva = clean_ws(record.get("reference_sva", ""))
    if not sva:
        return ""

    label_match = re.match(r"\s*([A-Za-z_][A-Za-z0-9_$]*)\s*:", sva)
    label_desc = prettify_name(label_match.group(1)) if label_match else ""
    kind = "assert"
    if re.search(r"\bassume\s+property\b", sva, flags=re.I):
        kind = "assume"
    elif re.search(r"\bcover\s+property\b", sva, flags=re.I):
        kind = "cover"

    prop_idx = re.search(r"\b(?:assert|assume|cover)\s+property\b", sva, flags=re.I)
    if prop_idx:
        open_idx = sva.find("(", prop_idx.end())
        inside, _ = extract_balanced(sva, open_idx)
    else:
        inside = None

    if inside is None:
        inside = sva

    base = describe_assertion_body(inside)

    prefix = ""
    if kind == "assume":
        prefix = "Assume "
        if base.startswith("that "):
            base = base[5:]
    elif kind == "cover":
        prefix = "Cover the case where "
        if base.startswith("that "):
            base = base[5:]
        elif base.startswith("when "):
            base = base[5:]

    if label_desc and label_desc not in base.lower():
        base = base[:-1] + f" This corresponds to {label_desc}."

    signal_text = re.split(r"\belse\b", inside or sva, maxsplit=1, flags=re.I)[0]
    signal_text = re.sub(r"//.*$", "", signal_text).strip()
    sigs = extract_signal_names(signal_text)
    if sigs:
        if base[-1] not in ".!?":
            base += "."
        base += f" Use signals {', '.join(sigs[:5])}."

    base = prefix + base
    base = clean_ws(base)
    if base and base[-1] not in ".!?":
        base += "."
    return base


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-path", required=True)
    ap.add_argument("--out", required=True,
                    help="output jsonl path; can be the same as --in-path")
    ap.add_argument("--limit", type=int, default=0,
                    help="only fill the first N missing records")
    ap.add_argument("--marker", default="heuristic-fill-v1")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for line in in_path.open():
        records.append(json.loads(line))

    missing = [i for i, r in enumerate(records) if needs_nl(r.get("nl", ""))]
    if args.limit:
        missing = missing[:args.limit]

    print(f"[fill-nl-heuristic] total={len(records)} missing={len(missing)}")
    for i, idx in enumerate(missing, 1):
        records[idx]["nl"] = fill_one(records[idx])
        records[idx]["nl_filled_by"] = args.marker
        if i % 200 == 0 or i == len(missing):
            print(f"  filled {i}/{len(missing)}")

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp_path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    tmp_path.replace(out_path)
    print(f"[fill-nl-heuristic] wrote {out_path}")


if __name__ == "__main__":
    main()
