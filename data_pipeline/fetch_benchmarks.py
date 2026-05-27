#!/usr/bin/env python3
"""
fetch_benchmarks.py — pull external SVA benchmarks, normalize to the
unified schema in data/SPLIT.md, route to data/train/ or data/test/, and
enforce a strict no-contamination rule between splits.

Split policy (see data/SPLIT.md):
  TEST  (held-out, required for SoTA comparison):
    - nl2sva_human       (FVEval, 79 CSV rows)
    - assertionbench     (NAACL 2025, 18 design packs with .gold assertions)
  TRAIN:
    - nl2sva_machine     (FVEval, 300 CSV rows)
    - fveval_design2sva  (FVEval, RTL-grounded SVA tasks)
    - cvdp               (NVIDIA, filtered to SVA/assertion-relevant records)
    - handcrafted_pilot  (existing pilot data in data/*.json)

Unified schema:
  {id, source, nl, reference_sva, rtl_context, expected_tcl, cex, split, hash}

Usage:
  python3 scripts/fetch_benchmarks.py --fetch           # git clone + HF download
  python3 scripts/fetch_benchmarks.py --build --all     # normalize + route
  python3 scripts/fetch_benchmarks.py --contamination-check
"""

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = EXPERIMENTS_DIR / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
RAW_DIR = DATA_DIR / "raw"
TEST_HASHES = TEST_DIR / "_hashes.json"

sys.path.insert(0, str(EXPERIMENTS_DIR))
from src.tcl_classifier import TCLClassifier  # noqa: E402


# -----------------------------------------------------------------------------
# Source registry
# -----------------------------------------------------------------------------
SOURCES: Dict[str, dict] = {
    # ---- TEST SETS ----
    "nl2sva_human": {
        "split": "test",
        "role": "Primary SoTA benchmark — CodeV-SVA-14B reports 75.8% here",
        "fetcher": {"kind": "git", "url": "https://github.com/NVlabs/FVEval.git",
                    "target_dir": "FVEval"},
        "raw_path": "FVEval/data_nl2sva/data/nl2sva_human.csv",
        "loader": "load_nl2sva_csv",
    },
    "assertionbench": {
        "split": "test",
        "role": "WaveformLens repair eval — 100 OpenCores designs, verified assertions",
        "fetcher": {"kind": "git",
                    "url": "https://github.com/achieve-lab/assertion_data_for_LLM.git",
                    "target_dir": "assertion_data_for_LLM"},
        "raw_path": "assertion_data_for_LLM/verified_assertions",
        "loader": "load_assertionbench",
    },

    # ---- TRAIN SETS ----
    "nl2sva_machine": {
        "split": "train",
        "role": "Large NL→SVA pair corpus (GPT-generated), 300 rows",
        "fetcher": {"kind": "git", "url": "https://github.com/NVlabs/FVEval.git",
                    "target_dir": "FVEval"},  # shared clone with nl2sva_human
        "raw_path": "FVEval/data_nl2sva/data/nl2sva_machine.csv",
        "loader": "load_nl2sva_csv",
    },
    "fveval_design2sva": {
        "split": "train",
        "role": "FVEval Design2SVA — RTL-grounded SVA generation",
        "fetcher": {"kind": "git", "url": "https://github.com/NVlabs/FVEval.git",
                    "target_dir": "FVEval"},
        "raw_path": "FVEval/data_design2sva",
        "loader": "load_design2sva",
    },
    "cvdp": {
        "split": "train",
        "role": "NVIDIA CVDP — filtered to SVA/assertion-relevant records",
        "fetcher": {"kind": "hf_dataset",
                    "repo_id": "nvidia/cvdp-benchmark-dataset",
                    "target_dir": "cvdp-benchmark-dataset"},
        "raw_path": "cvdp-benchmark-dataset",
        "loader": "load_cvdp",
    },
    "handcrafted_pilot": {
        "split": "train",
        "role": "Existing hand-crafted pilot SVAs in data/*.json",
        "fetcher": None,
        "raw_path": None,
        "loader": "load_handcrafted",
    },
    "github_scraped": {
        "split": "train",
        "role": "SVAs scraped from 15 open-source RTL repos (stars > 100)",
        "fetcher": None,  # handled by scripts/scrape_github_sva.py
        "raw_path": "github_scraped/scraped.jsonl",
        "loader": "load_github_scraped",
    },
    "opentitan_macros": {
        "split": "train",
        "role": "Expanded `ASSERT/`ASSERT_NEVER/`ASSERT_KNOWN/`COVER/`ASSUME macros "
                "from OpenTitan et al. (see scripts/expand_opentitan_macros.py)",
        "fetcher": None,
        "raw_path": "opentitan_macros/expanded.jsonl",
        "loader": "load_opentitan_macros",
    },
    "named_properties": {
        "split": "train",
        "role": "Named `property NAME; ... endproperty` blocks — cva6, cv32e40p/x, "
                "Caliptra-RTL, Surelog, Verilator testbench style",
        "fetcher": None,
        "raw_path": "named_properties/scraped.jsonl",
        "loader": "load_named_properties",
    },
    "snapshots_scraped": {
        "split": "train",
        "role": "Three-way scrape over 621 PR snapshots at "
                "${SVA_CORPUS_ROOT}/{HWE-train,HDL_all}/snapshots — "
                "captures inline + macro + named-property SVAs from many "
                "repos / versions our github_repos/ doesn't cover",
        "fetcher": None,
        "raw_path": "snapshots_scraped/scraped.jsonl",
        "loader": "load_snapshots_scraped",
    },
    "github_search_scraped": {
        "split": "train",
        "role": "Clone-scrape-delete over 39 GH-code-search candidates "
                "(stars>100). diffblue/hw-cbmc & AutoSVA are biggest.",
        "fetcher": None,
        "raw_path": "github_search_scraped/scraped.jsonl",
        "loader": "load_snapshots_scraped",
    },
}


# -----------------------------------------------------------------------------
# Core utilities
# -----------------------------------------------------------------------------
def sample_hash(nl: str, sva: str) -> str:
    content = (nl or "").strip() + "|||" + (sva or "").strip()
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def normalize_sample(orig_id, source, nl, reference_sva, rtl_context="",
                     cex=None, split="train", clf=None):
    level, _ = clf.classify(reference_sva)
    return {
        "id": f"{source}_{orig_id}",
        "source": source,
        "nl": nl or "",
        "reference_sva": reference_sva,
        "rtl_context": rtl_context or "",
        "expected_tcl": int(level),
        "cex": cex,
        "split": split,
        "hash": sample_hash(nl or "", reference_sva),
    }


# -----------------------------------------------------------------------------
# Fetchers
# -----------------------------------------------------------------------------
def fetch_one(name: str, cfg: dict) -> bool:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    f = cfg.get("fetcher")
    if not f:
        return True  # nothing to fetch (e.g. handcrafted)

    target = RAW_DIR / f["target_dir"]
    if target.exists():
        print(f"  [fetch:{name}] already present at {target}")
        return True

    if f["kind"] == "git":
        print(f"  [fetch:{name}] git clone {f['url']} → {target}")
        rc = subprocess.call(["git", "clone", "--depth", "1", f["url"], str(target)])
        return rc == 0
    if f["kind"] == "hf_dataset":
        print(f"  [fetch:{name}] huggingface-cli download {f['repo_id']} → {target}")
        rc = subprocess.call([
            "hf", "download", "--repo-type", "dataset",
            f["repo_id"], "--local-dir", str(target),
        ])
        return rc == 0
    print(f"  [fetch:{name}] unknown kind '{f['kind']}' — skipping")
    return False


# -----------------------------------------------------------------------------
# Per-source loaders — yield normalized dicts
# -----------------------------------------------------------------------------
def load_nl2sva_csv(name: str, raw_path: Path, split: str, clf: TCLClassifier):
    """FVEval NL2SVA-{Human,Machine} CSV:
       design_name, task_id, prompt, ref_solution, testbench

       FVEval `prompt` rows are sentence fragments meant to be wrapped by
       `SVAGEN_QUESTION_PREAMBLE` ("Create a SVA assertion that checks: ").
       We materialize the wrapped, self-contained NL here so downstream
       consumers (eval scripts, training loaders) don't need to know the
       FVEval framing convention."""
    if not raw_path.exists():
        return
    with open(raw_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            sva = (row.get("ref_solution") or "").strip()
            if not sva:
                continue
            # Strip leading 'asrt:' label that appears in some rows
            sva = re.sub(r"^[a-zA-Z_]\w*\s*:\s*", "", sva, count=1)
            raw_nl = (row.get("prompt") or "").strip()
            if raw_nl and not re.match(
                r"(?i)^\s*(create|generate|write)\s+(an?\s+|the\s+)?sva\b",
                raw_nl,
            ):
                nl = f"Create a SVA assertion that checks: {raw_nl}"
            else:
                nl = raw_nl
            yield normalize_sample(
                orig_id=f"{row.get('design_name','?')}_{row.get('task_id','?')}",
                source=name,
                nl=nl,
                reference_sva=sva,
                rtl_context=row.get("testbench", ""),
                split=split,
                clf=clf,
            )


def load_design2sva(name: str, raw_path: Path, split: str, clf: TCLClassifier):
    """FVEval Design2SVA — layout: data_design2sva/data/{...}.csv or per-design dirs.
       We discover CSVs under the tree and read rows with (prompt, ref_solution,
       testbench) fields where available."""
    if not raw_path.exists():
        return
    for csv_path in raw_path.rglob("*.csv"):
        try:
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    sva = (row.get("ref_solution") or row.get("solution")
                           or row.get("assertion") or "").strip()
                    if not sva:
                        continue
                    sva = re.sub(r"^[a-zA-Z_]\w*\s*:\s*", "", sva, count=1)
                    yield normalize_sample(
                        orig_id=f"{csv_path.stem}_{i}",
                        source=name,
                        nl=row.get("prompt", "") or row.get("nl", ""),
                        reference_sva=sva,
                        rtl_context=row.get("testbench", "") or row.get("rtl", ""),
                        split=split,
                        clf=clf,
                    )
        except Exception as e:
            print(f"  [{name}] skip {csv_path.name}: {e}")


# AssertionBench .gold files come in two flavors:
# (A) proper SV property blocks (predominant in *_filtered.gold):
#       property a45;
#       @(posedge clk) (LHS) |-> (RHS);
#       endproperty
#       assert_a45: assert property(a45);
# (B) report-style lines:
#       a1: (LHS) |-> (RHS)
#       IRank: : 0.00968
# We parse both. Form (A) is preferred because it yields real concurrent SVAs.
_PROP_BLOCK_RE = re.compile(
    r"property\s+([a-zA-Z_]\w*)\s*;\s*(.*?)\s*endproperty",
    re.DOTALL | re.IGNORECASE,
)
_REPORT_LINE_RE = re.compile(r"^a\d+\s*:\s*(.+)$")


def _parse_gold_file(path: Path) -> List[tuple]:
    """Return list of (kind, body) pairs from a .gold file.
    kind == 'property' → body is the complete property expression incl. clock
    kind == 'report'   → body is raw expression without clocking"""
    out = []
    if not path.exists():
        return out
    try:
        text = path.read_text(errors="ignore")
    except Exception:
        return out

    # Form A — property ... endproperty blocks
    for m in _PROP_BLOCK_RE.finditer(text):
        body = re.sub(r"\s+", " ", m.group(2)).strip().rstrip(";").strip()
        if body:
            out.append(("property", body))

    if out:
        return out

    # Form B — report-style lines (only when no property blocks found)
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if " :: " in line or line.startswith("Report"):
            continue
        m = _REPORT_LINE_RE.match(line)
        if m:
            body = m.group(1).strip()
            if body and not re.match(r"^[\d.]+$", body):
                out.append(("report", body))
    return out


def _find_rtl_for(design_dir: Path) -> str:
    """Concatenate .v files in the sub-design dir for RTL context."""
    chunks = []
    for v in list(design_dir.glob("*.v"))[:3]:
        try:
            chunks.append(f"// === {v.name} ===\n" + v.read_text(errors="ignore"))
        except Exception:
            pass
    return "\n".join(chunks)


def load_assertionbench(name: str, raw_path: Path, split: str, clf: TCLClassifier):
    """
    AssertionBench layout:
      verified_assertions/<family>/<sub_design>/<file>.gold
      verified_assertions/<family>/<sub_design>/<file>.v
    We wrap each raw assertion body into SVA form:
      assert property (@(posedge clk) <body>);
    """
    if not raw_path.exists():
        return
    count = 0
    # Prefer *_filtered.gold (proper property blocks); fall back to any .gold.
    gold_files = sorted(raw_path.rglob("*_filtered.gold"))
    if not gold_files:
        gold_files = sorted(raw_path.rglob("*.gold"))

    for gold in gold_files:
        family = gold.parent.parent.name
        sub = gold.parent.name
        asserts = _parse_gold_file(gold)
        if not asserts:
            continue
        rtl = _find_rtl_for(gold.parent)
        for k, (kind, body) in enumerate(asserts):
            if kind == "property":
                # Body already contains @(posedge clk) ... |-> ...
                sva = f"assert property ({body});"
            else:
                # Report-style: wrap with a default clock
                sva = f"assert property (@(posedge clk) {body});"
            yield normalize_sample(
                orig_id=f"{family}__{sub}__{gold.stem}__{k}",
                source=name,
                nl="",  # AssertionBench does not provide NL
                reference_sva=sva,
                rtl_context=rtl,
                split=split,
                clf=clf,
            )
            count += 1
    if count == 0:
        print(f"  [{name}] 0 assertions parsed — check .gold format")


_CVDP_SVA_KEYWORDS = re.compile(
    r"\b(?:SVA|SystemVerilog\s+[Aa]ssertion|assert\s+property|cover\s+property|"
    r"assume\s+property|formal\s+verification|property\s+block)\b"
)


def load_cvdp(name: str, raw_path: Path, split: str, clf: TCLClassifier):
    """CVDP — iterate all v1.0.4 jsonl, keep records whose prompt or response
    mentions SVA/assertion keywords. Extract reference SVA from response."""
    if not raw_path.exists():
        return
    jsonls = sorted(raw_path.glob("cvdp_v1.0.4_*.jsonl"))
    if not jsonls:
        return
    for jp in jsonls:
        with open(jp) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                prompt = (rec.get("input") or {}).get("prompt", "") or ""
                out = rec.get("output") or {}
                response = out.get("response", "") or ""
                ctx = out.get("context") or {}

                blob = prompt + "\n" + response + "\n" + " ".join(ctx.keys())
                if not _CVDP_SVA_KEYWORDS.search(blob):
                    continue

                # Prefer an explicit assertion body found in response/context
                sva = _extract_first_sva(response) or _extract_first_sva(
                    "\n".join(v for v in ctx.values() if isinstance(v, str))
                )
                if not sva:
                    continue
                yield normalize_sample(
                    orig_id=rec.get("id", f"{jp.stem}_{hash(line) & 0xFFFF:x}"),
                    source=name,
                    nl=prompt,
                    reference_sva=sva,
                    rtl_context=next(iter(ctx.values()), "") if isinstance(ctx, dict) else "",
                    split=split,
                    clf=clf,
                )


_SVA_BLOCK_RE = re.compile(
    r"assert\s+property\s*\([^;]*?\)\s*;",
    re.IGNORECASE | re.DOTALL,
)


def _extract_first_sva(text: str) -> Optional[str]:
    if not text:
        return None
    m = _SVA_BLOCK_RE.search(text)
    return m.group(0).strip() if m else None


def load_handcrafted(name: str, raw_path: Path, split: str, clf: TCLClassifier):
    """Pull existing hand-crafted pilot data from data/*.json."""
    for fname in ("sample_svas.json", "sva_examples.json",
                  "nl2sva_tasks_expanded.json", "nl2sva_tasks.json",
                  "nl_to_sva.json"):
        path = DATA_DIR / fname
        if not path.exists():
            continue
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception:
            continue
        items = data if isinstance(data, list) else (
            data.get("tasks") or data.get("examples") or data.get("data") or []
        )
        for i, it in enumerate(items):
            if not isinstance(it, dict):
                continue
            sva = (it.get("reference_sva") or it.get("sva")
                   or it.get("assertion") or "")
            if not sva:
                continue
            yield normalize_sample(
                orig_id=f"{fname}_{it.get('id', i)}",
                source=name,
                nl=it.get("nl") or it.get("description", ""),
                reference_sva=sva,
                rtl_context=it.get("rtl", ""),
                split=split,
                clf=clf,
            )


def load_github_scraped(name: str, raw_path: Path, split: str, clf: TCLClassifier):
    """Scraper output: one JSON record per SVA with
       {sva, nl_comment, module, file, line, source_repo, license, body_hash}"""
    if not raw_path.exists():
        return
    with open(raw_path) as f:
        for i, line in enumerate(f):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            sva = rec.get("sva", "").strip()
            if not sva:
                continue
            nl = rec.get("nl_comment", "") or ""
            module = rec.get("module", "") or ""
            rtl = f"// module {module}" if module else ""
            yield normalize_sample(
                orig_id=f"{rec.get('source_repo','?').replace('/','_')}"
                        f"_{rec.get('file','?').replace('/','_')}_L{rec.get('line','?')}",
                source=name,
                nl=nl,
                reference_sva=sva,
                rtl_context=rtl,
                split=split,
                clf=clf,
            )


def load_opentitan_macros(name: str, raw_path: Path, split: str, clf: TCLClassifier):
    """Macro-expanded SVAs: one JSON record with {macro, sva, args, file,
    line, source_repo, body_hash}."""
    if not raw_path.exists():
        return
    with open(raw_path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            sva = rec.get("sva", "").strip()
            if not sva:
                continue
            yield normalize_sample(
                orig_id=f"{rec.get('source_repo','?').replace('/','_')}"
                        f"_{rec.get('file','?').replace('/','_')}"
                        f"_L{rec.get('line','?')}_{rec.get('macro','?')}",
                source=name,
                nl=f"[{rec.get('macro','?')}] " + (rec.get("args", [""])[0] or ""),
                reference_sva=sva,
                rtl_context="",
                split=split,
                clf=clf,
            )


def load_named_properties(name: str, raw_path: Path, split: str, clf: TCLClassifier):
    """Named property blocks scraper output: {name, sva, body, file, line,
    module, nl_comment, source_repo, body_hash}"""
    if not raw_path.exists():
        return
    with open(raw_path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            sva = rec.get("sva", "").strip()
            if not sva:
                continue
            nl = rec.get("nl_comment", "") or ""
            module = rec.get("module", "") or ""
            rtl = f"// module {module}" if module else ""
            yield normalize_sample(
                orig_id=f"{rec.get('source_repo','?').replace('/','_')}"
                        f"_{rec.get('file','?').replace('/','_')}"
                        f"_L{rec.get('line','?')}_{rec.get('name','?')}",
                source=name,
                nl=nl,
                reference_sva=sva,
                rtl_context=rtl,
                split=split,
                clf=clf,
            )


def load_snapshots_scraped(name: str, raw_path: Path, split: str, clf: TCLClassifier):
    """scrape_snapshots.py output: {source_snapshot, strategy, sva,
    nl_comment, module, file, line, body_hash}"""
    if not raw_path.exists():
        return
    with open(raw_path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            sva = rec.get("sva", "").strip()
            if not sva:
                continue
            nl = rec.get("nl_comment", "") or ""
            module = rec.get("module", "") or ""
            rtl = f"// module {module}" if module else ""
            yield normalize_sample(
                orig_id=f"{rec.get('source_snapshot','?')}_"
                        f"{rec.get('strategy','?').replace(':','_')}_"
                        f"{rec.get('file','?').replace('/','_')}_L{rec.get('line','?')}",
                source=name,
                nl=nl,
                reference_sva=sva,
                rtl_context=rtl,
                split=split,
                clf=clf,
            )


LOADERS = {
    "load_nl2sva_csv": load_nl2sva_csv,
    "load_design2sva": load_design2sva,
    "load_assertionbench": load_assertionbench,
    "load_cvdp": load_cvdp,
    "load_handcrafted": load_handcrafted,
    "load_github_scraped": load_github_scraped,
    "load_opentitan_macros": load_opentitan_macros,
    "load_named_properties": load_named_properties,
    "load_snapshots_scraped": load_snapshots_scraped,
}


# -----------------------------------------------------------------------------
# Split I/O + contamination
# -----------------------------------------------------------------------------
def load_test_hashes() -> set:
    if not TEST_HASHES.exists():
        return set()
    with open(TEST_HASHES) as f:
        return set(json.load(f))


def save_test_hashes(hashes):
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    with open(TEST_HASHES, "w") as f:
        json.dump(sorted(set(hashes)), f, indent=2)


def write_split(samples, split, source_name):
    out_dir = TEST_DIR if split == "test" else TRAIN_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{source_name}.jsonl"
    with open(out_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    return out_path


def contamination_check() -> int:
    test_hashes = load_test_hashes()
    if not test_hashes:
        print("[contamination] no test hashes yet — run --build --all first")
        return 0
    bad = 0
    for p in TRAIN_DIR.glob("*.jsonl"):
        with open(p) as f:
            for line in f:
                s = json.loads(line)
                if s["hash"] in test_hashes:
                    print(f"  LEAK: {p.name} :: {s['id']} hash={s['hash']}")
                    bad += 1
    if bad == 0:
        print(f"[contamination] OK — 0 leaks across "
              f"{sum(1 for _ in TRAIN_DIR.glob('*.jsonl'))} train files "
              f"vs {len(test_hashes)} test hashes")
    return bad


# -----------------------------------------------------------------------------
# Build pipeline
# -----------------------------------------------------------------------------
def build(only: Optional[List[str]]) -> dict:
    clf = TCLClassifier()
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    summary = {"test": {}, "train": {}, "rejected_contamination": []}

    def process(name: str):
        cfg = SOURCES[name]
        print(f"\n=== {name}  (split={cfg['split']}) ===")
        print(f"    role: {cfg['role']}")
        raw_path = (RAW_DIR / cfg["raw_path"]) if cfg.get("raw_path") else Path("/dev/null")
        loader = LOADERS[cfg["loader"]]
        samples = list(loader(name, raw_path, cfg["split"], clf))
        print(f"    normalized: {len(samples)} samples")
        return samples

    # test first, to populate hash set before train
    test_hashes = set()
    for name, cfg in SOURCES.items():
        if cfg["split"] != "test":
            continue
        if only and name not in only:
            continue
        samples = process(name)
        if samples:
            write_split(samples, "test", name)
            test_hashes.update(s["hash"] for s in samples)
            summary["test"][name] = len(samples)
    save_test_hashes(test_hashes)

    for name, cfg in SOURCES.items():
        if cfg["split"] != "train":
            continue
        if only and name not in only:
            continue
        samples = process(name)
        kept, rejected = [], []
        for s in samples:
            if s["hash"] in test_hashes:
                rejected.append(s["id"])
            else:
                kept.append(s)
        if rejected:
            print(f"    REJECTED {len(rejected)} as test-set duplicates "
                  f"(first 3: {rejected[:3]})")
            summary["rejected_contamination"].extend(rejected)
        if kept:
            write_split(kept, "train", name)
            summary["train"][name] = len(kept)

    return summary


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true",
                    help="download raw benchmarks into data/raw/")
    ap.add_argument("--build", action="store_true",
                    help="normalize + route to data/train/ and data/test/")
    ap.add_argument("--all", action="store_true", help="all sources")
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--contamination-check", action="store_true")
    args = ap.parse_args()

    if args.contamination_check:
        sys.exit(1 if contamination_check() else 0)

    if args.fetch:
        names = args.only or list(SOURCES)
        seen = set()
        for n in names:
            cfg = SOURCES[n]
            f = cfg.get("fetcher")
            key = (f["kind"], f["url"] if f and "url" in f else
                   f.get("repo_id")) if f else None
            if key in seen:
                continue
            if key:
                seen.add(key)
            fetch_one(n, cfg)

    if args.build or args.all:
        summary = build(only=args.only)
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(json.dumps(summary, indent=2))
        print("\nTest hashes registered:", len(load_test_hashes()))
        if summary["rejected_contamination"]:
            print(f"Rejected {len(summary['rejected_contamination'])} "
                  f"contaminated train samples.")

    if not (args.fetch or args.build or args.all):
        print(__doc__)


if __name__ == "__main__":
    main()
