#!/usr/bin/env python3
"""
github_search_sva_wave3.py — Wave 3: lower star threshold + new query angles.

Targets that Wave 1/2 didn't hit:
  - Verification IP (PCIe/USB/AHB/AXI/Ethernet/DDR controllers)
  - VUnit / OSVVM / UVM-with-SVA testbench frameworks
  - Crypto / accelerator IP with formal properties
  - SiFive / WD / academic RISC-V variants
  - Older verification testbenches (lower stars but verified)

Star floor lowered: 30 (Wave 1+2 used 100). This catches small but real
verification IP repos.

Token: export GH_TOKEN before running.
Usage:
  export GH_TOKEN=...
  PYTHONPATH=. python3 scripts/github_search_sva_wave3.py
"""
import json
import os
import sys
import time
import urllib.request
import urllib.parse
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = EXPERIMENTS_DIR / "data" / "raw" / "github_search"
OUT_JSON = OUT_DIR / "candidates_v3.json"

MIN_STARS = 30          # was 100 in Wave 1+2
MAX_HITS_PER_QUERY = 1000
PER_PAGE = 100


# Wave-3 queries — different angles than wave 2
QUERIES = [
    # Verification framework identifiers
    "\"vunit\" \"assert property\" language:SystemVerilog",
    "\"osvvm\" \"assert property\" language:SystemVerilog",
    "\"uvm_report_info\" \"assert property\" language:SystemVerilog",

    # Interface assertions (often where VIP lives)
    "\"interface property\" language:SystemVerilog",
    "\"clocking\" \"assert property\" language:SystemVerilog",

    # Bind-time verification (formal IP injected via bind)
    "\"bind\" \"assert property\" language:SystemVerilog",

    # Specific protocol IP (likely to be verification IP)
    "\"axi\" \"assert property\" language:SystemVerilog",
    "\"ahb\" \"assert property\" language:SystemVerilog",
    "\"apb\" \"assert property\" language:SystemVerilog",
    "\"pcie\" \"assert property\" language:SystemVerilog",
    "\"ddr\" \"assert property\" language:SystemVerilog",
    "\"ethernet\" \"assert property\" language:SystemVerilog",
    "\"usb\" \"assert property\" language:SystemVerilog",
    "\"i2c\" \"assert property\" language:SystemVerilog",
    "\"spi\" \"assert property\" language:SystemVerilog",

    # Crypto / security accelerator (often heavily verified)
    "\"aes\" \"assert property\" language:SystemVerilog",
    "\"sha\" \"assert property\" language:SystemVerilog",

    # Cache / coherence (lots of complex temporal properties)
    "\"cache\" \"assert property\" language:SystemVerilog",
    "\"coherence\" \"assert property\" language:SystemVerilog",

    # Pipeline / scheduling (more L4 patterns)
    "\"pipeline\" \"assert property\" language:SystemVerilog",
    "\"hazard\" \"assert property\" language:SystemVerilog",

    # Pattern conventions
    "\"asrt:\" language:SystemVerilog",                # FVEval style label
    "\"a_\" \"assert property\" language:SystemVerilog", # 'a_xxx :' label convention
    "\"chk_\" \"assert property\" language:SystemVerilog",

    # Less common temporal operators
    "\"until_with\" language:SystemVerilog",
    "\"weak(\" language:SystemVerilog",
]


def _get_token():
    t = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not t:
        print("ERROR: set GITHUB_TOKEN or GH_TOKEN env var before running.",
              file=sys.stderr)
        sys.exit(2)
    return t


def _api(url, token):
    req = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "SVA4DAC-seed-finder",
        "Authorization": f"Bearer {token}",
    })
    with urllib.request.urlopen(req, timeout=15) as r:
        headers = dict(r.headers.items())
        return json.loads(r.read()), headers


def search_code(query, token, max_hits=MAX_HITS_PER_QUERY):
    got = 0; page = 1
    while got < max_hits:
        q = urllib.parse.quote(query)
        url = (f"https://api.github.com/search/code?"
               f"q={q}&per_page={PER_PAGE}&page={page}")
        try:
            data, hdrs = _api(url, token)
        except Exception as e:
            print(f"  [search] page {page}: {e}")
            break
        items = data.get("items", [])
        if not items:
            break
        for it in items:
            yield it
        got += len(items)
        rem = int(hdrs.get("X-RateLimit-Remaining", 9))
        if rem < 2:
            sleep_for = int(hdrs.get("X-RateLimit-Reset", time.time() + 60)) - int(time.time()) + 2
            print(f"  [search] rate-limit {rem}; sleeping {sleep_for}s")
            time.sleep(max(1, sleep_for))
        else:
            time.sleep(6)
        if len(items) < PER_PAGE:
            break
        page += 1
        if page > 10:
            break


def get_repo(full_name, token, cache):
    if full_name in cache:
        return cache[full_name]
    url = f"https://api.github.com/repos/{urllib.parse.quote(full_name)}"
    try:
        data, _ = _api(url, token)
    except Exception:
        cache[full_name] = None
        return None
    info = {
        "full_name": data.get("full_name"),
        "stars": data.get("stargazers_count", 0),
        "license": (data.get("license") or {}).get("spdx_id", "NOASSERTION"),
        "description": (data.get("description") or "")[:120],
    }
    cache[full_name] = info
    time.sleep(1)
    return info


def main():
    token = _get_token()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Dedup against Wave 1+2 already-processed repos
    existing = set()
    for f in [OUT_DIR / "candidates.json", OUT_DIR / "candidates_v2.json"]:
        if f.exists():
            for c in json.load(open(f)):
                existing.add(c["repo"])
    passers_path = EXPERIMENTS_DIR / "data" / "raw" / "github_scraped" / "_passers.json"
    if passers_path.exists():
        for p in json.load(open(passers_path)):
            existing.add(p["repo"])
    scraped_jsonl = EXPERIMENTS_DIR / "data" / "raw" / "github_search_scraped" / "scraped.jsonl"
    if scraped_jsonl.exists():
        for line in open(scraped_jsonl):
            try:
                existing.add(json.loads(line)["source_repo"])
            except Exception:
                pass
    print(f"[dedup] {len(existing)} repos already processed — will skip")

    per_repo_hits = {}
    for q in QUERIES:
        print(f"\n=== search: {q} ===")
        for hit in search_code(q, token):
            full = hit.get("repository", {}).get("full_name")
            if not full:
                continue
            per_repo_hits[full] = per_repo_hits.get(full, 0) + 1
        print(f"  total repos so far: {len(per_repo_hits)}")

    print(f"\nTotal unique repos seen: {len(per_repo_hits)}")
    repo_cache = {}
    candidates = []
    to_check = [r for r in per_repo_hits if r not in existing]
    print(f"New repos to check stars: {len(to_check)}")
    for repo in sorted(to_check, key=lambda r: -per_repo_hits[r]):
        info = get_repo(repo, token, repo_cache)
        if info is None:
            continue
        if info["stars"] < MIN_STARS:
            continue
        candidates.append({
            "repo": repo, "stars": info["stars"],
            "license": info["license"],
            "description": info["description"],
            "hits": per_repo_hits[repo],
        })
        print(f"  [OK] {repo:<45} {info['stars']:>6}★ "
              f"{info['license']:<18} hits={per_repo_hits[repo]}")
        if len(candidates) >= 300:
            break

    candidates.sort(key=lambda c: -c["stars"])
    with open(OUT_JSON, "w") as f:
        json.dump(candidates, f, indent=2)
    print(f"\n[ok] wrote {len(candidates)} new candidates to {OUT_JSON}")


if __name__ == "__main__":
    main()
