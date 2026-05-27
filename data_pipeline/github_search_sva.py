#!/usr/bin/env python3
"""
github_search_sva.py — use the GitHub code-search API to discover
repositories containing SVA (beyond our curated seed list), aggregate hits
per repo, filter by stars > MIN_STARS, deduplicate against scraped.SEEDS,
and write a candidate repo list for scripts/scrape_github_sva.py.

Token handling:
  - The PAT MUST be provided via env var GITHUB_TOKEN (or GH_TOKEN).
  - This script does NOT read or write any token to disk.

Output:
  data/raw/github_search/candidates.json
    [{ "repo": "owner/name", "stars": 1234, "hits": 42, "license": "Apache-2.0" }, ...]
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
OUT_JSON = OUT_DIR / "candidates_v2.json"  # wave-2; wave-1 is candidates.json

MIN_STARS = 100
MAX_HITS_PER_QUERY = 1000  # GitHub ceiling
PER_PAGE = 100             # GitHub max

# Wave 2: targeted queries biased toward L3 (ranged delays), L4 (sequence
# ops we haven't hit), and L5 (liveness). Skip generic queries we already
# exhausted in wave 1 (assert property / endproperty / ASSERT_NEVER /
# ASSERT_KNOWN / s_eventually).
QUERIES = [
    # L5 liveness / strong temporal operators
    "\"s_until\" language:SystemVerilog",
    "\"s_always\" language:SystemVerilog",
    "\"s_until_with\" language:SystemVerilog",
    "\"nexttime\" language:SystemVerilog",
    "\"strong(\" language:SystemVerilog",
    # L4 sequence operators we haven't explicitly hit
    "\"throughout\" language:SystemVerilog",
    "\"first_match\" language:SystemVerilog",
    "\"intersect\" language:SystemVerilog",
    # L3-signal bracketed ranged delays — GitHub indexes literal [ and $ poorly,
    # so use common pattern substrings that accompany them.
    "\"##[1:$]\" language:SystemVerilog",
    "\"[*1:$]\" language:SystemVerilog",
    # Other assertion verbs we haven't searched
    "\"assume property\" language:SystemVerilog",
    "\"cover property\" language:SystemVerilog",
    # UVM-style named properties (lots of testbenches use p_* / prop_* prefix)
    "\"property p_\" language:SystemVerilog",
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
        # Also return rate-limit headers for polite pacing
        headers = dict(r.headers.items())
        return json.loads(r.read()), headers


def search_code(query, token, max_hits=MAX_HITS_PER_QUERY):
    """Yield hit dicts for a query, paginated up to max_hits."""
    got = 0
    page = 1
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
        # Pacing — code search is rate-limited to 10 req/min
        rem = int(hdrs.get("X-RateLimit-Remaining", 9))
        if rem < 2:
            sleep_for = int(hdrs.get("X-RateLimit-Reset", time.time() + 60)) - int(time.time()) + 2
            print(f"  [search] rate-limit {rem} remain; sleeping {sleep_for}s")
            time.sleep(max(1, sleep_for))
        else:
            time.sleep(6)  # ~10 req/min ceiling
        if len(items) < PER_PAGE:
            break
        page += 1
        if page > 10:   # max 1000/query
            break


def get_repo(full_name, token, cache):
    if full_name in cache:
        return cache[full_name]
    url = f"https://api.github.com/repos/{urllib.parse.quote(full_name)}"
    try:
        data, _ = _api(url, token)
    except Exception as e:
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

    # Existing seeds to dedupe against:
    #   - wave-1 curated SEEDS (already cloned into data/raw/github_repos/)
    #   - wave-1 GH code-search candidates (already clone-scan-deleted)
    #   - repos seen in the github_search_scraped.jsonl output
    existing = set()
    passers_path = EXPERIMENTS_DIR / "data" / "raw" / "github_scraped" / "_passers.json"
    if passers_path.exists():
        existing = {p["repo"] for p in json.load(open(passers_path))}
    prev_cand = EXPERIMENTS_DIR / "data" / "raw" / "github_search" / "candidates.json"
    if prev_cand.exists():
        for c in json.load(open(prev_cand)):
            existing.add(c["repo"])
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

    # Star lookup (only for repos not in existing seeds)
    repo_cache = {}
    candidates = []
    to_check = [r for r in per_repo_hits if r not in existing]
    print(f"New repos to check stars: {len(to_check)}")
    for i, repo in enumerate(sorted(to_check, key=lambda r: -per_repo_hits[r])):
        info = get_repo(repo, token, repo_cache)
        if info is None:
            continue
        if info["stars"] < MIN_STARS:
            continue
        candidates.append({
            "repo": repo,
            "stars": info["stars"],
            "license": info["license"],
            "description": info["description"],
            "hits": per_repo_hits[repo],
        })
        print(f"  [OK] {repo:<45} {info['stars']:>6}★ "
              f"{info['license']:<18} hits={per_repo_hits[repo]}")
        if len(candidates) >= 200:
            break

    candidates.sort(key=lambda c: -c["stars"])
    with open(OUT_JSON, "w") as f:
        json.dump(candidates, f, indent=2)
    print(f"\n[ok] wrote {len(candidates)} new candidates to {OUT_JSON}")


if __name__ == "__main__":
    main()
