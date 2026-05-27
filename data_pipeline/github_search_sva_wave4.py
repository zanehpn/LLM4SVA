#!/usr/bin/env python3
"""
github_search_sva_wave4.py — Wave 4: more aggressive recall (star >= 10)
+ untried query angles (verification frameworks, signal patterns, file
naming conventions).

Wave 1+2+3 covered the high-star and direct-keyword search space. Wave 4
goes lower (stars >= 10) and uses query angles biased toward smaller
verification IP repos and file/dir naming conventions that prior waves
missed.
"""
import json, os, sys, time, urllib.request, urllib.parse
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = EXPERIMENTS_DIR / "data" / "raw" / "github_search"
OUT_JSON = OUT_DIR / "candidates_v4.json"

MIN_STARS = 10           # was 30 in Wave 3, 100 in Wave 1+2
MAX_HITS_PER_QUERY = 1000
PER_PAGE = 100


# Wave-4 queries — different angles than wave 1/2/3
QUERIES = [
    # Verification framework keywords paired with assertions
    "\"`uvm_info\" \"assert property\" language:SystemVerilog",
    "\"`SVTEST\" language:SystemVerilog",
    "\"verify_property\" language:SystemVerilog",
    "\"checker\" \"assert property\" language:SystemVerilog",

    # Bind-style verification IP
    "\"bind \" \"assert property\" language:SystemVerilog",

    # Less common assertion keywords
    "\"assume_property\" language:SystemVerilog",
    "\"`ASSERT_NEVER_AT\" language:SystemVerilog",
    "\"`ASSERT_AT_RESET\" language:SystemVerilog",
    "\"`ASSUME_FPV\" language:SystemVerilog",

    # Specific signal naming conventions in industrial SVAs
    "\"valid_o\" \"assert property\" language:SystemVerilog",
    "\"ready_i\" \"assert property\" language:SystemVerilog",
    "\"req_i\" \"assert property\" language:SystemVerilog",
    "\"gnt_o\" \"assert property\" language:SystemVerilog",

    # Power / DFT / clock domain verification
    "\"clock domain\" \"assert property\" language:SystemVerilog",
    "\"isolation\" \"assert property\" language:SystemVerilog",

    # Common verification block types
    "\"`MONITOR\" \"assert property\" language:SystemVerilog",
    "\"`SCOREBOARD\" \"assert property\" language:SystemVerilog",

    # Specific protocols (deeper than wave 3)
    "\"OCP\" \"assert property\" language:SystemVerilog",
    "\"NOC\" \"assert property\" language:SystemVerilog",
    "\"AHB-Lite\" \"assert property\" language:SystemVerilog",
    "\"AXI4-Stream\" \"assert property\" language:SystemVerilog",

    # Property labels / file-name hints
    "\"asrt_\" language:SystemVerilog",
    "\"property prop_\" language:SystemVerilog",
    "filename:assertions.sv",
    "filename:checkers.sv",
    "filename:asserts.sv",
    "filename:fv_constraints.sv",
    "filename:formal_properties.sv",

    # Academic verification corpora
    "\"course\" \"assert property\" language:SystemVerilog",
    "\"homework\" \"assert property\" language:SystemVerilog",
    "\"lab\" \"assert property\" language:SystemVerilog",
]


def _get_token():
    t = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not t:
        print("ERROR: set GITHUB_TOKEN or GH_TOKEN env var", file=sys.stderr)
        sys.exit(2)
    return t


def _api(url, token):
    req = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "SVA4DAC-seed-finder",
        "Authorization": f"Bearer {token}",
    })
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read()), dict(r.headers.items())


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
    info = {"full_name": data.get("full_name"),
            "stars": data.get("stargazers_count", 0),
            "license": (data.get("license") or {}).get("spdx_id", "NOASSERTION"),
            "description": (data.get("description") or "")[:120]}
    cache[full_name] = info
    time.sleep(1)
    return info


def main():
    token = _get_token()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    existing = set()
    for f in [OUT_DIR / "candidates.json", OUT_DIR / "candidates_v2.json",
              OUT_DIR / "candidates_v3.json"]:
        if f.exists():
            for c in json.load(open(f)):
                existing.add(c["repo"])
    passers = EXPERIMENTS_DIR / "data" / "raw" / "github_scraped" / "_passers.json"
    if passers.exists():
        for p in json.load(open(passers)):
            existing.add(p["repo"])
    scraped = EXPERIMENTS_DIR / "data" / "raw" / "github_search_scraped" / "scraped.jsonl"
    if scraped.exists():
        for line in open(scraped):
            try: existing.add(json.loads(line)["source_repo"])
            except: pass
    print(f"[dedup] {len(existing)} repos already processed — will skip")

    per_repo_hits = {}
    for q in QUERIES:
        print(f"\n=== search: {q} ===")
        for hit in search_code(q, token):
            full = hit.get("repository", {}).get("full_name")
            if not full: continue
            per_repo_hits[full] = per_repo_hits.get(full, 0) + 1
        print(f"  total repos so far: {len(per_repo_hits)}")

    print(f"\nTotal unique repos seen: {len(per_repo_hits)}")
    repo_cache = {}
    candidates = []
    to_check = [r for r in per_repo_hits if r not in existing]
    print(f"New repos to check stars: {len(to_check)}")
    for repo in sorted(to_check, key=lambda r: -per_repo_hits[r]):
        info = get_repo(repo, token, repo_cache)
        if info is None or info["stars"] < MIN_STARS:
            continue
        candidates.append({"repo": repo, "stars": info["stars"],
                           "license": info["license"],
                           "description": info["description"],
                           "hits": per_repo_hits[repo]})
        print(f"  [OK] {repo:<48} {info['stars']:>5}★ "
              f"{info['license']:<18} hits={per_repo_hits[repo]}")
        if len(candidates) >= 400:
            break

    candidates.sort(key=lambda c: -c["stars"])
    with open(OUT_JSON, "w") as f:
        json.dump(candidates, f, indent=2)
    print(f"\n[ok] wrote {len(candidates)} new candidates to {OUT_JSON}")


if __name__ == "__main__":
    main()
