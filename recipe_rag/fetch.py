"""Download every page in Wikibooks' Category:Recipes via the MediaWiki API.

Output: data/raw/cookbook.jsonl.gz, one JSON object per page:
    {"pageid", "title", "revid", "timestamp", "wikitext"}

This is the frozen snapshot of the corpus. Everything downstream reads this
file, so re-running the pipeline never depends on the live wiki. The content
is CC BY-SA 4.0 (Wikibooks), so the snapshot can be shared with course staff.

    uv run python -m recipe_rag.fetch            # skip if the file exists
    uv run python -m recipe_rag.fetch --force    # re-download
"""

from __future__ import annotations

import argparse
import gzip
import json
import time
from pathlib import Path

import requests

API = "https://en.wikibooks.org/w/api.php"
HEADERS = {"User-Agent": "DS4221-RecipeRAG/0.1 (university course project; educational use)"}
RAW = Path(__file__).resolve().parent.parent / "data" / "raw" / "cookbook.jsonl.gz"


def _get(params: dict) -> dict:
    params = {**params, "format": "json", "formatversion": 2}
    for attempt in range(8):
        r = requests.get(API, params=params, headers=HEADERS, timeout=60)
        if r.status_code == 200:
            return r.json()
        wait = int(r.headers.get("Retry-After", 0) or 0) or 5 * 2 ** attempt
        print(f"  HTTP {r.status_code}; waiting {wait}s", flush=True)
        time.sleep(wait)
    r.raise_for_status()
    return {}


def list_recipe_pages() -> list[int]:
    """Page ids of every Cookbook-namespace member of Category:Recipes."""
    ids, cont = [], {}
    while True:
        d = _get({"action": "query", "list": "categorymembers",
                  "cmtitle": "Category:Recipes", "cmnamespace": 102,
                  "cmlimit": 500, "cmtype": "page", **cont})
        ids += [m["pageid"] for m in d["query"]["categorymembers"]]
        if "continue" not in d:
            return ids
        cont = d["continue"]


PARTIAL = RAW.with_suffix(".partial.jsonl")


def fetch_pages(pageids: list[int]) -> list[dict]:
    """Fetch in batches of 20, appending to a .partial file so an interrupted
    or rate-limited run resumes where it stopped."""
    out = []
    if PARTIAL.exists():
        out = [json.loads(l) for l in PARTIAL.open(encoding="utf-8")]
    done = {p["pageid"] for p in out}
    todo = [i for i in pageids if i not in done]
    partial = PARTIAL.open("a", encoding="utf-8")
    for i in range(0, len(todo), 20):
        batch = todo[i:i + 20]
        d = _get({"action": "query", "prop": "revisions",
                  "rvprop": "content|ids|timestamp", "rvslots": "main",
                  "pageids": "|".join(map(str, batch))})
        for p in d["query"]["pages"]:
            if "revisions" not in p:
                continue
            rev = p["revisions"][0]
            rec = {"pageid": p["pageid"], "title": p["title"],
                   "revid": rev["revid"], "timestamp": rev["timestamp"],
                   "wikitext": rev["slots"]["main"]["content"]}
            out.append(rec)
            partial.write(json.dumps(rec, ensure_ascii=False) + "\n")
        partial.flush()
        print(f"  fetched {len(done) + min(i + 20, len(todo))}/{len(pageids)}", flush=True)
        time.sleep(1.0)          # be polite to the Wikimedia API
    partial.close()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if RAW.exists() and not args.force:
        print(f"{RAW} exists; use --force to re-download")
        return
    ids = list_recipe_pages()
    print(f"{len(ids)} recipe pages in Category:Recipes")
    pages = sorted(fetch_pages(ids), key=lambda p: p["title"])
    RAW.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(RAW, "wt", encoding="utf-8") as f:
        for p in pages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    PARTIAL.unlink(missing_ok=True)
    print(f"wrote {len(pages)} pages -> {RAW}")


if __name__ == "__main__":
    main()
