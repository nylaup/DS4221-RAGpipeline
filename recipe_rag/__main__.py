"""Command line for the recipe RAG pipeline.

    uv run python -m recipe_rag ask "What oven temperature does the corn soufflé use?"
    uv run python -m recipe_rag search "cassava flakes" -k 5        # retrieval only
    uv run python -m recipe_rag find taro                           # labeling helper
    uv run python -m recipe_rag export-partner                      # question swap file

Common options: --config section|whole  --dim 384|64  --index ivf|brute
                --probe N  --no-filter
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
import time

from .db import ROOT, connect


def _retriever(a):
    from .retrieve import Retriever
    return Retriever(a.config, a.dim, a.index, a.probe, use_filters=not a.no_filter)


def cmd_search(a):
    r = _retriever(a)
    res = r.retrieve(a.question, k=a.k)
    if res["filters"]:
        print("filters:", "; ".join(res["filters"]))
    for rank, c in enumerate(res["chunks"], 1):
        print(f"\n#{rank}  {c['score']:.3f}  {c['chunk_id']}")
        print(f"    {c['title']} — {c['section'] or 'whole recipe'}  ({c['url']})")
        print(textwrap.indent(textwrap.shorten(c["text"].replace("\n", " "), 300), "    "))
    print(f"\nembed {res['embed_ms']:.1f} ms, search {res['search_ms']:.1f} ms "
          f"[{a.config}/{a.dim}d/{a.index}{f' probe={a.probe}' if a.index == 'ivf' else ''}]")


def cmd_ask(a):
    from .generate import generate, llm_configured, parse_citations
    t0 = time.perf_counter()
    r = _retriever(a)
    res = r.retrieve(a.question, k=a.k)
    ans = generate(a.question, res["chunks"])
    print(ans)
    cites = parse_citations(ans, res["chunks"]) if llm_configured() else []
    if cites:
        print("\nSources:")
        rows = r.con.execute("""
            SELECT c.chunk_id, d.title, c.section, d.url FROM chunks c JOIN documents d USING (doc_id)
            WHERE c.chunk_id IN (SELECT unnest(?::VARCHAR[]))""", [cites]).fetchall()
        for cid, title, section, url in sorted(rows, key=lambda x: cites.index(x[0])):
            print(f"  [{cid}] {title} — {section or 'whole recipe'}  {url}")
    if res["filters"]:
        print("filters:", "; ".join(res["filters"]))
    print(f"({time.perf_counter() - t0:.1f}s total)", file=sys.stderr)


def cmd_find(a):
    """Keyword search over titles, categories and text — for LABELING
    relevant_doc_ids (find every recipe that should count), not retrieval."""
    con = connect(read_only=True)
    pat = f"%{a.term}%"
    rows = con.execute("""
        SELECT d.doc_id, d.category, d.total_minutes,
               d.title ILIKE ? AS in_title,
               coalesce(array_to_string(d.categories, '|') ILIKE ?, false) AS in_cats,
               count(*) FILTER (WHERE c.text ILIKE ?) AS text_hits
        FROM documents d JOIN chunks c USING (doc_id)
        WHERE c.chunk_config = 'whole'
        GROUP BY ALL
        HAVING in_title OR in_cats OR text_hits > 0
        ORDER BY in_title DESC, in_cats DESC, d.doc_id
        LIMIT ?""", [pat, pat, pat, a.limit]).fetchall()
    for doc_id, cat, mins, t, c, n in rows:
        where = ",".join(w for w, f in (("title", t), ("category", c), ("text", n)) if f)
        print(f"{doc_id:55s} {str(cat or ''):28s} {str(mins or ''):>5s}  [{where}]")
    print(f"{len(rows)} docs")


def cmd_export_partner(a):
    """Write the question-swap package: our questions + doc ids + corpus pointer."""
    con = connect(read_only=True)
    if a.ids:
        rows = con.execute("""SELECT q_id, question, q_type, relevant_doc_ids, author
                              FROM eval_questions WHERE source_team = 'ours' AND q_id IN
                              (SELECT unnest(?::VARCHAR[])) ORDER BY q_id""", [a.ids]).fetchall()
    else:   # default: 10 questions balanced across the four types (3/3/2/2)
        rows = con.execute("""SELECT q_id, question, q_type, relevant_doc_ids, author FROM (
                                SELECT *, row_number() OVER (PARTITION BY q_type ORDER BY q_id) AS rn
                                FROM eval_questions WHERE source_team = 'ours')
                              WHERE rn <= CASE WHEN q_type IN ('lookup', 'procedure') THEN 3 ELSE 2 END
                              ORDER BY q_type, q_id""").fetchall()
    drafts = [r[0] for r in rows if r[4] == "DRAFT"]
    if drafts:
        print(f"WARNING: {len(drafts)} exported questions are still unreviewed DRAFTs: {drafts}")
    out = ROOT / "results" / "partner_swap.json"
    out.parent.mkdir(exist_ok=True)
    payload = {
        "corpus": "Wikibooks Cookbook, Category:Recipes snapshot (CC BY-SA 4.0). "
                  "File: data/raw/cookbook.jsonl.gz in our repo; doc_id = page title "
                  "without 'Cookbook:' and with spaces as underscores; "
                  "URL = https://en.wikibooks.org/wiki/Cookbook:<doc_id>",
        "questions": [{"q_id": q, "question": t, "q_type": ty, "relevant_doc_ids": rel, "author": au}
                      for q, t, ty, rel, au in rows]}
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(rows)} questions -> {out}")


def main():
    ap = argparse.ArgumentParser(prog="recipe_rag", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("ask", cmd_ask), ("search", cmd_search)):
        p = sub.add_parser(name)
        p.add_argument("question", nargs="+")
        p.add_argument("-k", type=int, default=5)
        p.add_argument("--config", default="section", choices=["section", "whole"])
        p.add_argument("--dim", type=int, default=384, choices=[384, 64])
        p.add_argument("--index", default="ivf", choices=["ivf", "brute"])
        p.add_argument("--probe", type=int, default=8)
        p.add_argument("--no-filter", action="store_true")
        p.set_defaults(fn=fn)
    p = sub.add_parser("find"); p.add_argument("term"); p.add_argument("--limit", type=int, default=60)
    p.set_defaults(fn=cmd_find)
    p = sub.add_parser("export-partner"); p.add_argument("ids", nargs="*")
    p.set_defaults(fn=cmd_export_partner)
    a = ap.parse_args()
    if hasattr(a, "question"):
        a.question = " ".join(a.question)
    a.fn(a)


if __name__ == "__main__":
    main()
