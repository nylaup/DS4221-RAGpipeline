"""Evaluation: retrieval quality, ANN tradeoff, answer quality.

    uv run python -m recipe_rag.evaluate retrieval   # recall@k / MRR per configuration
    uv run python -m recipe_rag.evaluate ann         # probe sweep: recall vs comparisons vs latency
    uv run python -m recipe_rag.evaluate answers     # generate + LLM-judge (needs LLM_*)

Outputs go to results/ (CSV + a markdown summary per command).

Retrieval metrics are DOC-level: a question's relevant_doc_ids are recipes,
and the ranking is the retriever's doc_ranking (docs ordered by best chunk).
    recall@k = |relevant ∩ top-k docs| / |relevant|
    MRR@k    = 1 / rank of the first relevant doc within the top k (else 0)

The ANN sweep measures the index alone (chunk level, like Lab 9):
recall@10 of IVF against brute force for the same query vectors, using 300
synthetic queries ("How do I make <title>?") so the speed numbers rest on
more than our ~40 hand-written questions.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import time
from collections import defaultdict

import numpy as np

from .build import RESULTS
from .db import connect
from .embed import embed
from .index import recall_at_k
from .retrieve import Retriever

KS = (1, 5, 10)


def questions(con, team: str | None = None) -> list[dict]:
    sql = "SELECT q_id, question, q_type, relevant_doc_ids, author, source_team FROM eval_questions"
    rows = con.execute(sql + (" WHERE source_team = ?" if team else "") + " ORDER BY q_id",
                       [team] if team else []).fetchall()
    cols = ["q_id", "question", "q_type", "relevant", "author", "team"]
    return [dict(zip(cols, r)) for r in rows]


def metrics(ranking: list[str], relevant: list[str]) -> dict:
    rel = set(relevant)
    out = {}
    for k in KS:
        top = ranking[:k]
        out[f"recall@{k}"] = len(rel & set(top)) / len(rel)
        rr = next((1 / (i + 1) for i, d in enumerate(top) if d in rel), 0.0)
        out[f"mrr@{k}"] = rr
        out[f"precision@{k}"] = len(rel & set(top)) / k
    return out


def _mean(rows, key):
    return statistics.fmean(r[key] for r in rows) if rows else float("nan")


# ------------------------------------------------------------ retrieval

CONFIG_GRID = [
    # (chunk config, dim, index, probe, filters)
    ("whole", 384, "brute", None, True),
    ("section", 384, "brute", None, True),
    ("section", 64, "brute", None, True),
    ("whole", 64, "brute", None, True),
    ("section", 384, "ivf", 8, True),
    ("section", 64, "ivf", 8, True),
    ("section", 384, "brute", None, False),
]


def run_retrieval(con, grid=CONFIG_GRID) -> list[dict]:
    qs = questions(con)
    if not qs:
        raise SystemExit("no eval questions loaded: fill data/eval_questions.jsonl and run "
                         "`python -m recipe_rag.build --questions-only`")
    qtext = [q["question"] for q in qs]
    q384 = embed(qtext)                          # embed once, reuse across configs
    summary, per_q = [], []
    for config, dim, kind, probe, filt in grid:
        r = Retriever(config, dim, kind, probe or 1, use_filters=filt, con=con)
        name = f"{config}/{dim}d/{kind}" + (f"(p={probe})" if probe else "") + ("" if filt else "/nofilter")
        rows = []
        for q, v in zip(qs, q384):
            qv = r.pca.transform(v) if r.pca is not None else v
            res = r.retrieve(q["question"], k=10, qv=qv)
            m = metrics(res["doc_ranking"], q["relevant"])
            rows.append({"config": name, **q, "relevant": "|".join(q["relevant"]),
                         "top5": "|".join(res["doc_ranking"][:5]),
                         "filters": "; ".join(res["filters"]),
                         "search_ms": res["search_ms"], **m})
        per_q += rows
        groups = [("all", rows)]
        groups += [(f"type:{t}", [x for x in rows if x["q_type"] == t])
                   for t in sorted({x["q_type"] for x in rows})]
        groups += [(f"team:{t}", [x for x in rows if x["team"] == t])
                   for t in sorted({x["team"] for x in rows})]
        for group, sub in groups:
            summary.append({"config": name, "group": group, "n": len(sub),
                            **{f"{m}@{k}": round(_mean(sub, f"{m}@{k}"), 3)
                               for m in ("recall", "precision", "mrr") for k in KS},
                            "search_ms": round(_mean(sub, "search_ms"), 2)})
        s = summary[-len(groups)]
        print(f"{name:34s} recall@5={s['recall@5']:.3f} recall@10={s['recall@10']:.3f} "
              f"MRR@10={s['mrr@10']:.3f}")
    _write_csv(RESULTS / "retrieval_summary.csv", summary)
    _write_csv(RESULTS / "retrieval_per_question.csv", per_q)
    _write_md(RESULTS / "retrieval_summary.md", "Retrieval quality (doc level)", summary)
    return summary


# ------------------------------------------------------------ ANN sweep

def run_ann(con, n_queries: int = 300, configs=(("section", 384), ("section", 64))) -> list[dict]:
    titles = [t for (t,) in con.execute("SELECT title FROM documents ORDER BY doc_id").fetchall()]
    rng = random.Random(4221)
    qtext = [f"How do I make {t}?" for t in rng.sample(titles, min(n_queries, len(titles)))]
    qtext += [q["question"] for q in questions(con)]
    q384 = embed(qtext)
    build = json.loads((RESULTS / "build_times.json").read_text()) if (RESULTS / "build_times.json").exists() else {}
    rows = []
    for config, dim in configs:
        brute = Retriever(config, dim, "brute", con=con)
        ivf = Retriever(config, dim, "ivf", con=con)
        qvs = [brute.pca.transform(v) if brute.pca is not None else v for v in q384]
        t = time.perf_counter()
        exact = [brute.index.search(qv, 10) for qv in qvs]
        brute_s = time.perf_counter() - t
        n = len(brute.vectors)
        rows.append({"config": f"{config}/{dim}d", "index": "brute", "probe": "-",
                     "n_lists": "-", "recall@10": 1.0, "comparisons/q": n,
                     "ms/q": round(brute_s / len(qvs) * 1e3, 3), "qps": round(len(qvs) / brute_s, 1),
                     "build_s": "-"})
        n_lists = ivf.index.n_lists
        for probe in [p for p in (1, 2, 4, 8, 16, 32, 64) if p <= n_lists] + [n_lists]:
            ivf.index.comparisons = 0
            t = time.perf_counter()
            approx = [ivf.index.search(qv, 10, probe=probe) for qv in qvs]
            s = time.perf_counter() - t
            rec = statistics.fmean(recall_at_k(a, e, 10) for a, e in zip(approx, exact))
            rows.append({"config": f"{config}/{dim}d", "index": "ivf", "probe": probe,
                         "n_lists": n_lists, "recall@10": round(rec, 3),
                         "comparisons/q": round(ivf.index.comparisons / len(qvs)),
                         "ms/q": round(s / len(qvs) * 1e3, 3), "qps": round(len(qvs) / s, 1),
                         "build_s": round(build.get(f"{config}.{dim}.ivf_build_s", float("nan")), 2)})
            print(rows[-1])
    _write_csv(RESULTS / "ann_sweep.csv", rows)
    _write_md(RESULTS / "ann_sweep.md", f"IVF vs brute force ({len(qtext)} queries, chunk-level recall@10)", rows)
    return rows


# ------------------------------------------------------------ answers

JUDGE = """You are grading a cooking assistant's answer. Score each criterion 0 or 1.
correct: the answer actually answers the question and agrees with the reference recipe(s).
faithful: every claim in the answer is supported by the excerpts (no invented steps or numbers).
cited: the answer cites excerpt ids, and the cited excerpts support the claims next to them.
Reply with JSON only: {"correct": 0|1, "faithful": 0|1, "cited": 0|1, "reason": "<one sentence>"}"""


def run_answers(con, config="section", dim=384, kind="ivf", probe=8, k=5, spot=10) -> list[dict]:
    from .generate import build_prompt, chat, generate, llm_configured, parse_citations
    if not llm_configured():
        raise SystemExit("set LLM_BASE_URL / LLM_API_KEY / LLM_MODEL first")
    r = Retriever(config, dim, kind, probe, con=con)
    out = []
    for q in questions(con):
        res = r.retrieve(q["question"], k=k)
        ans = generate(q["question"], res["chunks"])
        cites = parse_citations(ans, res["chunks"])
        judge_in = (f"Question: {q['question']}\nReference recipe ids: {q['relevant']}\n\n"
                    f"{build_prompt(q['question'], res['chunks'])}\n\nAssistant answer:\n{ans}")
        raw = chat([{"role": "system", "content": JUDGE}, {"role": "user", "content": judge_in}])
        try:
            g = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])
        except ValueError:
            g = {"correct": None, "faithful": None, "cited": None, "reason": "unparseable: " + raw[:200]}
        cited_docs = sorted({c.split(":")[0] for c in cites})
        out.append({"q_id": q["q_id"], "q_type": q["q_type"], "team": q["team"],
                    "question": q["question"], "answer": ans, "cited_chunks": cites,
                    "cited_docs": cited_docs,
                    "cites_relevant": bool(set(cited_docs) & set(q["relevant"])), **g})
        print(q["q_id"], {k2: g.get(k2) for k2 in ("correct", "faithful", "cited")})
    RESULTS.mkdir(exist_ok=True)
    with (RESULTS / "answers.jsonl").open("w", encoding="utf-8") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    scored = [o for o in out if o["correct"] is not None]
    summ = defaultdict(list)
    for o in scored:
        for grp in ("all", f"type:{o['q_type']}", f"team:{o['team']}"):
            summ[grp].append(o)
    rows = [{"group": g, "n": len(v),
             **{c: round(statistics.fmean(x[c] for x in v), 3) for c in ("correct", "faithful", "cited")},
             "cites_relevant": round(statistics.fmean(x["cites_relevant"] for x in v), 3)}
            for g, v in summ.items()]
    _write_md(RESULTS / "answers_summary.md", f"Answer quality ({config}/{dim}d/{kind}, LLM judge)", rows)
    rng = random.Random(7)
    sample = rng.sample(out, min(spot, len(out)))
    _write_csv(RESULTS / "spot_check.csv",
               [{"q_id": s["q_id"], "question": s["question"], "answer": s["answer"],
                 "judge_correct": s["correct"], "judge_faithful": s["faithful"], "judge_cited": s["cited"],
                 "human_correct": "", "human_faithful": "", "human_cited": "", "notes": ""} for s in sample])
    print(f"wrote results/answers.jsonl, answers_summary.md, spot_check.csv ({len(sample)} to grade by hand)")
    return out


# ------------------------------------------------------------ io

def _write_csv(path, rows):
    path.parent.mkdir(exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _write_md(path, title, rows):
    if not rows:
        return
    cols = list(rows[0].keys())
    lines = [f"# {title}", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["retrieval", "ann", "answers", "all"])
    args = ap.parse_args()
    con = connect(read_only=True)
    if args.what in ("retrieval", "all"):
        run_retrieval(con)
    if args.what in ("ann", "all"):
        run_ann(con)
    if args.what in ("answers", "all"):
        run_answers(con)


if __name__ == "__main__":
    main()
