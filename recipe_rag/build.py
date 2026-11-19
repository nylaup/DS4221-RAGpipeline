"""Build everything from the raw snapshot: tables, chunks, vectors, indexes.

    uv run python -m recipe_rag.build            # all configs
    uv run python -m recipe_rag.build --fresh    # delete the database first

Steps (each timed; timings go to results/build_times.json):
  1. parse every page -> documents         (insert once; never rewritten)
  2. chunk with each config -> chunks
  3. embed chunks (384-d) and fit PCA (64-d) -> data/vectors/*.npy
  4. build one IVF index per (config, dim) -> data/index/*.npz
     and record each chunk's IVF list in embeddings.centroid_id
  5. load data/eval_questions.jsonl (+ partner_questions.jsonl) -> eval_questions
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import time
from pathlib import Path
from urllib.parse import quote

import numpy as np

from . import chunk as chunker
from .db import DB_PATH, ROOT, bulk_insert, connect
from .embed import MODEL_TAG, count_tokens, embed
from .fetch import RAW
from .index import PCA, IVFIndex
from .wikitext import parse_recipe

VEC_DIR = ROOT / "data" / "vectors"
IDX_DIR = ROOT / "data" / "index"
RESULTS = ROOT / "results"
DIMS = (384, 64)


def model_tag(dim: int) -> str:
    return MODEL_TAG if dim == 384 else f"{MODEL_TAG}+pca{dim}"


def vec_path(config: str, dim: int) -> Path:
    return VEC_DIR / f"{config}_{dim}.npy"


def ivf_path(config: str, dim: int) -> Path:
    return IDX_DIR / f"ivf_{config}_{dim}.npz"


def pca_path(config: str) -> Path:
    return VEC_DIR / f"pca_{config}.npz"


def default_n_lists(n: int) -> int:
    return max(8, round(math.sqrt(n)))


def slug(title: str) -> str:
    return title.removeprefix("Cookbook:").replace(" ", "_")


# ------------------------------------------------------------ steps

def load_documents(con) -> list[dict]:
    pages = [json.loads(l) for l in gzip.open(RAW, "rt", encoding="utf-8")]
    docs = []
    for p in pages:
        r = parse_recipe(p["title"], p["wikitext"])
        r.update(doc_id=slug(p["title"]), revid=p["revid"],
                 url="https://en.wikibooks.org/wiki/" + quote(p["title"].replace(" ", "_"), safe=":/(),'"))
        docs.append(r)
    bulk_insert(con, "documents",
                ["doc_id", "source", "source_version", "title", "url", "category", "difficulty",
                 "servings", "total_time", "total_minutes", "categories"],
                [(d["doc_id"], "wikibooks-cookbook", d["revid"], d["title"], d["url"], d["category"],
                  d["difficulty"], d["servings"], d["time_text"], d["total_minutes"], d["categories"])
                 for d in docs])
    return docs


def write_chunks(con, docs: list[dict], config: str) -> list[tuple[str, str]]:
    rows = []
    for d in docs:
        for i, c in enumerate(chunker.chunk(d, config)):
            cid = f"{d['doc_id']}:{d['revid']}:{config}:{i}"
            rows.append((cid, d["doc_id"], i, c["section"], config, c["text"]))
    ntok = count_tokens([r[5] for r in rows])
    con.execute("DELETE FROM embeddings WHERE chunk_id IN (SELECT chunk_id FROM chunks WHERE chunk_config = ?)", [config])
    con.execute("DELETE FROM chunks WHERE chunk_config = ?", [config])
    bulk_insert(con, "chunks",
                ["chunk_id", "doc_id", "ordinal", "section", "chunk_config", "text", "n_tokens"],
                [r + (n,) for r, n in zip(rows, ntok)])
    return [(r[0], r[5]) for r in rows]


def load_questions(con) -> int:
    con.execute("DELETE FROM eval_questions")
    n = 0
    for fname, team in (("eval_questions.jsonl", "ours"), ("partner_questions.jsonl", "partner")):
        path = ROOT / "data" / fname
        if not path.exists():
            continue
        for line in path.open(encoding="utf-8"):
            if not line.strip() or line.lstrip().startswith("//"):
                continue
            q = json.loads(line)
            con.execute("INSERT INTO eval_questions VALUES (?, ?, ?, ?, ?, ?)",
                        [q["q_id"], q["question"], q["q_type"], q["relevant_doc_ids"],
                         q["author"], q.get("source_team", team)])
            n += 1
    bad = con.execute("""
        SELECT q_id, d FROM (SELECT q_id, unnest(relevant_doc_ids) AS d FROM eval_questions)
        WHERE d NOT IN (SELECT doc_id FROM documents)""").fetchall()
    if bad:
        raise SystemExit(f"eval_questions reference unknown doc_ids: {bad}")
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true", help="delete the database first")
    ap.add_argument("--configs", nargs="+", default=list(chunker.CONFIGS))
    ap.add_argument("--questions-only", action="store_true",
                    help="only reload the eval question files")
    args = ap.parse_args()

    if args.fresh and DB_PATH.exists():
        DB_PATH.unlink()
    con = connect()
    if args.questions_only:
        print(f"{load_questions(con)} eval questions loaded")
        return

    times: dict = {}
    t = time.perf_counter()
    docs = load_documents(con)
    times["parse_documents_s"] = time.perf_counter() - t
    print(f"{len(docs)} documents ({times['parse_documents_s']:.1f}s)")

    VEC_DIR.mkdir(parents=True, exist_ok=True)
    for config in args.configs:
        t = time.perf_counter()
        rows = write_chunks(con, docs, config)
        times[f"{config}.chunk_s"] = time.perf_counter() - t
        print(f"[{config}] {len(rows)} chunks")

        t = time.perf_counter()
        v384 = embed([text for _, text in rows], progress=True)
        times[f"{config}.embed_s"] = time.perf_counter() - t
        np.save(vec_path(config, 384), v384)

        t = time.perf_counter()
        pca = PCA(64).fit(v384)
        v64 = pca.transform(v384)
        times[f"{config}.pca_s"] = time.perf_counter() - t
        pca.save(pca_path(config))
        np.save(vec_path(config, 64), v64)

        for dim, vecs in ((384, v384), (64, v64)):
            n_lists = default_n_lists(len(vecs))
            t = time.perf_counter()
            ivf = IVFIndex(vecs, n_lists=n_lists, iters=10)
            times[f"{config}.{dim}.ivf_build_s"] = time.perf_counter() - t
            times[f"{config}.{dim}.n_lists"] = n_lists
            ivf.save(ivf_path(config, dim))
            con.execute("DELETE FROM embeddings WHERE model = ? AND chunk_id IN "
                        "(SELECT chunk_id FROM chunks WHERE chunk_config = ?)", [model_tag(dim), config])
            bulk_insert(con, "embeddings", ["chunk_id", "model", "dim", "vec_row", "centroid_id"],
                [(cid, model_tag(dim), dim, i, int(ivf.assign[i])) for i, (cid, _) in enumerate(rows)])
            print(f"[{config}/{dim}] IVF {n_lists} lists built in "
                  f"{times[f'{config}.{dim}.ivf_build_s']:.2f}s")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "build_times.json").write_text(json.dumps(times, indent=2))
    print(f"{load_questions(con)} eval questions loaded")
    con.close()


if __name__ == "__main__":
    main()
