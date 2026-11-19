"""Retrieval: question -> top-k chunks (and a doc ranking), with optional
metadata filters for constrained questions.

    r = Retriever(config="section", dim=384, index="ivf", probe=8)
    hits = r.retrieve("What oven temperature does the corn soufflé use?", k=5)

Constrained questions ("soup under 30 minutes") need more than similarity:
an embedding cannot compare 20 < 30. parse_filters() pulls a time limit and,
when a limit is present, a recipe category out of the question; DuckDB turns
those into an allowed-id mask that the index honors (index.py). Filtering is
on by default and can be switched off (use_filters=False) for comparison.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import numpy as np

from .build import ivf_path, model_tag, pca_path, vec_path
from .db import connect
from .embed import embed_one
from .index import PCA, BruteForceIndex, IVFIndex


# ------------------------------------------------------------ filters

@dataclass
class Filters:
    max_minutes: int | None = None
    max_difficulty: int | None = None
    category_like: str | None = None
    notes: list[str] = field(default_factory=list)

    def active(self) -> bool:
        return self.max_minutes is not None or self.max_difficulty is not None


_TIME = re.compile(r"(?:under|less than|within|in|at most|no more than|max(?:imum)?|<)\s*"
                   r"(?:an?\s+)?(\d+(?:\.\d+)?)?\s*(minutes?|mins?|hours?|hrs?|hour)\b", re.I)


_TIME_AFTER = re.compile(r"(\d+(?:\.\d+)?)\s*(minutes?|mins?|hours?|hrs?)\s+or\s+(?:less|fewer|under)\b", re.I)


def parse_filters(question: str, category_keys: dict[str, str]) -> Filters:
    f = Filters()
    m = _TIME.search(question)
    m2 = None if m else _TIME_AFTER.search(question)
    if m2:
        n = float(m2.group(1))
        f.max_minutes = int(n * (60 if m2.group(2).lower().startswith("h") else 1))
        f.notes.append(f"total_minutes <= {f.max_minutes}")
        m = m2
    elif m:
        n = float(m.group(1)) if m.group(1) else 1.0
        f.max_minutes = int(n * (60 if m.group(2).lower().startswith(("h",)) else 1))
        if re.match(r"(?i)under|less than|<", m.group(0)):
            f.max_minutes -= 1            # strict: "under 30" excludes exactly 30
        f.notes.append(f"total_minutes <= {f.max_minutes}")
    if re.search(r"\b(easy|simple|beginner)\b", question, re.I) and m:
        f.max_difficulty = 2
        f.notes.append("difficulty <= 2")
    if f.active():
        words = re.findall(r"[a-z]+", question.lower())
        for w in words:
            for cand in (w, w.rstrip("s"), w[:-2] if w.endswith("es") else w):
                if cand in category_keys:
                    f.category_like = category_keys[cand]
                    f.notes.append(f"category ILIKE '%{f.category_like}%'")
                    break
            if f.category_like:
                break
    return f


# ------------------------------------------------------------ retriever

class Retriever:
    def __init__(self, config: str = "section", dim: int = 384, index: str = "ivf",
                 probe: int = 8, use_filters: bool = True, con=None):
        self.config, self.dim, self.kind, self.probe = config, dim, index, probe
        self.use_filters = use_filters
        self.con = con or connect(read_only=True)
        rows = self.con.execute("""
            SELECT c.chunk_id, c.doc_id, c.section, c.text, d.title, d.url,
                   d.total_minutes, d.difficulty, d.category
            FROM embeddings e JOIN chunks c USING (chunk_id) JOIN documents d USING (doc_id)
            WHERE c.chunk_config = ? AND e.model = ?
            ORDER BY e.vec_row""", [config, model_tag(dim)]).fetchall()
        if not rows:
            raise SystemExit(f"no embeddings for config={config} dim={dim}; run the build first")
        cols = ["chunk_id", "doc_id", "section", "text", "title", "url",
                "total_minutes", "difficulty", "category"]
        self.chunks = [dict(zip(cols, r)) for r in rows]
        self.vectors = np.load(vec_path(config, dim))
        assert len(self.vectors) == len(self.chunks), "vectors and chunk table out of sync; rebuild"
        self.pca = PCA.load(pca_path(config)) if dim != 384 else None
        if index == "ivf":
            self.index = IVFIndex.load(ivf_path(config, dim), self.vectors)
        elif index == "brute":
            self.index = BruteForceIndex(self.vectors)
        else:
            raise ValueError(index)
        self._minutes = np.array([c["total_minutes"] if c["total_minutes"] is not None else -1
                                  for c in self.chunks])
        self._difficulty = np.array([c["difficulty"] or 0 for c in self.chunks])
        self._category = np.array([(c["category"] or "").lower() for c in self.chunks])
        self.category_keys = self._category_vocab()
        embed_one("warm up")          # load the model now, not inside the first query's latency

    def _category_vocab(self) -> dict[str, str]:
        """'soup' -> 'soup' for infobox categories like 'Soup recipes'."""
        keys = {}
        for (cat,) in self.con.execute(
                "SELECT DISTINCT lower(category) FROM documents WHERE category IS NOT NULL").fetchall():
            m = (re.fullmatch(r"([a-z]+) recipes?", cat.strip())
                 or re.fullmatch(r"recipes? for ([a-z]+?)s?", cat.strip()))
            if m:
                keys[m.group(1)] = m.group(1)
                keys[m.group(1).rstrip("s")] = m.group(1)
        return keys

    def embed_query(self, question: str) -> np.ndarray:
        q = embed_one(question)
        return self.pca.transform(q) if self.pca is not None else q

    def allowed_mask(self, f: Filters) -> np.ndarray | None:
        if not f.active():
            return None
        mask = np.ones(len(self.chunks), bool)
        if f.max_minutes is not None:
            mask &= (self._minutes >= 0) & (self._minutes <= f.max_minutes)
        if f.max_difficulty is not None:
            mask &= (self._difficulty > 0) & (self._difficulty <= f.max_difficulty)
        if f.category_like:
            cat_mask = np.char.find(self._category.astype(str), f.category_like) >= 0
            if (mask & cat_mask).any():
                mask &= cat_mask
            else:
                f.notes.append("(category filter dropped: no matches)")
        return mask if mask.any() else None

    def search(self, qv: np.ndarray, k: int, allowed=None) -> list[tuple[float, int]]:
        if self.kind == "ivf":
            return self.index.search(qv, k, probe=self.probe, allowed=allowed)
        return self.index.search(qv, k, allowed=allowed)

    def retrieve(self, question: str, k: int = 5, qv: np.ndarray | None = None) -> dict:
        """Top-k chunks plus a doc ranking (docs ordered by their best chunk,
        from a deeper top-50 chunk list so recall@10 over docs is defined)."""
        t0 = time.perf_counter()
        if qv is None:
            qv = self.embed_query(question)
        t1 = time.perf_counter()
        f = parse_filters(question, self.category_keys) if self.use_filters else Filters()
        allowed = self.allowed_mask(f)
        hits = self.search(qv, max(k, 50), allowed)
        t2 = time.perf_counter()
        chunks = [{**self.chunks[i], "score": s} for s, i in hits[:k]]
        docs: list[str] = []
        for _, i in hits:
            d = self.chunks[i]["doc_id"]
            if d not in docs:
                docs.append(d)
        return {"chunks": chunks, "doc_ranking": docs, "filters": f.notes,
                "embed_ms": (t1 - t0) * 1e3, "search_ms": (t2 - t1) * 1e3}
