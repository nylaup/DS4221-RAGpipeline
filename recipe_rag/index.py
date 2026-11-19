"""Vector indexes: our Lab 9 microvector, ported to NumPy.

microvector_reference.py (Lab 9, pure Python) is the specification. This file
runs the SAME algorithms — seeded k-means with the same initial sample, the
same argmax assignment, the same probe selection, the same tie-breaking (score
descending, then id ascending), and the same comparison counting — with the
dot products done as NumPy matrix products. tests/test_index.py checks that
both give identical results on a sample. The port exists because pure-Python
k-means over ~20k x 384 vectors takes hours; NumPy takes seconds.

Two additions over Lab 9, both needed for the recipe questions:
  * `allowed`: a boolean mask of ids a metadata filter admits (e.g. "soups
    under 30 minutes"). Brute force just skips the rest. IVF post-filters
    inside the probed lists and, if that leaves fewer than k hits, keeps
    probing further lists (in centroid order) until it has k or runs out.
  * save()/load(): the lists and centroids persist in data/index/*.npz, so
    the index is built once, not per query.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np


def normalize_rows(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


def _topk(scores: np.ndarray, ids: np.ndarray, k: int) -> list[tuple[float, int]]:
    """Best k (score, id), score descending then id ascending — heapq.nsmallest
    with key (-score, id), as in microvector."""
    if len(ids) == 0:
        return []
    order = np.lexsort((ids, -scores))[:k]
    return [(float(scores[i]), int(ids[i])) for i in order]


# ---------------------------------------------------------------- exact

class BruteForceIndex:
    """Exact baseline: score the query against every vector."""

    kind = "brute"

    def __init__(self, vectors: np.ndarray):
        self.vectors = np.ascontiguousarray(vectors)
        self.comparisons = 0

    def search(self, query: np.ndarray, k: int, allowed: np.ndarray | None = None,
               **_) -> list[tuple[float, int]]:
        if allowed is None:
            ids = np.arange(len(self.vectors))
            scores = self.vectors @ query
        else:
            ids = np.flatnonzero(allowed)
            scores = self.vectors[ids] @ query
        self.comparisons += len(ids)
        return _topk(scores, ids, k)


# ---------------------------------------------------------------- IVF

def kmeans(vectors: np.ndarray, n_clusters: int, iters: int = 5,
           seed: int = 6042) -> np.ndarray:
    """microvector.kmeans: seeded sample of starting centroids, then
    assign-to-nearest / average-and-normalize, `iters` times. Empty
    clusters keep their previous centroid."""
    rng = random.Random(seed)
    start = rng.sample(range(len(vectors)), n_clusters)   # same picks as rng.sample(vectors, C)
    centroids = vectors[start].copy()
    for _ in range(iters):
        assign = np.argmax(vectors @ centroids.T, axis=1)
        for c in range(n_clusters):
            members = vectors[assign == c]
            if len(members):
                mean = members.mean(axis=0)
                centroids[c] = mean / (np.linalg.norm(mean) or 1.0)
    return centroids


class IVFIndex:
    """Inverted file: k-means the vectors into `n_lists` lists; a query
    scores every centroid, then brute-forces the `probe` nearest lists."""

    kind = "ivf"

    def __init__(self, vectors: np.ndarray, n_lists: int = 64, iters: int = 5,
                 seed: int = 6042, centroids: np.ndarray | None = None,
                 assign: np.ndarray | None = None):
        self.vectors = np.ascontiguousarray(vectors)
        self.centroids = kmeans(self.vectors, n_lists, iters, seed) if centroids is None else centroids
        if assign is None:
            assign = np.argmax(self.vectors @ self.centroids.T, axis=1)
        self.assign = assign.astype(np.int32)
        self.lists = [np.flatnonzero(self.assign == c) for c in range(len(self.centroids))]
        self.comparisons = 0

    @property
    def n_lists(self) -> int:
        return len(self.centroids)

    def search(self, query: np.ndarray, k: int, probe: int = 1,
               allowed: np.ndarray | None = None) -> list[tuple[float, int]]:
        cscores = self.centroids @ query
        self.comparisons += self.n_lists
        order = np.lexsort((np.arange(self.n_lists), -cscores))
        chosen = list(order[:probe])
        cand = np.concatenate([self.lists[c] for c in chosen]) if chosen else np.array([], int)
        if allowed is not None:
            cand = cand[allowed[cand]]
            # filtered search: widen the probe until k admitted candidates exist
            nxt = probe
            while len(cand) < k and nxt < self.n_lists:
                extra = self.lists[order[nxt]]
                cand = np.concatenate([cand, extra[allowed[extra]]])
                nxt += 1
        scores = self.vectors[cand] @ query
        self.comparisons += len(cand)
        return _topk(scores, cand, k)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, centroids=self.centroids, assign=self.assign)

    @classmethod
    def load(cls, path: Path, vectors: np.ndarray) -> "IVFIndex":
        z = np.load(path)
        return cls(vectors, centroids=z["centroids"], assign=z["assign"])


# ---------------------------------------------------------------- PCA

class PCA:
    """PCA by SVD on the centered chunk vectors; outputs are re-normalized
    so dot product stays cosine similarity (as in Lab 9's 64-d copy)."""

    def __init__(self, dim: int):
        self.dim = dim
        self.mean: np.ndarray | None = None
        self.components: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "PCA":
        self.mean = x.mean(axis=0)
        _, _, vt = np.linalg.svd(x - self.mean, full_matrices=False)
        self.components = vt[:self.dim]
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        single = x.ndim == 1
        x2 = np.atleast_2d(x)
        out = normalize_rows((x2 - self.mean) @ self.components.T).astype(np.float32)
        return out[0] if single else out

    def save(self, path: Path) -> None:
        np.savez(path, mean=self.mean, components=self.components)

    @classmethod
    def load(cls, path: Path) -> "PCA":
        z = np.load(path)
        p = cls(z["components"].shape[0])
        p.mean, p.components = z["mean"], z["components"]
        return p


def recall_at_k(approx: list[tuple[float, int]], exact: list[tuple[float, int]], k: int) -> float:
    """Fraction of the exact top-k the approximate search found (microvector)."""
    truth = {i for _, i in exact[:k]}
    found = {i for _, i in approx[:k]}
    return len(truth & found) / k
