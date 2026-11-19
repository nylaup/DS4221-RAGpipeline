"""Unit tests that need no corpus, model, or LLM:  uv run pytest -q"""

import random

import numpy as np

from recipe_rag import microvector_reference as ref
from recipe_rag.chunk import MAX_WORDS, chunk
from recipe_rag.index import BruteForceIndex, IVFIndex, kmeans, normalize_rows
from recipe_rag.retrieve import parse_filters
from recipe_rag.wikitext import clean, parse_minutes, parse_recipe


def _vectors(n=300, d=16, seed=1):
    rng = np.random.default_rng(seed)
    return normalize_rows(rng.normal(size=(n, d)))


# ---------------------------------------------- index == Lab 9 reference

def test_kmeans_matches_reference():
    v = _vectors()
    ours = kmeans(v, 10, iters=5)
    theirs = np.array(ref.kmeans(v.tolist(), 10, iters=5))
    assert np.allclose(ours, theirs, atol=1e-9)


def test_ivf_search_matches_reference():
    v = _vectors()
    q = _vectors(20, seed=2)
    ours, theirs = IVFIndex(v, n_lists=10, iters=5), ref.IVFIndex(v.tolist(), n_clusters=10)
    for probe in (1, 3, 10):
        for qv in q:
            a = ours.search(qv, 5, probe=probe)
            b = theirs.search(qv.tolist(), 5, probe=probe)
            assert [i for _, i in a] == [i for _, i in b]
    assert ours.comparisons == theirs.comparisons


def test_brute_matches_reference_and_filter():
    v = _vectors()
    q = _vectors(5, seed=3)
    ours, theirs = BruteForceIndex(v), ref.BruteForceIndex(v.tolist())
    for qv in q:
        assert [i for _, i in ours.search(qv, 7)] == [i for _, i in theirs.search(qv.tolist(), 7)]
    allowed = np.zeros(len(v), bool)
    allowed[::7] = True
    hits = ours.search(q[0], 5, allowed=allowed)
    assert all(i % 7 == 0 for _, i in hits) and len(hits) == 5


def test_ivf_filtered_search_widens_probe():
    v = _vectors()
    ivf = IVFIndex(v, n_lists=10)
    allowed = np.zeros(len(v), bool)
    allowed[[3, 50, 250]] = True              # only 3 admissible ids, scattered
    hits = ivf.search(v[0], 3, probe=1, allowed=allowed)
    assert sorted(i for _, i in hits) == [3, 50, 250]


def test_ivf_full_probe_is_exact():
    v = _vectors()
    ivf, bf = IVFIndex(v, n_lists=10), BruteForceIndex(v)
    for qv in _vectors(10, seed=4):
        assert ivf.search(qv, 10, probe=10) == bf.search(qv, 10)


# ---------------------------------------------- parsing

def test_parse_minutes():
    assert parse_minutes("Prep: 20 minutes; Baking: 60 minutes") == 80
    assert parse_minutes("1 hour 30 minutes") == 90
    assert parse_minutes("20–30 minutes") == 30          # range -> upper end
    assert parse_minutes("About 1½ hours") == 90
    assert parse_minutes("overnight") is None
    assert parse_minutes(clean("Prep: 10 minutes{{br}}Cooking: 30 minutes")) == 40
    assert parse_minutes(clean("Prep: 5\nminutes</br>Cooking: 10 minutes</br>Total: 15 minutes")) == 15


def test_clean_links_templates():
    s = clean("* 1 [[Cookbook:Cup|cup]] {{convert|8|oz|g|abbr=on}} [[Cookbook:Butter]] '''bold'''")
    assert s == "- 1 cup 8 oz Butter bold"
    assert clean("#Preheat\n#Bake") == "1. Preheat\n2. Bake"


SAMPLE = """{{recipesummary|category=Soup recipes|servings=4|time=25 minutes|difficulty=2}}
{{recipe}}
A quick [[Cookbook:Soup|soup]] for weeknights, made with leeks and potatoes.
==Ingredients==
* 2 [[Cookbook:Leek|leeks]]
* 3 potatoes, diced into small cubes for quick cooking
==Procedure==
# Sweat the leeks in butter over low heat until soft.
# Add potatoes and stock, then simmer for 20 minutes until tender.
==See also==
* [[Cookbook:Potato]]
[[Category:Recipes using leek]]
"""


def test_parse_recipe_and_chunk():
    r = parse_recipe("Cookbook:Leek Soup", SAMPLE)
    assert r["title"] == "Leek Soup" and r["category"] == "Soup recipes"
    assert r["total_minutes"] == 25 and r["difficulty"] == 2
    assert "Recipes using leek" in r["categories"]
    names = [s["section"] for s in r["sections"]]
    assert names[0] == "Introduction" and "Procedure" in names
    sec = chunk(r, "section")
    assert sec[0]["text"].startswith("Recipe: Leek Soup\nCategory: Soup recipes")
    assert all(c["text"].startswith("Recipe: Leek Soup") for c in sec)
    assert not any("See also" in (c["section"] or "") for c in sec)
    assert all(len(c["text"].split()) <= MAX_WORDS + 40 for c in sec)
    assert len(chunk(r, "whole")) == 1


def test_parse_filters():
    keys = {"soup": "soup", "salad": "salad"}
    f = parse_filters("Show me soup under 30 minutes", keys)
    assert f.max_minutes == 29 and f.category_like == "soup"
    f = parse_filters("easy salads within an hour", keys)
    assert f.max_minutes == 60 and f.max_difficulty == 2 and f.category_like == "salad"
    f = parse_filters("Easy salad that takes 15 minutes or less", keys)
    assert f.max_minutes == 15 and f.max_difficulty == 2 and f.category_like == "salad"
    f = parse_filters("How do I make Nigerian mashed taro?", keys)
    assert not f.active() and f.category_like is None
