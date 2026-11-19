"""Our chunker. Two configurations, compared in the evaluation:

whole    one chunk per recipe: header + every section. Simple, keeps the
         recipe together, but all-MiniLM-L6-v2 truncates input at 256
         word pieces, so long recipes lose their procedure from the vector.
section  one chunk per section (Introduction, Ingredients, Procedure, ...),
         each prefixed with the recipe title and section name so a chunk
         about "Bake at 175 °C" still knows it belongs to "1-2-3-4 Cake".
         Sections over MAX_WORDS split at line boundaries (list items stay
         whole); sections under MIN_WORDS merge into the next one.

Both configs put a metadata header (category, time, difficulty, servings)
in the first chunk, so "soup under 30 minutes" has text to match.

chunk_id = doc_id:source_version:config:ordinal — stable across runs as long
as the snapshot (revision id) and the config are unchanged.
"""

from __future__ import annotations

import re

MAX_WORDS = 180     # ~240 word pieces: under MiniLM's 256 limit with the prefix
MIN_WORDS = 12
SKIP_SECTIONS = re.compile(r"(?i)^(see also|external links?|references?|sources?|"
                           r"further reading|related recipes?|gallery)$")
CONFIGS = ("whole", "section")


def header(doc: dict) -> str:
    bits = []
    if doc.get("category"):
        bits.append(f"Category: {doc['category']}")
    if doc.get("time_text"):
        bits.append(f"Time: {doc['time_text']}")
    if doc.get("difficulty"):
        bits.append(f"Difficulty: {doc['difficulty']}/5")
    if doc.get("servings"):
        bits.append(f"Servings: {doc['servings']}")
    return f"Recipe: {doc['title']}" + ("\n" + " | ".join(bits) if bits else "")


def _split_long(text: str, max_words: int) -> list[str]:
    parts, cur, n = [], [], 0
    for line in text.splitlines():
        w = len(line.split())
        if cur and n + w > max_words:
            parts.append("\n".join(cur).strip())
            cur, n = [], 0
        if w > max_words:                       # one giant paragraph: split on sentences
            for sent in re.split(r"(?<=[.!?])\s+", line):
                sw = len(sent.split())
                if cur and n + sw > max_words:
                    parts.append("\n".join(cur).strip())
                    cur, n = [], 0
                cur.append(sent); n += sw
            continue
        cur.append(line); n += w
    if cur and "\n".join(cur).strip():
        parts.append("\n".join(cur).strip())
    return [p for p in parts if p]


def chunk_whole(doc: dict) -> list[dict]:
    body = "\n\n".join(f"{s['section']}:\n{s['text']}" for s in doc["sections"]
                       if not SKIP_SECTIONS.match(s["section"]))
    return [{"section": None, "text": header(doc) + "\n\n" + body}]


def chunk_sections(doc: dict) -> list[dict]:
    secs = [dict(s) for s in doc["sections"] if not SKIP_SECTIONS.match(s["section"])]
    merged: list[dict] = []
    carry = None
    for s in secs:
        if carry:
            s = {"section": f"{carry['section']} / {s['section']}",
                 "text": carry["text"] + "\n" + s["text"]}
            carry = None
        if len(s["text"].split()) < MIN_WORDS:
            carry = s
            continue
        merged.append(s)
    if carry:
        if merged:
            merged[-1]["text"] += "\n" + carry["text"]
        else:
            merged.append(carry)

    out = []
    for i, s in enumerate(merged):
        pieces = _split_long(s["text"], MAX_WORDS)
        for j, piece in enumerate(pieces):
            label = s["section"] + (f" (part {j + 1})" if len(pieces) > 1 else "")
            prefix = header(doc) if not out else f"Recipe: {doc['title']}"
            out.append({"section": label, "text": f"{prefix}\nSection: {label}\n{piece}"})
    if not out:
        out = [{"section": None, "text": header(doc)}]
    return out


def chunk(doc: dict, config: str) -> list[dict]:
    if config == "whole":
        return chunk_whole(doc)
    if config == "section":
        return chunk_sections(doc)
    raise ValueError(config)
