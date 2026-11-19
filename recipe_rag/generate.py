"""Generation: question + retrieved chunks -> cited answer.

Talks to any OpenAI-compatible chat endpoint, set by environment variables
(same names as Lab 10):

    LLM_BASE_URL   course endpoint, or http://localhost:11434/v1 for Ollama
    LLM_API_KEY    (any string for Ollama)
    LLM_MODEL      e.g. the course's Gemma 4 E4B id, or gemma4:e4b in Ollama

With no LLM_BASE_URL it runs in echo mode and returns the prompt, so the
retrieval half of the pipeline works offline.

The model cites chunk ids in [brackets]; parse_citations() keeps only ids
that were actually in the prompt, and the CLI joins them back to recipes.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request

SYSTEM = ("You are a cooking assistant. Answer ONLY from the recipe excerpts provided. "
          "After every claim, cite the excerpt id in square brackets, e.g. "
          "[Corn_Soufflé:123:section:2]. If the excerpts do not contain the answer, "
          "say you could not find it in the cookbook. Be concise.")


def build_prompt(question: str, chunks: list[dict]) -> str:
    blocks = [f"[{c['chunk_id']}] (recipe: {c['title']})\n{c['text']}" for c in chunks]
    return ("Recipe excerpts:\n\n" + "\n\n---\n\n".join(blocks)
            + f"\n\nQuestion: {question}\nAnswer with citations:")


def llm_configured() -> bool:
    return bool(os.environ.get("LLM_BASE_URL"))


def chat(messages: list[dict], temperature: float = 0.0, timeout: int = 120) -> str:
    base = os.environ["LLM_BASE_URL"].rstrip("/")
    req = urllib.request.Request(
        base + "/chat/completions",
        headers={"Authorization": f"Bearer {os.environ.get('LLM_API_KEY', 'none')}",
                 "Content-Type": "application/json"},
        data=json.dumps({"model": os.environ.get("LLM_MODEL", ""), "messages": messages,
                         "temperature": temperature}).encode())
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)["choices"][0]["message"]["content"].strip()


def generate(question: str, chunks: list[dict]) -> str:
    prompt = build_prompt(question, chunks)
    if not llm_configured():
        return "[echo mode: set LLM_BASE_URL / LLM_API_KEY / LLM_MODEL for a real answer]\n\n" + prompt
    return chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}])


def parse_citations(answer: str, chunks: list[dict]) -> list[str]:
    valid = {c["chunk_id"] for c in chunks}
    seen = []
    for cid in re.findall(r"\[([^\[\]]+)\]", answer):
        for part in re.split(r"[,;]\s*", cid):
            part = part.strip()
            if part in valid and part not in seen:
                seen.append(part)
    return seen
