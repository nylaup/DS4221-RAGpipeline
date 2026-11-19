# RAG over Recipes

DS 4221 final project. Wissal Khlouf, Liam Ward, Nyla Upal, Kalenga Mumba.

Ask a cooking question and get an answer grounded in the Wikibooks Cookbook (about 3,800 recipes, CC BY-SA 4.0), with citations to the chunks the answer came from.

```
uv run python -m recipe_rag ask "What oven temperature does the corn soufflé use?"
```

## Reproduce from scratch

Needs Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                   # install dependencies
uv run python -m recipe_rag.fetch         # corpus snapshot (skipped: already in data/raw/)
uv run python -m recipe_rag.build --fresh # tables, chunks, embeddings, IVF indexes (~5 min on a laptop CPU)
uv run pytest -q                          # unit tests (index == Lab 9 reference, parsing, filters)
```

To generate answers, point the pipeline at an OpenAI-compatible endpoint. That can be the course endpoint or a local Ollama:

```bash
export LLM_BASE_URL=...   # course endpoint, or http://localhost:11434/v1
export LLM_API_KEY=...    # anything for Ollama
export LLM_MODEL=...      # e.g. the course Gemma 4 E4B id, or gemma4:e4b
```

Without these variables, `ask` runs in echo mode and prints the prompt it would send.

## Commands

| Command | What it does |
|---|---|
| `python -m recipe_rag ask "<q>"` | retrieve the top 5 chunks, generate an answer, list the cited recipes |
| `python -m recipe_rag search "<q>"` | retrieval only: the top-k chunks with scores and latency |
| `python -m recipe_rag find <term>` | keyword search over titles, categories, and text, for **labeling** `relevant_doc_ids` |
| `python -m recipe_rag export-partner [q_ids…]` | writes `results/partner_swap.json` for the question swap |
| `python -m recipe_rag.build --questions-only` | reload the eval question files into DuckDB |
| `python -m recipe_rag.evaluate retrieval` | recall@k and MRR (k = 1, 5, 10) for every configuration |
| `python -m recipe_rag.evaluate ann` | IVF probe sweep: recall, comparisons, latency, and QPS against brute force |
| `python -m recipe_rag.evaluate answers` | generate answers, score them with an LLM judge, and write a spot-check sheet |

Retrieval options are `--config section|whole`, `--dim 384|64`, `--index ivf|brute`, `--probe N`, and `--no-filter`.

## Architecture

```
data/raw/cookbook.jsonl.gz      frozen snapshot (MediaWiki API, Category:Recipes, revision ids kept)
        │  wikitext.py          our parser: infobox, sections, categories, markup cleanup, time → minutes
        ▼
documents ──► chunk.py          'whole' (1/recipe) or 'section' (per heading, title-prefixed, ≤180 words)
        ▼
chunks ──► embed.py             all-MiniLM-L6-v2, 384-d unit vectors  ──► PCA → 64-d copy
        ▼
data/vectors/*.npy ──► index.py IVF (k-means lists, probe P nearest) | BruteForce baseline
        ▼
retrieve.py                     top-k chunks; for constrained questions, a DuckDB-metadata filter mask
        ▼
generate.py                     prompt with chunk ids → LLM → citations parsed → joined to documents
```

**Tables (DuckDB, `data/recipes.duckdb`, schema in `recipe_rag/db.py`):**

- `documents`: PK `doc_id` (page slug).
- `chunks`: PK `chunk_id = doc_id:revision:config:ordinal`, FK `doc_id`.
- `embeddings`: PK `(chunk_id, model)`, FK `chunk_id`. It stores `vec_row` (the row in the `.npy` file) and `centroid_id` (the IVF list). The vectors themselves live in the index files, which the spec allows.
- `eval_questions`: PK `q_id`. The build checks that every `relevant_doc_ids` entry exists in `documents`.

**The index is our Lab 9 microvector.** `recipe_rag/microvector_reference.py` is the Lab 9 file, unchanged. `recipe_rag/index.py` runs the same algorithm in NumPy: the same seeded k-means, the same assignment, the same probe order, the same tie-breaking, and the same comparison counting. `tests/test_pipeline.py` checks that both return identical results. We ported it because pure-Python k-means over about 15k × 384 vectors takes hours.

## Evaluation questions

Questions live in `data/eval_questions.jsonl` (ours) and `data/partner_questions.jsonl` (the partner team's), one JSON object per line:

```json
{"q_id": "lw-03", "question": "...", "q_type": "lookup|procedure|ingredient|constrained", "relevant_doc_ids": ["Corn_Soufflé"], "author": "Liam Ward"}
```

Use `python -m recipe_rag find <term>` to find **every** recipe that should count as relevant. After editing either file, run `build --questions-only`.

**Status: 4 proposal questions + 28 DRAFTS.** `scripts/draft_questions.py` generates the file. The drafts were written by an AI assistant as a starting point and carry `"author": "DRAFT"`. Our proposal promises human-written questions, so each team member should take about 7 drafts, review or rewrite them, check the labels, and put their own name in `author`. Edit `scripts/draft_questions.py` and re-run it, or edit the JSONL directly. `export-partner` warns if it would send a draft.

How labels are made:

- **Lookup and procedure:** labeled by hand. Every variant of a recipe counts (e.g. Carrot Cake I and II).
- **Ingredient:** the wiki category "Recipes using X". This is incomplete: *Plain_Kefir* is not in "Recipes using kefir". Add any missing recipes when you review.
- **Constrained:** an SQL rule over the infobox category and the parsed time, stored in `label_rule`. The retrieval filter reads the same metadata, so the filtered precision on this type is optimistic by construction. Say so in the report.

## Results so far (Nov 19, 32 questions, drafts not yet reviewed)

Full tables are in `results/`: `retrieval_summary.md`, `retrieval_per_question.csv`, `ann_sweep.md`, and `build_times.json`.

**Retrieval quality (doc level):**

| config | recall@5 | recall@10 | MRR@10 |
|---|---|---|---|
| whole / 384d / brute | 0.704 | 0.781 | 0.922 |
| **section / 384d / brute** | **0.709** | **0.804** | **0.984** |
| section / 64d / brute | 0.671 | 0.758 | 0.857 |
| whole / 64d / brute | 0.633 | 0.730 | 0.839 |
| section / 384d / IVF probe 8 | 0.688 | 0.771 | 0.940 |
| section / 64d / IVF probe 8 | 0.646 | 0.728 | 0.832 |
| section / 384d / brute, **no filter** | 0.624 | 0.666 | 0.783 |

**IVF vs brute force (section chunks, 304 queries, chunk-level recall@10):**

| index | comparisons / query | ms / query | recall@10 |
|---|---|---|---|
| brute force | 13,985 | 1.12 | 1.000 |
| IVF, 118 lists, probe 1 | 258 | 0.05 | 0.457 |
| IVF, probe 4 | 672 | 0.15 | 0.733 |
| IVF, probe 8 | 1,231 | 0.30 | 0.830 |
| IVF, probe 16 | 2,322 | 0.61 | 0.904 |
| IVF, probe 32 | 4,407 | 1.27 | 0.956 |

**Observations to develop in the report:**

- **Truncation:** 64% of whole-recipe chunks are longer than MiniLM's 256-token limit, so the model never sees their procedure. Section chunks fix this, and they win on MRR.
- **Constrained questions need the metadata filter.** Without it, recall@10 on that type is 0.09; similarity alone cannot express "under 30 minutes".
- **Coverage gap:** only 1,455 of 3,801 recipes list a time, so the filter can never return the rest.
- **Sub-word confusion:** "What recipes use agar?" retrieves *Agidi* and *Agbalumo*, because MiniLM matches word pieces of rare words.
- **IVF speed:** at 14k vectors, one NumPy matrix product is already about 1 ms. IVF wins clearly on comparisons, but in wall-clock time only at probe ≤ 16. Its advantage grows with corpus size.
- **Answer quality has not been measured yet.** It needs the course LLM endpoint: run `evaluate answers`.
