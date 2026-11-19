"""DuckDB catalog: the four tables from the proposal, with their keys.

Vectors themselves live in data/vectors/*.npy (row-aligned with
embeddings.vec_row) and the IVF lists in data/index/*.npz, so — as the
project spec allows when the index holds vectors — `embeddings` has no
vector column. It records which model/dim produced each chunk's vector,
where that vector is, and which IVF list (centroid) it was assigned to.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "recipes.duckdb"

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id         VARCHAR PRIMARY KEY,      -- Wikibooks page slug, e.g. Corn_Soufflé
    source         VARCHAR NOT NULL,         -- 'wikibooks-cookbook'
    source_version BIGINT  NOT NULL,         -- MediaWiki revision id of the snapshot
    title          VARCHAR NOT NULL,
    url            VARCHAR NOT NULL,
    category       VARCHAR,                  -- infobox category, e.g. 'Soup recipes'
    difficulty     INTEGER,                  -- infobox 1-5
    servings       VARCHAR,
    total_time     VARCHAR,                  -- infobox text, as written
    total_minutes  INTEGER,                  -- parsed from total_time (upper bound)
    categories     VARCHAR[],                -- wiki categories ('Recipes using cassava', ...)
    ingested_at    TIMESTAMP NOT NULL DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id     VARCHAR PRIMARY KEY,        -- doc_id:version:config:ordinal
    doc_id       VARCHAR NOT NULL REFERENCES documents(doc_id),
    ordinal      INTEGER NOT NULL,           -- position within the document
    section      VARCHAR,                    -- heading the chunk came from
    chunk_config VARCHAR NOT NULL,           -- 'whole' | 'section'
    text         VARCHAR NOT NULL,           -- exactly what was embedded
    n_tokens     INTEGER NOT NULL,           -- model word-piece count (before truncation)
    UNIQUE (doc_id, chunk_config, ordinal)
);

CREATE TABLE IF NOT EXISTS embeddings (
    chunk_id    VARCHAR NOT NULL REFERENCES chunks(chunk_id),
    model       VARCHAR NOT NULL,            -- 'all-MiniLM-L6-v2' | 'all-MiniLM-L6-v2+pca64'
    dim         INTEGER NOT NULL,
    vec_row     INTEGER NOT NULL,            -- row in data/vectors/<config>_<dim>.npy
    centroid_id INTEGER,                     -- IVF list of the default index build
    PRIMARY KEY (chunk_id, model)
);

CREATE TABLE IF NOT EXISTS eval_questions (
    q_id             VARCHAR PRIMARY KEY,
    question         VARCHAR NOT NULL,
    q_type           VARCHAR NOT NULL,       -- lookup | procedure | ingredient | constrained
    relevant_doc_ids VARCHAR[] NOT NULL,     -- every entry must exist in documents
    author           VARCHAR NOT NULL,       -- team member (or partner team) who wrote it
    source_team      VARCHAR NOT NULL DEFAULT 'ours'
);
"""


def bulk_insert(con: duckdb.DuckDBPyConnection, table: str, columns: list[str],
                rows: list[tuple]) -> None:
    """Fast insert: rows -> temp NDJSON -> INSERT ... SELECT (executemany is
    row-at-a-time in DuckDB and takes minutes for ~50k rows)."""
    import json
    import tempfile
    if not rows:
        return
    with tempfile.NamedTemporaryFile("w", suffix=".ndjson", delete=False, encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(dict(zip(columns, r)), ensure_ascii=False) + "\n")
        path = f.name
    cols = ", ".join(columns)
    types = dict(con.execute(
        "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = ?",
        [table]).fetchall())
    spec = "{" + ", ".join(f"'{c}': '{types[c]}'" for c in columns) + "}"
    con.execute(f"INSERT INTO {table} ({cols}) SELECT {cols} FROM read_json(?, columns = {spec}, format = 'newline_delimited') "
                + ("ON CONFLICT DO NOTHING" if table == "documents" else ""), [path])
    Path(path).unlink()


def connect(path: Path | str = DB_PATH, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path), read_only=read_only)
    if not read_only:
        con.execute(SCHEMA)
    return con
