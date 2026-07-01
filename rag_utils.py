"""
rag_utils.py — Local, dependency-light RAG layer for the pipeline's reports.

Uses fastembed (ONNX, no torch, ~130MB model, downloads once from
Hugging Face on first run) to embed report text, and DuckDB's built-in
array_cosine_similarity() to do vector search — no separate vector DB
needed, it lives right next to the retail_clean table in the same
warehouse file.
"""
import duckdb
from datetime import datetime
from fastembed import TextEmbedding

EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384

_model = None


def get_embedder():
    global _model
    if _model is None:
        print("      [RAG] Loading local embedding model (first run downloads it)...")
        _model = TextEmbedding(model_name=EMBED_MODEL_NAME)
    return _model


def embed_text(text: str):
    model = get_embedder()
    return list(model.embed([text]))[0].tolist()


def ensure_table(con):
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS report_embeddings (
            report_id   VARCHAR,
            report_text VARCHAR,
            embedding   FLOAT[{EMBED_DIM}],
            created_at  TIMESTAMP
        )
    """)


def store_report_embedding(duckdb_path: str, report_id: str, report_text: str):
    """Embed a report's text and store it for future semantic search."""
    embedding = embed_text(report_text)
    con = duckdb.connect(duckdb_path)
    try:
        ensure_table(con)
        con.execute(
            "INSERT INTO report_embeddings VALUES (?, ?, ?, ?)",
            [report_id, report_text, embedding, datetime.now()]
        )
    finally:
        con.close()


def search_similar_reports(duckdb_path: str, query: str, top_k: int = 3):
    """Return the top_k most semantically similar past reports to `query`."""
    query_embedding = embed_text(query)
    con = duckdb.connect(duckdb_path)
    try:
        ensure_table(con)
        rows = con.execute(f"""
            SELECT report_id,
            report_text,
            array_cosine_similarity(embedding, ?::FLOAT[{EMBED_DIM}]) AS score
            FROM report_embeddings
            ORDER BY score DESC
            LIMIT ?
        """, [query_embedding, top_k]).fetchall()
    finally:
        con.close()

    return [
        {"report_id": r[0], "preview": r[1][:400], "similarity": round(r[2], 3)}
        for r in rows
    ]