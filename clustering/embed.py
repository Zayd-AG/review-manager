"""Embed normalized reviews and upsert them into PostgreSQL with pgvector.

Run from the project root:
    python clustering/embed.py

The PostgreSQL server must have the pgvector extension available. The schema
below is created automatically when the script runs.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import execute_values


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "reviews_normalized.jsonl"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSIONS = 384

load_dotenv(PROJECT_ROOT / ".env")
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".hf_cache"))

from sentence_transformers import SentenceTransformer

REVIEWS_SCHEMA_SQL = f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS reviews (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    source TEXT NOT NULL,
    app_name TEXT NOT NULL,
    rating INTEGER,
    date TIMESTAMPTZ,
    embedding vector({EMBEDDING_DIMENSIONS})
);

ALTER TABLE reviews
    ADD COLUMN IF NOT EXISTS embedding vector({EMBEDDING_DIMENSIONS});

CREATE INDEX IF NOT EXISTS reviews_embedding_cosine_idx
    ON reviews USING hnsw (embedding vector_cosine_ops);
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    return args


def load_reviews(path: Path) -> list[dict[str, Any]]:
    reviews: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            review = json.loads(line)
            if not isinstance(review, dict):
                raise ValueError(f"Expected an object on line {line_number}")
            if not isinstance(review.get("id"), str) or not isinstance(
                review.get("text"), str
            ):
                raise ValueError(f"Review on line {line_number} needs string id and text")
            reviews.append(review)
    if not reviews:
        raise ValueError(f"No reviews found in {path}")
    return reviews


def vector_literal(vector: Any) -> str:
    """Return pgvector's text representation without requiring another package."""
    return "[" + ",".join(f"{float(value):.8f}" for value in vector) + "]"


def ensure_reviews_schema(connection: Any) -> None:
    with connection.cursor() as cursor:
        cursor.execute(REVIEWS_SCHEMA_SQL)
    connection.commit()


def upsert_batch(
    connection: Any, reviews: list[dict[str, Any]], embeddings: Any
) -> None:
    values = [
        (
            review["id"],
            review["text"],
            review.get("source"),
            review.get("app_name"),
            review.get("rating"),
            review.get("date"),
            vector_literal(embedding),
        )
        for review, embedding in zip(reviews, embeddings, strict=True)
    ]
    statement = """
        INSERT INTO reviews (id, text, source, app_name, rating, date, embedding)
        VALUES %s
        ON CONFLICT (id) DO UPDATE SET
            text = EXCLUDED.text,
            source = EXCLUDED.source,
            app_name = EXCLUDED.app_name,
            rating = EXCLUDED.rating,
            date = EXCLUDED.date,
            embedding = EXCLUDED.embedding
    """
    with connection.cursor() as cursor:
        execute_values(
            cursor,
            statement,
            values,
            template="(%s, %s, %s, %s, %s, %s, %s::vector)",
            page_size=len(values),
        )
    connection.commit()


def main() -> None:
    args = parse_args()
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is required in .env")

    reviews = load_reviews(INPUT_PATH)
    model = SentenceTransformer(args.model)
    connection = psycopg2.connect(database_url)
    try:
        ensure_reviews_schema(connection)
        for start in range(0, len(reviews), args.batch_size):
            batch = reviews[start : start + args.batch_size]
            embeddings = model.encode(
                [str(review["text"]) for review in batch],
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            upsert_batch(connection, batch, embeddings)
            print(f"Embedded and saved {min(start + len(batch), len(reviews))}/{len(reviews)}")
    finally:
        connection.close()

    print(
        f"Done: stored {len(reviews)} normalized {EMBEDDING_DIMENSIONS}-dimension "
        f"embeddings using {args.model}."
    )


if __name__ == "__main__":
    main()
