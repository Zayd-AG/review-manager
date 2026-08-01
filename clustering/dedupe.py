"""Cluster near-duplicate reviews already embedded in PostgreSQL with pgvector.

Run after clustering/embed.py:
    python clustering/dedupe.py --threshold 0.85
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import execute_values


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_THRESHOLD = 0.85

CLUSTERS_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS clusters (
    id TEXT PRIMARY KEY,
    similarity_threshold REAL NOT NULL,
    representative_review_id TEXT NOT NULL REFERENCES reviews(id),
    representative_text TEXT NOT NULL,
    review_count INTEGER NOT NULL,
    review_ids JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS cluster_members (
    cluster_id TEXT NOT NULL REFERENCES clusters(id) ON DELETE CASCADE,
    review_id TEXT NOT NULL REFERENCES reviews(id),
    PRIMARY KEY (cluster_id, review_id)
);
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args()
    if not 0 < args.threshold <= 1:
        parser.error("--threshold must be greater than 0 and no more than 1")
    return args


def ensure_cluster_schema(connection: Any) -> None:
    with connection.cursor() as cursor:
        cursor.execute(CLUSTERS_SCHEMA_SQL)
    connection.commit()


def find_similarity_edges(connection: Any, threshold: float) -> list[tuple[str, str]]:
    """Return exact above-threshold review pairs using pgvector cosine distance."""
    statement = """
        SELECT left_review.id, right_review.id
        FROM reviews AS left_review
        JOIN reviews AS right_review ON left_review.id < right_review.id
        WHERE left_review.embedding IS NOT NULL
          AND right_review.embedding IS NOT NULL
          AND 1 - (left_review.embedding <=> right_review.embedding) >= %s
    """
    with connection.cursor() as cursor:
        cursor.execute(statement, (threshold,))
        return [(str(left_id), str(right_id)) for left_id, right_id in cursor.fetchall()]


def connected_components(edges: list[tuple[str, str]]) -> list[list[str]]:
    """Treat matching pairs as an undirected graph and return clusters of size >= 2."""
    neighbours: dict[str, set[str]] = defaultdict(set)
    for left_id, right_id in edges:
        neighbours[left_id].add(right_id)
        neighbours[right_id].add(left_id)

    clusters: list[list[str]] = []
    unseen = set(neighbours)
    while unseen:
        start = unseen.pop()
        component = {start}
        pending = [start]
        while pending:
            review_id = pending.pop()
            new_neighbours = neighbours[review_id] - component
            component.update(new_neighbours)
            pending.extend(new_neighbours)
            unseen.difference_update(new_neighbours)
        clusters.append(sorted(component))
    return clusters


def fetch_reviews(connection: Any, review_ids: list[str]) -> dict[str, tuple[str, np.ndarray]]:
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT id, text, embedding::text FROM reviews WHERE id = ANY(%s)",
            (review_ids,),
        )
        return {
            str(review_id): (str(text), np.asarray(json.loads(vector_text)))
            for review_id, text, vector_text in cursor.fetchall()
        }


def representative_review(
    review_data: dict[str, tuple[str, np.ndarray]], review_ids: list[str]
) -> tuple[str, str]:
    """Pick the review with the highest mean cosine similarity to its cluster."""
    vectors = np.vstack([review_data[review_id][1] for review_id in review_ids])
    similarities = vectors @ vectors.T  # Vectors were normalized during embedding.
    central_index = int(np.argmax(similarities.mean(axis=1)))
    review_id = review_ids[central_index]
    return review_id, review_data[review_id][0]


def cluster_id(threshold: float, review_ids: list[str]) -> str:
    identity = f"{threshold:.6f}:" + ",".join(review_ids)
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def replace_clusters(
    connection: Any, clusters: list[list[str]], threshold: float
) -> None:
    """Replace only clusters generated with this threshold, preserving other runs."""
    with connection.cursor() as cursor:
        cursor.execute(
            "DELETE FROM cluster_members WHERE cluster_id IN "
            "(SELECT id FROM clusters WHERE similarity_threshold = %s)",
            (threshold,),
        )
        cursor.execute(
            "DELETE FROM clusters WHERE similarity_threshold = %s", (threshold,)
        )

        cluster_rows: list[tuple[Any, ...]] = []
        member_rows: list[tuple[str, str]] = []
        for review_ids in clusters:
            review_data = fetch_reviews(connection, review_ids)
            representative_id, representative_text = representative_review(
                review_data, review_ids
            )
            identifier = cluster_id(threshold, review_ids)
            cluster_rows.append(
                (
                    identifier,
                    threshold,
                    representative_id,
                    representative_text,
                    len(review_ids),
                    json.dumps(review_ids),
                )
            )
            member_rows.extend((identifier, review_id) for review_id in review_ids)

        if cluster_rows:
            execute_values(
                cursor,
                """
                INSERT INTO clusters (
                    id, similarity_threshold, representative_review_id,
                    representative_text, review_count, review_ids
                ) VALUES %s
                """,
                cluster_rows,
                template="(%s, %s, %s, %s, %s, %s::jsonb)",
            )
            execute_values(
                cursor,
                "INSERT INTO cluster_members (cluster_id, review_id) VALUES %s",
                member_rows,
            )
    connection.commit()


def main() -> None:
    args = parse_args()
    load_dotenv(PROJECT_ROOT / ".env")
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is required in .env")

    connection = psycopg2.connect(database_url)
    try:
        ensure_cluster_schema(connection)
        edges = find_similarity_edges(connection, args.threshold)
        clusters = connected_components(edges)
        replace_clusters(connection, clusters, args.threshold)
    finally:
        connection.close()

    print(
        f"Saved {len(clusters)} clusters from {len(edges)} matching review pairs "
        f"at cosine similarity >= {args.threshold:.2f}."
    )


if __name__ == "__main__":
    main()
