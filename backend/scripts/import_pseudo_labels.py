"""Import existing teacher pseudo-labels into PostgreSQL and label clusters.

Run from the project root:
    python backend/scripts/import_pseudo_labels.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import execute_values

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LABELS_PATH = PROJECT_ROOT / "data" / "processed" / "pseudo_labels.jsonl"
BATCH_SIZE = 500

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from labeling.prompts import CATEGORIES, SEVERITIES


def load_labels(path: Path) -> list[tuple[str, str, str, str]]:
    labels: list[tuple[str, str, str, str]] = []
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            record: dict[str, Any] = json.loads(line)
            review_id = record.get("review_id")
            category = record.get("category")
            severity = record.get("severity")
            justification = record.get("justification")
            if (
                not isinstance(review_id, str)
                or category not in CATEGORIES
                or severity not in SEVERITIES
                or not isinstance(justification, str)
            ):
                raise ValueError(f"Invalid pseudo-label on line {line_number}")
            labels.append((review_id, category, severity, justification))
    if not labels:
        raise ValueError(f"No labels found in {path}")
    return labels


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is required in .env")

    labels = load_labels(LABELS_PATH)
    connection = psycopg2.connect(database_url)
    try:
        updated_reviews = 0
        with connection.cursor() as cursor:
            for start in range(0, len(labels), BATCH_SIZE):
                batch = labels[start : start + BATCH_SIZE]
                execute_values(
                    cursor,
                    """
                    UPDATE reviews AS review
                    SET category = label.category,
                        severity = label.severity,
                        justification = label.justification
                    FROM (VALUES %s) AS label(id, category, severity, justification)
                    WHERE review.id = label.id
                    """,
                    batch,
                    template="(%s, %s, %s, %s)",
                    page_size=len(batch),
                )
                updated_reviews += cursor.rowcount

            cursor.execute(
                """
                UPDATE clusters AS cluster
                SET category = review.category,
                    severity = review.severity
                FROM reviews AS review
                WHERE cluster.representative_review_id = review.id
                """
            )
            updated_clusters = cursor.rowcount
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()

    print(
        f"Imported labels for {updated_reviews} reviews and updated "
        f"{updated_clusters} cluster labels."
    )


if __name__ == "__main__":
    main()
