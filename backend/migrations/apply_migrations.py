"""Apply the backend's SQL migrations in filename order.

Run from the project root:
    python backend/migrations/apply_migrations.py
"""

from __future__ import annotations

import os
from pathlib import Path

import psycopg2
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MIGRATIONS_DIR = Path(__file__).resolve().parent


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is required in .env")

    migrations = sorted(MIGRATIONS_DIR.glob("[0-9][0-9][0-9]_*.sql"))
    if not migrations:
        raise RuntimeError("No SQL migrations found")

    connection = psycopg2.connect(database_url)
    try:
        with connection.cursor() as cursor:
            for migration in migrations:
                cursor.execute(migration.read_text(encoding="utf-8"))
                print(f"Applied {migration.name}")
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


if __name__ == "__main__":
    main()
