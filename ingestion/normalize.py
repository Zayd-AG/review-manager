"""Merge raw review files into a normalized JSONL dataset.

Run from the project root:
    python ingestion/normalize.py
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
JSONL_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "reviews_normalized.jsonl"
JSON_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "reviews_normalized.json"
REQUIRED_FIELDS = ("text", "source", "app_name", "rating", "date")
APP_NAME_ALIASES = {
    "com.zhiliaoapp.musically": "tiktok",
    "com.discord": "discord",
    "com.anthropic.claude": "claude",
}


def review_id(review: dict[str, Any]) -> str:
    """Create a stable ID that distinguishes reviews across apps and sources."""
    identity = "\x1f".join(
        str(review[field]) for field in ("text", "source", "app_name", "date")
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def normalize_review(raw_review: dict[str, Any]) -> dict[str, str | int]:
    """Return the project's shared review schema from a raw scraper record."""
    missing_fields = [field for field in REQUIRED_FIELDS if field not in raw_review]
    if missing_fields:
        raise ValueError(f"missing required fields: {', '.join(missing_fields)}")

    raw_app_name = str(raw_review["app_name"])
    normalized = {
        "text": str(raw_review["text"]),
        "source": str(raw_review["source"]),
        "app_name": APP_NAME_ALIASES.get(raw_app_name, raw_app_name),
        "rating": int(raw_review["rating"]),
        "date": str(raw_review["date"]),
    }
    return {"id": review_id(normalized), **normalized}


def load_raw_reviews(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as raw_file:
        raw_reviews = json.load(raw_file)
    if not isinstance(raw_reviews, list):
        raise ValueError("expected a JSON array")
    if not all(isinstance(review, dict) for review in raw_reviews):
        raise ValueError("expected every array item to be a JSON object")
    return raw_reviews


def main() -> None:
    normalized_reviews: list[dict[str, str | int]] = []
    seen_ids: set[str] = set()
    skipped_records = 0

    for raw_path in sorted(RAW_DATA_DIR.rglob("*.json")):
        try:
            raw_reviews = load_raw_reviews(raw_path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            print(f"Skipping {raw_path}: {error}")
            continue

        for raw_review in raw_reviews:
            try:
                normalized = normalize_review(raw_review)
            except (TypeError, ValueError) as error:
                skipped_records += 1
                print(f"Skipping invalid review in {raw_path}: {error}")
                continue

            if normalized["id"] in seen_ids:
                skipped_records += 1
                continue

            seen_ids.add(normalized["id"])
            normalized_reviews.append(normalized)

    JSONL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with JSONL_OUTPUT_PATH.open("w", encoding="utf-8") as output_file:
        for review in normalized_reviews:
            output_file.write(json.dumps(review, ensure_ascii=False) + "\n")
    JSON_OUTPUT_PATH.write_text(
        json.dumps(normalized_reviews, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    source_counts: Counter[str] = Counter(
        str(review["source"]) for review in normalized_reviews
    )
    app_counts: Counter[str] = Counter(
        str(review["app_name"]) for review in normalized_reviews
    )

    summary: dict[str, Any] = {
        "total_reviews": len(normalized_reviews),
        "output_paths": {
            "jsonl": str(JSONL_OUTPUT_PATH),
            "json": str(JSON_OUTPUT_PATH),
        },
        "by_source": dict(sorted(source_counts.items())),
        "by_app_name": dict(sorted(app_counts.items())),
        "skipped_records": skipped_records,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
