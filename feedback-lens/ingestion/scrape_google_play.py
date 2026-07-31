"""Fetch a small, recent Google Play review sample for specified app packages.

Example:
    python ingestion/scrape_google_play.py com.spotify.music com.discord
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from google_play_scraper import Sort, reviews


MAX_REVIEWS_PER_APP = 100
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "google_play"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch up to 100 newest Google Play reviews per app package."
    )
    parser.add_argument(
        "packages",
        nargs="+",
        help="Google Play package names, e.g. com.spotify.music",
    )
    return parser.parse_args()


def serialize_review(review: dict[str, Any], app_name: str) -> dict[str, str | int]:
    """Keep only the fields used by the project raw-review schema."""
    return {
        "text": review.get("content", ""),
        "rating": review.get("score", 0),
        "date": review["at"].isoformat(),
        "app_name": app_name,
        "source": "google_play",
    }


def fetch_reviews(package_name: str) -> list[dict[str, str | int]]:
    """Return no more than MAX_REVIEWS_PER_APP newest reviews for one package."""
    fetched_reviews, _ = reviews(
        package_name,
        lang="en",
        country="us",
        sort=Sort.NEWEST,
        count=MAX_REVIEWS_PER_APP,
    )
    return [serialize_review(review, package_name) for review in fetched_reviews]


def save_reviews(package_name: str, app_reviews: list[dict[str, str | int]]) -> Path:
    RAW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RAW_OUTPUT_DIR / f"{package_name}.json"
    output_path.write_text(json.dumps(app_reviews, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    args = parse_args()

    for package_name in args.packages:
        try:
            app_reviews = fetch_reviews(package_name)
            output_path = save_reviews(package_name, app_reviews)
            print(f"{package_name}: pulled {len(app_reviews)} reviews -> {output_path}")
        except Exception as error:
            print(f"{package_name}: failed to pull reviews ({error})")


if __name__ == "__main__":
    main()
