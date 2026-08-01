"""Fetch a small App Store review sample from Apple's public review feed.

Example:
    python ingestion/scrape_app_store.py tiktok:835599320 discord:985746746
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_REVIEW_LIMIT = 100
REVIEWS_PER_PAGE = 50
MAX_REVIEW_PAGES = 10
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "app_store"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch up to 100 newest App Store reviews per app by default."
    )
    parser.add_argument(
        "apps",
        nargs="+",
        metavar="APP_NAME:APP_ID",
        help="App Store app pairs, e.g. discord:985746746",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_REVIEW_LIMIT,
        help="Maximum reviews to fetch per app (default: 100)",
    )
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be at least 1")
    return args


def parse_app_spec(app_spec: str) -> tuple[str, str]:
    app_name, separator, app_id = app_spec.partition(":")
    if not separator or not app_name or not app_id.isdigit():
        raise ValueError(
            f"Invalid app specification '{app_spec}'. Use APP_NAME:APP_ID, "
            "for example discord:985746746."
        )
    return app_name, app_id


def review_feed_url(app_id: str, page: int) -> str:
    return (
        "https://itunes.apple.com/us/rss/customerreviews/"
        f"id={app_id}/sortBy=mostRecent/page={page}/json"
    )


def fetch_feed_page(app_id: str, page: int) -> list[dict[str, Any]]:
    request = Request(
        review_feed_url(app_id, page),
        headers={"User-Agent": "feedback-lens/0.1"},
    )
    try:
        with urlopen(request, timeout=30) as response:
            payload = json.load(response)
    except HTTPError as error:
        raise RuntimeError(
            f"Apple returned HTTP {error.code} for page {page}"
        ) from error
    except URLError as error:
        raise RuntimeError(
            f"Could not reach Apple's review feed: {error.reason}"
        ) from error
    except json.JSONDecodeError as error:
        raise RuntimeError("Apple returned an invalid review-feed response") from error

    entries = payload.get("feed", {}).get("entry", [])
    return entries if isinstance(entries, list) else []


def serialize_review(review: dict[str, Any], app_name: str) -> dict[str, str | int]:
    """Convert one Apple feed entry to the project's raw-review schema."""
    return {
        "text": review.get("content", {}).get("label", ""),
        "rating": int(review["im:rating"]["label"]),
        "date": review["updated"]["label"],
        "app_name": app_name,
        "source": "app_store",
    }


def fetch_reviews(
    app_name: str, app_id: str, limit: int
) -> list[dict[str, str | int]]:
    """Return no more than limit newest reviews for one iOS app."""
    collected_reviews: list[dict[str, str | int]] = []
    page = 1

    while len(collected_reviews) < limit and page <= MAX_REVIEW_PAGES:
        entries = fetch_feed_page(app_id, page)
        if not entries:
            break

        for entry in entries:
            try:
                collected_reviews.append(serialize_review(entry, app_name))
            except KeyError:
                continue
            except TypeError:
                continue
            except ValueError:
                continue
            if len(collected_reviews) == limit:
                break

        if collected_reviews:
            output_path = save_reviews(app_name, collected_reviews)
            print(
                f"{app_name}: checkpoint after page {page} "
                f"({len(collected_reviews)} reviews) -> {output_path}"
            )

        if len(entries) < REVIEWS_PER_PAGE:
            break
        page += 1

    if not collected_reviews:
        raise RuntimeError("Apple returned no usable reviews; no file was written")
    return collected_reviews


def save_reviews(app_name: str, app_reviews: list[dict[str, str | int]]) -> Path:
    RAW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RAW_OUTPUT_DIR / f"{app_name}.json"
    output_path.write_text(json.dumps(app_reviews, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    args = parse_args()

    for app_spec in args.apps:
        try:
            app_name, app_id = parse_app_spec(app_spec)
            app_reviews = fetch_reviews(app_name, app_id, args.limit)
            output_path = save_reviews(app_name, app_reviews)
            print(f"{app_name}: pulled {len(app_reviews)} reviews -> {output_path}")
        except Exception as error:  # noqa: BLE001
            print(f"{app_spec}: failed to pull reviews ({error})")


if __name__ == "__main__":
    main()
