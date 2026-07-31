"""Dry-run teacher labeling for the first 10 normalized reviews.

Run from the project root after setting ANTHROPIC_API_KEY in feedback-lens/.env:
    python labeling/teacher_labeler.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import anthropic
from dotenv import load_dotenv

from prompts import CATEGORIES, SEVERITIES, build_labeling_prompt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "reviews_normalized.jsonl"
DRY_RUN_LIMIT = 10


def load_first_reviews(path: Path, limit: int) -> list[dict[str, Any]]:
    reviews: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                review = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number}") from error
            if not isinstance(review, dict):
                raise ValueError(f"Expected an object on line {line_number}")
            reviews.append(review)
            if len(reviews) == limit:
                break
    return reviews


def parse_label(response_text: str) -> dict[str, str]:
    """Parse and validate the structured response from the teacher model."""
    try:
        label = json.loads(response_text)
    except json.JSONDecodeError as error:
        raise ValueError("Model response was not valid JSON") from error

    expected_keys = {"category", "severity", "justification"}
    if not isinstance(label, dict) or set(label) != expected_keys:
        raise ValueError("Model response did not match the required label schema")
    if label["category"] not in CATEGORIES:
        raise ValueError(f"Unknown category: {label['category']}")
    if label["severity"] not in SEVERITIES:
        raise ValueError(f"Unknown severity: {label['severity']}")
    if not isinstance(label["justification"], str) or not label["justification"].strip():
        raise ValueError("Justification must be a non-empty string")
    return label


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Add it to feedback-lens/.env before running."
        )
    model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")

    reviews = load_first_reviews(INPUT_PATH, DRY_RUN_LIMIT)
    if not reviews:
        raise RuntimeError(f"No reviews found in {INPUT_PATH}")

    client = anthropic.Anthropic(api_key=api_key)
    for review in reviews:
        response = client.messages.create(
            model=model,
            max_tokens=256,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": build_labeling_prompt(str(review.get("text", ""))),
                }
            ],
        )
        label = parse_label(response.content[0].text)
        print(
            json.dumps(
                {
                    "review_id": review.get("id"),
                    "app_name": review.get("app_name"),
                    "source": review.get("source"),
                    "label": label,
                },
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
