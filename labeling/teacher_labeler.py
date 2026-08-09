"""Teacher-label normalized reviews with safe sample and full-dataset modes.

Run a 10-review paid sample from the project root:
    python labeling/teacher_labeler.py

Run the full dataset only after reviewing cost and explicitly confirming:
    python labeling/teacher_labeler.py --full --confirm-paid-full-dataset
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import anthropic
from dotenv import load_dotenv

from prompts import CATEGORIES, LABEL_SCHEMA, SEVERITIES, build_labeling_prompt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "reviews_normalized.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "pseudo_labels.jsonl"
DRY_RUN_LIMIT = 10
MAX_UNCONFIRMED_LABELS = 20
BATCH_SIZE = 50
MAX_RETRIES = 3
CHARS_PER_TOKEN_ESTIMATE = 4
ESTIMATED_OUTPUT_TOKENS_PER_REVIEW = 80
MODEL_PRICING_PER_MILLION = {
    "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Label a 10-review dry run or estimate full-dataset labeling cost."
    )
    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="Estimate full-dataset tokens and cost without calling the API.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DRY_RUN_LIMIT,
        help="Maximum pending reviews to label (default: 10).",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Label every remaining review; requires --confirm-paid-full-dataset.",
    )
    parser.add_argument(
        "--confirm-paid-full-dataset",
        action="store_true",
        help="Explicitly approve a full paid teacher-labeling run.",
    )
    return parser.parse_args()


def load_reviews(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
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
            if limit is not None and len(reviews) == limit:
                break
    return reviews


def count_reviews(path: Path) -> int:
    with path.open(encoding="utf-8") as input_file:
        return sum(1 for line in input_file if line.strip())


def estimate_tokens(text: str) -> int:
    """Estimate token count using a four-characters-per-token approximation."""
    return math.ceil(len(text) / CHARS_PER_TOKEN_ESTIMATE)


def print_cost_estimate(model: str) -> None:
    if model not in MODEL_PRICING_PER_MILLION:
        known_models = ", ".join(sorted(MODEL_PRICING_PER_MILLION))
        raise ValueError(
            f"No pricing configured for '{model}'. Add it to MODEL_PRICING_PER_MILLION "
            f"(known: {known_models})."
        )

    sample_reviews = load_reviews(INPUT_PATH, DRY_RUN_LIMIT)
    if not sample_reviews:
        raise RuntimeError(f"No reviews found in {INPUT_PATH}")
    dataset_review_count = count_reviews(INPUT_PATH)

    average_input_tokens = sum(
        estimate_tokens(build_labeling_prompt(str(review.get("text", ""))))
        for review in sample_reviews
    ) / len(sample_reviews)
    estimated_input_tokens = math.ceil(average_input_tokens * dataset_review_count)
    estimated_output_tokens = ESTIMATED_OUTPUT_TOKENS_PER_REVIEW * dataset_review_count
    pricing = MODEL_PRICING_PER_MILLION[model]
    estimated_cost = (
        estimated_input_tokens * pricing["input"]
        + estimated_output_tokens * pricing["output"]
    ) / 1_000_000

    print(
        json.dumps(
            {
                "mode": "estimate_only",
                "model": model,
                "dataset_reviews": dataset_review_count,
                "sample_reviews": len(sample_reviews),
                "average_estimated_input_tokens_per_review": round(
                    average_input_tokens, 1
                ),
                "assumed_output_tokens_per_review": ESTIMATED_OUTPUT_TOKENS_PER_REVIEW,
                "estimated_input_tokens": estimated_input_tokens,
                "estimated_output_tokens": estimated_output_tokens,
                "estimated_total_tokens": estimated_input_tokens
                + estimated_output_tokens,
                "estimated_cost_usd": round(estimated_cost, 4),
                "assumptions": {
                    "token_estimation": "4 characters per token",
                    "input_price_per_million_usd": pricing["input"],
                    "output_price_per_million_usd": pricing["output"],
                    "output_token_estimate": (
                        "80 tokens per JSON label; sample API usage was not persisted"
                    ),
                },
            },
            indent=2,
        )
    )


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


def response_text(response: anthropic.types.Message) -> str:
    """Extract all text blocks, ignoring any non-text model content blocks."""
    return "".join(
        block.text for block in response.content if block.type == "text"
    ).strip()


def load_completed_ids(path: Path) -> set[str]:
    """Return IDs already written to the incremental pseudo-label file."""
    if not path.exists():
        return set()

    completed_ids: set[str] = set()
    with path.open(encoding="utf-8") as output_file:
        for line_number, line in enumerate(output_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path} on line {line_number}") from error
            review_id = record.get("review_id")
            if isinstance(review_id, str):
                completed_ids.add(review_id)
    return completed_ids


def label_review(
    client: anthropic.Anthropic, model: str, review: dict[str, Any]
) -> dict[str, str]:
    """Label one review, retrying transient API and validation failures."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=256,
                temperature=0,
                thinking={"type": "disabled"},
                output_config={
                    "format": {"type": "json_schema", "schema": LABEL_SCHEMA}
                },
                messages=[
                    {
                        "role": "user",
                        "content": build_labeling_prompt(str(review.get("text", ""))),
                    }
                ],
            )
            return parse_label(response_text(response))
        except (anthropic.APIError, ValueError) as error:
            if attempt == MAX_RETRIES:
                raise RuntimeError(
                    f"Failed to label review {review.get('id')} after {MAX_RETRIES} attempts"
                ) from error
            delay_seconds = 2 ** (attempt - 1)
            print(
                f"Retrying review {review.get('id')} after error: {error} "
                f"(attempt {attempt}/{MAX_RETRIES})"
            )
            time.sleep(delay_seconds)

    raise RuntimeError("Unreachable retry state")


def label_dataset(
    client: anthropic.Anthropic, model: str, limit: int | None
) -> None:
    all_reviews = load_reviews(INPUT_PATH)
    completed_ids = load_completed_ids(OUTPUT_PATH)
    pending_reviews = [
        review for review in all_reviews if str(review.get("id", "")) not in completed_ids
    ]
    if limit is not None:
        pending_reviews = pending_reviews[:limit]

    print(
        f"Starting with {len(completed_ids)} saved labels; "
        f"{len(pending_reviews)} reviews remain."
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    newly_labeled = 0
    with OUTPUT_PATH.open("a", encoding="utf-8") as output_file:
        for batch_start in range(0, len(pending_reviews), BATCH_SIZE):
            batch = pending_reviews[batch_start : batch_start + BATCH_SIZE]
            for review in batch:
                label = label_review(client, model, review)
                output_file.write(
                    json.dumps(
                        {
                            "review_id": review.get("id"),
                            "category": label["category"],
                            "severity": label["severity"],
                            "justification": label["justification"],
                            "teacher_model": model,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                output_file.flush()
                os.fsync(output_file.fileno())
                newly_labeled += 1

            total_completed = len(completed_ids) + newly_labeled
            print(
                f"Processed {total_completed}/{len(all_reviews)} reviews "
                f"(completed batch of {len(batch)})."
            )


def main() -> None:
    args = parse_args()
    load_dotenv(PROJECT_ROOT / ".env")
    model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")
    if args.estimate_only:
        print_cost_estimate(model)
        return
    if args.limit < 1:
        raise ValueError("--limit must be at least 1")
    if args.full and not args.confirm_paid_full_dataset:
        raise RuntimeError(
            "Full paid labeling requires --confirm-paid-full-dataset. "
            "Run --estimate-only first to review expected cost."
        )
    if args.full:
        label_limit: int | None = None
    else:
        label_limit = args.limit
        if label_limit > MAX_UNCONFIRMED_LABELS:
            raise RuntimeError(
                f"Labeling more than {MAX_UNCONFIRMED_LABELS} reviews requires "
                "--full --confirm-paid-full-dataset."
            )

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Add it to the project root .env before running."
        )

    client = anthropic.Anthropic(api_key=api_key)
    label_dataset(client, model, label_limit)


if __name__ == "__main__":
    main()
