"""Prepare pseudo-labeled reviews for instruction tuning."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NORMALIZED_PATH = PROJECT_ROOT / "data" / "processed" / "reviews_normalized.jsonl"
PSEUDO_LABELS_PATH = PROJECT_ROOT / "data" / "processed" / "pseudo_labels.jsonl"
GOLD_SET_PATH = PROJECT_ROOT / "eval" / "gold_set.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "finetuning" / "data"
TRAIN_PATH = OUTPUT_DIR / "train.jsonl"
VAL_PATH = OUTPUT_DIR / "val.jsonl"
TRAIN_RATIO = 0.9
RANDOM_SEED = 42

TASK_DESCRIPTION = (
    "Classify this product review using exactly one category from bug, "
    "feature_request, praise, churn_risk, pricing_complaint, "
    "usability_complaint, or other; choose severity low, medium, or high; "
    "and provide a one-sentence justification. Return only valid JSON with "
    "category, severity, and justification keys."
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Expected an object in {path} on line {line_number}")
            records.append(record)
    return records


def load_gold_ids() -> set[str]:
    if not GOLD_SET_PATH.exists():
        return set()
    return {
        record["review_id"]
        for record in read_jsonl(GOLD_SET_PATH)
        if isinstance(record.get("review_id"), str)
    }


def build_examples() -> tuple[list[dict[str, str]], int]:
    normalized_reviews = {
        record["id"]: record
        for record in read_jsonl(NORMALIZED_PATH)
        if isinstance(record.get("id"), str)
    }
    gold_ids = load_gold_ids()
    pseudo_labels = read_jsonl(PSEUDO_LABELS_PATH)
    examples: list[dict[str, str]] = []
    excluded_gold = 0
    missing_reviews: list[str] = []

    for label in pseudo_labels:
        review_id = label.get("review_id")
        if not isinstance(review_id, str):
            raise ValueError("Pseudo-label record is missing a string review_id")
        if review_id in gold_ids:
            excluded_gold += 1
            continue
        review = normalized_reviews.get(review_id)
        if review is None:
            missing_reviews.append(review_id)
            continue

        response = {
            "category": label.get("category"),
            "severity": label.get("severity"),
            "justification": label.get("justification"),
        }
        examples.append(
            {
                "instruction": (
                    f"{TASK_DESCRIPTION}\n\nReview:\n{review.get('text', '')}"
                ),
                "response": json.dumps(response, ensure_ascii=False),
            }
        )

    if missing_reviews:
        raise ValueError(
            f"Could not find {len(missing_reviews)} pseudo-labeled reviews in the normalized data"
        )
    return examples, excluded_gold


def write_jsonl(path: Path, records: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    examples, excluded_gold = build_examples()
    random.Random(RANDOM_SEED).shuffle(examples)
    split_index = int(len(examples) * TRAIN_RATIO)
    train_examples = examples[:split_index]
    val_examples = examples[split_index:]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(TRAIN_PATH, train_examples)
    write_jsonl(VAL_PATH, val_examples)

    print(
        json.dumps(
            {
                "total_examples": len(examples),
                "excluded_gold_set": excluded_gold,
                "train_examples": len(train_examples),
                "val_examples": len(val_examples),
                "train_path": str(TRAIN_PATH),
                "val_path": str(VAL_PATH),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
