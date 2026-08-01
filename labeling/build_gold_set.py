"""Interactively build a manually labeled review gold set.

Run from the project root:
    python labeling/build_gold_set.py

Enter ``q`` at any category or severity prompt to quit safely and resume later.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "reviews_normalized.jsonl"
GOLD_OUTPUT_PATH = PROJECT_ROOT / "eval" / "gold_set.jsonl"
STATE_PATH = PROJECT_ROOT / "eval" / ".gold_set_state.json"
GOLD_SET_SIZE = 250

CATEGORIES = (
    "bug",
    "feature_request",
    "praise",
    "churn_risk",
    "pricing_complaint",
    "usability_complaint",
    "other",
)
SEVERITIES = ("low", "medium", "high")


def load_reviews() -> dict[str, dict[str, Any]]:
    reviews: dict[str, dict[str, Any]] = {}
    with INPUT_PATH.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            review = json.loads(line)
            if not isinstance(review, dict) or not isinstance(review.get("id"), str):
                raise ValueError(f"Invalid review record on line {line_number}")
            reviews[review["id"]] = review
    return reviews


def load_used_ids() -> set[str]:
    if not GOLD_OUTPUT_PATH.exists():
        return set()

    used_ids: set[str] = set()
    with GOLD_OUTPUT_PATH.open(encoding="utf-8") as gold_file:
        for line_number, line in enumerate(gold_file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            review_id = record.get("review_id") if isinstance(record, dict) else None
            if not isinstance(review_id, str):
                raise ValueError(f"Invalid gold-set record on line {line_number}")
            used_ids.add(review_id)
    return used_ids


def load_state() -> dict[str, Any] | None:
    if not STATE_PATH.exists():
        return None
    with STATE_PATH.open(encoding="utf-8") as state_file:
        state = json.load(state_file)
    if not isinstance(state, dict) or not isinstance(state.get("review_ids"), list):
        raise ValueError(f"Invalid gold-set state file: {STATE_PATH}")
    return state


def save_state(review_ids: list[str], next_index: int) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    state = {"review_ids": review_ids, "next_index": next_index}
    temporary_path = STATE_PATH.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(STATE_PATH)


def choose_review_queue(
    all_reviews: dict[str, dict[str, Any]], used_ids: set[str]
) -> tuple[list[str], int]:
    state = load_state()
    if state is not None:
        queued_ids = [
            review_id
            for review_id in state["review_ids"]
            if review_id in all_reviews
        ]
        saved_index = int(state.get("next_index", 0))
        if queued_ids and saved_index < len(queued_ids):
            return queued_ids, min(saved_index, len(queued_ids))

    available_ids = [review_id for review_id in all_reviews if review_id not in used_ids]
    sample_size = min(GOLD_SET_SIZE, len(available_ids))
    return random.SystemRandom().sample(available_ids, sample_size), 0


def prompt_choice(prompt: str, choices: tuple[str, ...]) -> str | None:
    choices_text = "/".join(choices)
    while True:
        answer = input(f"{prompt} [{choices_text}] (q=quit): ").strip().lower()
        if answer == "q":
            return None
        if answer in choices:
            return answer
        print(f"Please enter one of: {choices_text}, or q to quit.")


def main() -> None:
    all_reviews = load_reviews()
    used_ids = load_used_ids()
    if len(used_ids) >= GOLD_SET_SIZE:
        print(f"Gold set already contains {len(used_ids)} reviews; nothing to label.")
        return

    queue, next_index = choose_review_queue(all_reviews, used_ids)
    save_state(queue, next_index)

    GOLD_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with GOLD_OUTPUT_PATH.open("a", encoding="utf-8") as gold_file:
        for index in range(next_index, len(queue)):
            review = all_reviews[queue[index]]
            print("\n" + "=" * 72)
            print(f"Review {len(used_ids) + index + 1} of target {GOLD_SET_SIZE}")
            print(
                f"App: {review.get('app_name')} | Source: {review.get('source')} "
                f"| Rating: {review.get('rating')} | Date: {review.get('date')}"
            )
            print("\n" + str(review.get("text", "")) + "\n")

            category = prompt_choice("Category", CATEGORIES)
            if category is None:
                save_state(queue, index)
                print(f"Paused. Saved progress at {len(used_ids) + index} labels.")
                return
            severity = prompt_choice("Severity", SEVERITIES)
            if severity is None:
                save_state(queue, index)
                print(f"Paused. Saved progress at {len(used_ids) + index} labels.")
                return

            gold_record = {
                "review_id": review["id"],
                "category": category,
                "severity": severity,
                "justification": "Manually labeled gold-set example.",
                "teacher_model": "human",
            }
            gold_file.write(json.dumps(gold_record, ensure_ascii=False) + "\n")
            gold_file.flush()
            os.fsync(gold_file.fileno())
            save_state(queue, index + 1)

            completed = len(used_ids) + index + 1
            print(f"Saved gold label {completed}/{GOLD_SET_SIZE}.")

    print(f"Gold-set labeling complete: {len(used_ids) + len(queue)} labels.")


if __name__ == "__main__":
    main()
