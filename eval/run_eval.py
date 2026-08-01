"""Compare the base model, LoRA adapter, and teacher on the human gold set.

Examples:
    # Required paid-API sample (all three systems, first 10 reviews)
    python eval/run_eval.py --limit 10

    # Full comparison after reviewing the sample output
    python eval/run_eval.py --confirm-paid-teacher

    # Run only the free local systems
    python eval/run_eval.py --systems base lora
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

# huggingface_hub reads these settings while it is imported. Load .env first,
# and default to the same local cache used by the training commands so Windows
# does not download a second copy under the OneDrive-backed user cache.
if local_app_data := os.getenv("LOCALAPPDATA"):
    os.environ.setdefault("HF_HOME", str(Path(local_app_data) / "huggingface"))

import anthropic
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finetuning.prepare_dataset import TASK_DESCRIPTION
from labeling.prompts import (
    CATEGORIES,
    LABEL_SCHEMA,
    SEVERITIES,
    build_labeling_prompt,
)


GOLD_SET_PATH = PROJECT_ROOT / "eval" / "gold_set.jsonl"
NORMALIZED_PATH = PROJECT_ROOT / "data" / "processed" / "reviews_normalized.jsonl"
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"
BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH = (
    PROJECT_ROOT
    / "finetuning"
    / "checkpoints"
    / "qwen2-5-1-5b-feedback-lens-lora-3ep-v1"
)
DEFAULT_TEACHER_MODEL = "claude-sonnet-4-5"
LOCAL_MAX_NEW_TOKENS = 160
TEACHER_MAX_TOKENS = 256
MAX_RETRIES = 3
PAID_SAMPLE_CAP = 20

# Standard API prices in USD per million tokens. Claude Sonnet 4.5 uses the
# same $3 input / $15 output pricing as Claude Sonnet 4.
ANTHROPIC_PRICING_PER_MILLION = {
    "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        help="Evaluate only the first N gold examples (use 10-20 for a test run).",
    )
    parser.add_argument(
        "--systems",
        nargs="+",
        choices=("base", "lora", "teacher"),
        default=("base", "lora", "teacher"),
        help="Systems to evaluate (default: all three).",
    )
    parser.add_argument(
        "--confirm-paid-teacher",
        action="store_true",
        help="Explicitly approve more than 20 paid teacher API calls.",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be greater than zero")
    return args


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path} on line {line_number}") from error
            if not isinstance(record, dict):
                raise ValueError(f"Expected an object in {path} on line {line_number}")
            records.append(record)
    return records


def load_gold_examples(limit: int | None) -> list[dict[str, str]]:
    normalized_by_id = {
        record["id"]: record
        for record in read_jsonl(NORMALIZED_PATH)
        if isinstance(record.get("id"), str)
    }
    gold_records = read_jsonl(GOLD_SET_PATH)
    if limit is not None:
        gold_records = gold_records[:limit]

    examples: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for index, gold in enumerate(gold_records, start=1):
        review_id = gold.get("review_id")
        category = gold.get("category")
        severity = gold.get("severity")
        if not isinstance(review_id, str) or not review_id:
            raise ValueError(f"Gold record {index} has no valid review_id")
        if review_id in seen_ids:
            raise ValueError(f"Duplicate gold review_id: {review_id}")
        if category not in CATEGORIES:
            raise ValueError(f"Gold record {review_id} has invalid category: {category}")
        if severity not in SEVERITIES:
            raise ValueError(f"Gold record {review_id} has invalid severity: {severity}")
        review = normalized_by_id.get(review_id)
        if review is None:
            raise ValueError(f"Gold review {review_id} is missing from normalized data")
        text = review.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Gold review {review_id} has no usable text")
        seen_ids.add(review_id)
        examples.append(
            {
                "review_id": review_id,
                "text": text,
                "category": str(category),
                "severity": str(severity),
            }
        )
    if not examples:
        raise RuntimeError("The selected gold set is empty")
    return examples


def parse_label(response_text: str) -> dict[str, str]:
    """Validate a JSON label, allowing only a harmless Markdown JSON fence."""
    normalized = response_text.strip()
    if normalized.startswith("```") and normalized.endswith("```"):
        first_newline = normalized.find("\n")
        fence_header = normalized[:first_newline].strip().lower()
        if first_newline != -1 and fence_header in {"```", "```json"}:
            normalized = normalized[first_newline + 1 : -3].strip()

    try:
        label = json.loads(normalized)
    except json.JSONDecodeError as error:
        raise ValueError("Response was not valid JSON") from error

    expected_keys = {"category", "severity", "justification"}
    if not isinstance(label, dict) or set(label) != expected_keys:
        raise ValueError("Response did not match the required label schema")
    if label["category"] not in CATEGORIES:
        raise ValueError(f"Unknown category: {label['category']}")
    if label["severity"] not in SEVERITIES:
        raise ValueError(f"Unknown severity: {label['severity']}")
    if not isinstance(label["justification"], str) or not label["justification"].strip():
        raise ValueError("Justification must be a non-empty string")
    return label


def local_prompt(review_text: str) -> str:
    return f"{TASK_DESCRIPTION}\n\nReview:\n{review_text}"


def load_local_model(with_adapter: bool) -> tuple[Any, Any]:
    if with_adapter and not (ADAPTER_PATH / "adapter_model.safetensors").exists():
        raise FileNotFoundError(f"LoRA adapter not found at {ADAPTER_PATH}")

    tokenizer_source = ADAPTER_PATH if with_adapter else BASE_MODEL_ID
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        device = torch.device("cuda")
    else:
        dtype = torch.float32
        device = torch.device("cpu")

    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, dtype=dtype)
    model = (
        PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        if with_adapter
        else base_model
    )
    model.to(device)
    model.eval()
    return model, tokenizer


def generate_local(model: Any, tokenizer: Any, review_text: str) -> str:
    messages = [{"role": "user", "content": local_prompt(review_text)}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=LOCAL_MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = generated[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def response_text(response: anthropic.types.Message) -> str:
    return "".join(
        block.text for block in response.content if block.type == "text"
    ).strip()


def call_teacher(
    client: anthropic.Anthropic, model: str, review_text: str
) -> tuple[dict[str, str], int, int]:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=TEACHER_MAX_TOKENS,
                temperature=0,
                thinking={"type": "disabled"},
                output_config={
                    "format": {"type": "json_schema", "schema": LABEL_SCHEMA}
                },
                messages=[
                    {"role": "user", "content": build_labeling_prompt(review_text)}
                ],
            )
            return (
                parse_label(response_text(response)),
                response.usage.input_tokens,
                response.usage.output_tokens,
            )
        except (anthropic.APIError, ValueError) as error:
            if attempt == MAX_RETRIES:
                raise RuntimeError(
                    f"Teacher request failed after {MAX_RETRIES} attempts"
                ) from error
            delay = 2 ** (attempt - 1)
            print(f"Teacher request failed; retrying in {delay}s: {error}")
            time.sleep(delay)
    raise RuntimeError("Unreachable retry state")


def synchronize_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def calculate_metrics(
    examples: list[dict[str, str]], predictions: list[dict[str, str] | None]
) -> dict[str, Any]:
    if len(examples) != len(predictions):
        raise ValueError("Examples and predictions have different lengths")

    per_category: dict[str, dict[str, float | int]] = {}
    for category in CATEGORIES:
        true_positive = sum(
            prediction is not None
            and example["category"] == category
            and prediction["category"] == category
            for example, prediction in zip(examples, predictions, strict=True)
        )
        false_positive = sum(
            prediction is not None
            and example["category"] != category
            and prediction["category"] == category
            for example, prediction in zip(examples, predictions, strict=True)
        )
        false_negative = sum(
            example["category"] == category
            and (prediction is None or prediction["category"] != category)
            for example, prediction in zip(examples, predictions, strict=True)
        )
        support = sum(example["category"] == category for example in examples)
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_category[category] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    category_correct = sum(
        prediction is not None and prediction["category"] == example["category"]
        for example, prediction in zip(examples, predictions, strict=True)
    )
    severity_correct = sum(
        prediction is not None and prediction["severity"] == example["severity"]
        for example, prediction in zip(examples, predictions, strict=True)
    )
    exact_correct = sum(
        prediction is not None
        and prediction["category"] == example["category"]
        and prediction["severity"] == example["severity"]
        for example, prediction in zip(examples, predictions, strict=True)
    )
    count = len(examples)
    return {
        "overall_accuracy": category_correct / count,
        "severity_accuracy": severity_correct / count,
        "exact_label_accuracy": exact_correct / count,
        "per_category": per_category,
    }


def evaluate_predictions(
    system_name: str,
    model_name: str,
    examples: list[dict[str, str]],
    predict: Callable[[str], dict[str, str]],
    cost_per_1000: Callable[[], float],
) -> dict[str, Any]:
    predictions: list[dict[str, str] | None] = []
    latencies: list[float] = []
    invalid_outputs = 0

    for index, example in enumerate(examples, start=1):
        synchronize_cuda()
        started = time.perf_counter()
        try:
            prediction = predict(example["text"])
        except ValueError as error:
            prediction = None
            invalid_outputs += 1
            print(f"{system_name} {index}/{len(examples)}: invalid output ({error})")
        synchronize_cuda()
        latencies.append(time.perf_counter() - started)
        predictions.append(prediction)
        if index % 10 == 0 or index == len(examples):
            print(f"{system_name}: processed {index}/{len(examples)}")

    return {
        "system": system_name,
        "model": model_name,
        "examples": len(examples),
        "valid_predictions": len(examples) - invalid_outputs,
        "invalid_predictions": invalid_outputs,
        "metrics": calculate_metrics(examples, predictions),
        "average_latency_seconds": sum(latencies) / len(latencies),
        "estimated_cost_per_1000_reviews_usd": cost_per_1000(),
    }


def evaluate_local(
    system_name: str, examples: list[dict[str, str]], with_adapter: bool
) -> dict[str, Any]:
    print(f"\nLoading {system_name}...")
    model, tokenizer = load_local_model(with_adapter=with_adapter)
    try:
        return evaluate_predictions(
            system_name=system_name,
            model_name=(f"{BASE_MODEL_ID} + {ADAPTER_PATH.name}" if with_adapter else BASE_MODEL_ID),
            examples=examples,
            predict=lambda text: parse_label(generate_local(model, tokenizer, text)),
            cost_per_1000=lambda: 0.0,
        )
    finally:
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def evaluate_teacher(
    examples: list[dict[str, str]], model: str
) -> dict[str, Any]:
    pricing = ANTHROPIC_PRICING_PER_MILLION.get(model)
    if pricing is None:
        known = ", ".join(sorted(ANTHROPIC_PRICING_PER_MILLION))
        raise ValueError(f"No pricing configured for {model}; known models: {known}")
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set in the project root .env")

    client = anthropic.Anthropic(api_key=api_key)
    input_tokens = 0
    output_tokens = 0

    def predict(text: str) -> dict[str, str]:
        nonlocal input_tokens, output_tokens
        label, request_input_tokens, request_output_tokens = call_teacher(
            client, model, text
        )
        input_tokens += request_input_tokens
        output_tokens += request_output_tokens
        return label

    def cost_per_1000() -> float:
        scale = 1000 / len(examples)
        return (
            input_tokens * scale * pricing["input"]
            + output_tokens * scale * pricing["output"]
        ) / 1_000_000

    result = evaluate_predictions(
        system_name="teacher",
        model_name=model,
        examples=examples,
        predict=predict,
        cost_per_1000=cost_per_1000,
    )
    result["api_usage"] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "actual_sample_cost_usd": (
            input_tokens * pricing["input"] + output_tokens * pricing["output"]
        )
        / 1_000_000,
        "pricing_per_million_tokens_usd": pricing,
    }
    return result


def format_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(value)) for width, value in zip(widths, row, strict=True)]

    def render(row: list[str]) -> str:
        return " | ".join(value.ljust(width) for value, width in zip(row, widths, strict=True))

    separator = "-+-".join("-" * width for width in widths)
    return "\n".join((render(headers), separator, *(render(row) for row in rows)))


def print_results(results: list[dict[str, Any]]) -> None:
    summary_rows = []
    for result in results:
        metrics = result["metrics"]
        summary_rows.append(
            [
                result["system"],
                f"{metrics['overall_accuracy']:.3f}",
                f"{metrics['severity_accuracy']:.3f}",
                f"{metrics['exact_label_accuracy']:.3f}",
                f"{result['average_latency_seconds'] * 1000:.1f}",
                f"${result['estimated_cost_per_1000_reviews_usd']:.4f}",
                str(result["invalid_predictions"]),
            ]
        )
    print("\nOverall results")
    print(
        format_table(
            ["System", "Category acc", "Severity acc", "Exact acc", "Latency ms", "Cost/1K", "Invalid"],
            summary_rows,
        )
    )

    for result in results:
        category_rows = []
        for category in CATEGORIES:
            values = result["metrics"]["per_category"][category]
            category_rows.append(
                [
                    category,
                    f"{values['precision']:.3f}",
                    f"{values['recall']:.3f}",
                    f"{values['f1']:.3f}",
                    str(values["support"]),
                ]
            )
        print(f"\nPer-category metrics: {result['system']}")
        print(format_table(["Category", "Precision", "Recall", "F1", "Support"], category_rows))


def build_summary_table(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "system": result["system"],
            "model": result["model"],
            "overall_accuracy": result["metrics"]["overall_accuracy"],
            "severity_accuracy": result["metrics"]["severity_accuracy"],
            "exact_label_accuracy": result["metrics"]["exact_label_accuracy"],
            "average_latency_seconds": result["average_latency_seconds"],
            "estimated_cost_per_1000_reviews_usd": result[
                "estimated_cost_per_1000_reviews_usd"
            ],
            "invalid_predictions": result["invalid_predictions"],
        }
        for result in results
    ]


def write_unique_results(output: dict[str, Any]) -> Path:
    """Write to the next comparison_try_NNN file without overwriting a run."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    try_number = 1
    while True:
        output_path = RESULTS_DIR / f"comparison_try_{try_number:03d}.json"
        try:
            with output_path.open("x", encoding="utf-8") as output_file:
                json.dump(output, output_file, ensure_ascii=False, indent=2)
                output_file.write("\n")
            return output_path
        except FileExistsError:
            try_number += 1


def main() -> None:
    args = parse_args()
    examples = load_gold_examples(args.limit)
    systems = list(dict.fromkeys(args.systems))

    if (
        "teacher" in systems
        and len(examples) > PAID_SAMPLE_CAP
        and not args.confirm_paid_teacher
    ):
        raise RuntimeError(
            f"Refusing {len(examples)} paid teacher calls without approval. "
            f"First run --limit 10 (or at most {PAID_SAMPLE_CAP}), review the "
            "output, then rerun with --confirm-paid-teacher."
        )

    results: list[dict[str, Any]] = []
    if "base" in systems:
        results.append(evaluate_local("base_zero_shot", examples, with_adapter=False))
    if "lora" in systems:
        results.append(evaluate_local("lora_finetuned", examples, with_adapter=True))
    if "teacher" in systems:
        teacher_model = os.getenv("ANTHROPIC_MODEL", DEFAULT_TEACHER_MODEL)
        results.append(evaluate_teacher(examples, teacher_model))

    print_results(results)
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "gold_set_path": str(GOLD_SET_PATH),
        "gold_set_examples_evaluated": len(examples),
        "metric_definitions": {
            "overall_accuracy": "Category accuracy across all selected gold reviews.",
            "severity_accuracy": "Severity accuracy across all selected gold reviews.",
            "exact_label_accuracy": "Both category and severity are correct.",
            "invalid_outputs": "Invalid JSON/schema outputs count as incorrect.",
            "latency": "Per-review inference/API time; model loading is excluded.",
            "local_cost": "Reported as $0 beyond local compute, as requested.",
        },
        "summary_table": build_summary_table(results),
        "detailed_results": results,
    }
    output_path = write_unique_results(output)
    print(f"\nSaved comparison to {output_path}")


if __name__ == "__main__":
    main()
