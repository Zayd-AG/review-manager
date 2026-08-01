"""Print a comparison table and create a simple evaluation dashboard.

Examples:
    python eval/view_results.py eval/results/comparison_try_001.json
    python eval/view_results.py  # Uses the newest comparison*.json file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"
COLORS = ("#7c3aed", "#0891b2", "#ea580c")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "comparison",
        nargs="?",
        type=Path,
        help="Comparison JSON to display (default: newest comparison*.json).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="PNG output path (default: <comparison-name>_dashboard.png).",
    )
    return parser.parse_args()


def find_latest_comparison() -> Path:
    files = list(RESULTS_DIR.glob("comparison*.json"))
    if not files:
        raise FileNotFoundError(f"No comparison JSON files found in {RESULTS_DIR}")
    return max(files, key=lambda path: path.stat().st_mtime)


def load_comparison(path: Path) -> dict[str, Any]:
    with path.resolve().open(encoding="utf-8") as input_file:
        comparison = json.load(input_file)
    if not isinstance(comparison, dict) or not comparison.get("summary_table"):
        raise ValueError("Comparison JSON must contain a non-empty summary_table")
    return comparison


def build_summary_table(comparison: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for result in comparison["summary_table"]:
        rows.append(
            {
                "System": result["system"],
                "Category Accuracy": float(result["overall_accuracy"]) * 100,
                "Severity Accuracy": float(result["severity_accuracy"]) * 100,
                "Exact Accuracy": float(result["exact_label_accuracy"]) * 100,
                "Latency (ms)": float(result["average_latency_seconds"]) * 1000,
                "Cost / 1K ($)": float(result["estimated_cost_per_1000_reviews_usd"]),
                "Invalid": int(result["invalid_predictions"]),
            }
        )
    return pd.DataFrame(rows)


def print_table(summary: pd.DataFrame, review_count: Any, source: Path) -> None:
    display = summary.copy()
    for column in ("Category Accuracy", "Severity Accuracy", "Exact Accuracy"):
        display[column] = display[column].map(lambda value: f"{value:.1f}%")
    display["Latency (ms)"] = display["Latency (ms)"].map(lambda value: f"{value:.1f}")
    display["Cost / 1K ($)"] = display["Cost / 1K ($)"].map(
        lambda value: f"${value:.4f}"
    )
    display = display.rename(
        columns={
            "Category Accuracy": "Category",
            "Severity Accuracy": "Severity",
            "Exact Accuracy": "Exact",
            "Latency (ms)": "Latency ms",
            "Cost / 1K ($)": "Cost/1K",
        }
    )

    print(f"Evaluation — {review_count} reviews")
    print(f"Comparison: {source}\n")
    print(display.to_string(index=False, justify="center", col_space={"System": 18}))


def label_bars(axis: Any, bars: Any, suffix: str) -> None:
    for bar in bars:
        value = float(bar.get_height())
        axis.annotate(
            f"{value:.1f}{suffix}",
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def create_dashboard(
    summary: pd.DataFrame, review_count: Any, output_path: Path
) -> None:
    figure = plt.figure(figsize=(15, 14), constrained_layout=True)
    grid = figure.add_gridspec(3, 1, height_ratios=(0.75, 1.15, 1.05))
    table_axis = figure.add_subplot(grid[0, 0])
    accuracy_axis = figure.add_subplot(grid[1, 0])
    latency_axis = figure.add_subplot(grid[2, 0])

    table_axis.axis("off")
    table_axis.set_title(
        f"Evaluation\n{review_count} reviews",
        fontsize=18,
        fontweight="bold",
        pad=24,
    )
    table_data = summary.copy()
    for column in ("Category Accuracy", "Severity Accuracy", "Exact Accuracy"):
        table_data[column] = table_data[column].map(lambda value: f"{value:.1f}%")
    table_data["Latency (ms)"] = table_data["Latency (ms)"].map(lambda value: f"{value:.1f}")
    table_data["Cost / 1K ($)"] = table_data["Cost / 1K ($)"].map(
        lambda value: f"${value:.4f}"
    )
    rendered_table = table_axis.table(
        cellText=table_data.values,
        colLabels=table_data.columns,
        cellLoc="center",
        loc="center",
    )
    rendered_table.auto_set_font_size(False)
    rendered_table.set_fontsize(9)
    rendered_table.scale(1, 1.7)
    for (row, _column), cell in rendered_table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#dbeafe")
            cell.set_text_props(weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f8fafc")

    systems = summary["System"].tolist()
    positions = list(range(len(systems)))
    width = 0.24
    accuracy_columns = (
        ("Category Accuracy", "Category"),
        ("Severity Accuracy", "Severity"),
        ("Exact Accuracy", "Exact"),
    )
    for index, (column, label) in enumerate(accuracy_columns):
        offsets = [position + (index - 1) * width for position in positions]
        bars = accuracy_axis.bar(
            offsets,
            summary[column],
            width=width,
            color=COLORS[index],
            label=label,
        )
        label_bars(accuracy_axis, bars, "%")
    accuracy_axis.set_title("Accuracy comparison")
    accuracy_axis.set_ylabel("Accuracy (%)")
    accuracy_axis.set_ylim(0, 110)
    accuracy_axis.set_xticks(positions, systems, rotation=12, ha="right")
    accuracy_axis.legend(frameon=False)
    accuracy_axis.grid(axis="y", alpha=0.25)

    latency_bars = latency_axis.bar(
        systems,
        summary["Latency (ms)"],
        color=[COLORS[index % len(COLORS)] for index in positions],
    )
    label_bars(latency_axis, latency_bars, " ms")
    latency_axis.set_title("Average latency per review")
    latency_axis.set_ylabel("Milliseconds")
    latency_axis.tick_params(axis="x", rotation=12)
    latency_axis.grid(axis="y", alpha=0.25)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    comparison_path = (
        args.comparison.resolve() if args.comparison else find_latest_comparison()
    )
    comparison = load_comparison(comparison_path)
    summary = build_summary_table(comparison)
    review_count = comparison.get("gold_set_examples_evaluated", "unknown")
    output_path = (
        args.output.resolve()
        if args.output
        else comparison_path.with_name(f"{comparison_path.stem}_dashboard.png")
    )
    if output_path.suffix.lower() != ".png":
        raise ValueError("--output must use a .png extension")

    print_table(summary, review_count, comparison_path)
    create_dashboard(summary, review_count, output_path)
    print(f"\nSaved dashboard to {output_path}")


if __name__ == "__main__":
    main()
