"""Run Feedback Lens stages with visible progress and safe paid/scrape guards.

Examples:
    # Rebuild local normalized data, embeddings, and clusters.
    python pipeline/run_pipeline.py

    # Preview a 20-review source sample without scraping anything.
    python pipeline/run_pipeline.py --google-packages com.discord --scrape-limit 20 --dry-run

    # Run a reviewed paid sample, then rebuild local database stages.
    python pipeline/run_pipeline.py --stages label normalize embed cluster --label-limit 10

    # Full scrapes and full teacher labeling always require separate explicit flags.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "data" / "processed" / "pipeline_runs"
SAFE_SCRAPE_LIMIT = 20
STAGES = ("label", "normalize", "embed", "cluster")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=STAGES,
        default=("normalize", "embed", "cluster"),
        help="Stages to run (default: normalize embed cluster).",
    )
    parser.add_argument("--google-packages", nargs="+", default=[])
    parser.add_argument("--app-store-apps", nargs="+", default=[])
    parser.add_argument("--scrape-limit", type=int, default=SAFE_SCRAPE_LIMIT)
    parser.add_argument("--confirm-full-scrape", action="store_true")
    parser.add_argument("--label-limit", type=int, default=10)
    parser.add_argument("--full-labeling", action="store_true")
    parser.add_argument("--confirm-paid-full-labeling", action="store_true")
    parser.add_argument("--embed-batch-size", type=int, default=64)
    parser.add_argument("--cluster-threshold", type=float, default=0.85)
    parser.add_argument(
        "--dry-run", action="store_true", help="Print planned commands without running them."
    )
    args = parser.parse_args()
    if args.scrape_limit < 1 or args.label_limit < 1 or args.embed_batch_size < 1:
        parser.error("limits and batch size must be at least 1")
    if not 0 < args.cluster_threshold <= 1:
        parser.error("--cluster-threshold must be greater than 0 and no more than 1")
    if args.scrape_limit > SAFE_SCRAPE_LIMIT and not args.confirm_full_scrape:
        parser.error(
            f"--scrape-limit above {SAFE_SCRAPE_LIMIT} requires --confirm-full-scrape"
        )
    if args.full_labeling and not args.confirm_paid_full_labeling:
        parser.error(
            "--full-labeling requires --confirm-paid-full-labeling; use --estimate-only first"
        )
    if args.label_limit > SAFE_SCRAPE_LIMIT and not args.full_labeling:
        parser.error(
            f"--label-limit above {SAFE_SCRAPE_LIMIT} requires --full-labeling "
            "and --confirm-paid-full-labeling"
        )
    return args


def planned_commands(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    commands: list[tuple[str, list[str]]] = []
    if args.google_packages:
        commands.append(
            (
                "scrape_google_play",
                [
                    "ingestion/scrape_google_play.py",
                    *args.google_packages,
                    "--limit",
                    str(args.scrape_limit),
                ],
            )
        )
    if args.app_store_apps:
        commands.append(
            (
                "scrape_app_store",
                [
                    "ingestion/scrape_app_store.py",
                    *args.app_store_apps,
                    "--limit",
                    str(args.scrape_limit),
                ],
            )
        )
    for stage in args.stages:
        if stage == "label":
            command = ["labeling/teacher_labeler.py"]
            if args.full_labeling:
                command.extend(["--full", "--confirm-paid-full-dataset"])
            else:
                command.extend(["--limit", str(args.label_limit)])
        elif stage == "normalize":
            command = ["ingestion/normalize.py"]
        elif stage == "embed":
            command = ["clustering/embed.py", "--batch-size", str(args.embed_batch_size)]
        else:
            command = [
                "clustering/dedupe.py",
                "--threshold",
                str(args.cluster_threshold),
            ]
        commands.append((stage, command))
    return commands


def write_report(report: dict[str, Any]) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = REPORT_DIR / f"pipeline_run_{timestamp}.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    args = parse_args()
    commands = planned_commands(args)
    report: dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": args.dry_run,
        "stages": [],
    }
    if args.dry_run:
        print("Dry run: no scraper, paid API, model, or database stage will run.")

    try:
        for index, (stage, command) in enumerate(commands, start=1):
            printable_command = " ".join([sys.executable, *command])
            print(f"[{index}/{len(commands)}] {stage}: {printable_command}")
            stage_report = {"stage": stage, "command": command}
            if not args.dry_run:
                started = time.perf_counter()
                subprocess.run([sys.executable, *command], cwd=PROJECT_ROOT, check=True)
                stage_report["duration_seconds"] = round(time.perf_counter() - started, 2)
                print(f"[{index}/{len(commands)}] {stage}: complete")
            report["stages"].append(stage_report)
    except subprocess.CalledProcessError as error:
        report["status"] = "failed"
        report["failed_stage"] = stage
        report["return_code"] = error.returncode
        raise
    else:
        report["status"] = "planned" if args.dry_run else "completed"
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        report_path = write_report(report)
        print(f"Pipeline report: {report_path}")


if __name__ == "__main__":
    main()
