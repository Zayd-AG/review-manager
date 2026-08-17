"""Safe, small-batch review importing and local LoRA labeling."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import threading
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal
from urllib.request import Request, urlopen
from uuid import uuid4

from google_play_scraper import Sort, reviews as google_play_reviews
from sqlalchemy import text

from backend.app.db import SessionLocal
from backend.app.models import FeedbackItem
from backend.app.services.classifier import classifier


MAX_IMPORT_REVIEWS = 20
PREVIEW_REVIEW_LIMIT = 30
ImportSource = Literal["google_play", "app_store"]
logger = logging.getLogger(__name__)


@dataclass
class ImportJob:
    id: str
    source: ImportSource
    app_name: str
    requested_reviews: int
    status: Literal["queued", "running", "completed", "failed"] = "queued"
    fetched_reviews: int = 0
    labeled_reviews: int = 0
    saved_reviews: int = 0
    error: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def snapshot(self) -> dict[str, str | int | None]:
        with self.lock:
            return {
                "id": self.id,
                "source": self.source,
                "app_name": self.app_name,
                "requested_reviews": self.requested_reviews,
                "status": self.status,
                "fetched_reviews": self.fetched_reviews,
                "labeled_reviews": self.labeled_reviews,
                "saved_reviews": self.saved_reviews,
                "error": self.error,
            }


jobs: dict[str, ImportJob] = {}
jobs_lock = threading.Lock()
import_lock = threading.Lock()
cluster_rebuild_lock = threading.Lock()
embedding_model: Any | None = None
embedding_model_lock = threading.Lock()


def create_job(source: ImportSource, app_name: str, requested_reviews: int) -> ImportJob:
    with jobs_lock:
        if any(job.status in {"queued", "running"} for job in jobs.values()):
            raise RuntimeError("Another review import is already running.")
        job = ImportJob(
            id=str(uuid4()),
            source=source,
            app_name=app_name,
            requested_reviews=requested_reviews,
        )
        jobs[job.id] = job
    return job


def has_active_import() -> bool:
    with jobs_lock:
        return any(job.status in {"queued", "running"} for job in jobs.values())


def get_job(job_id: str) -> ImportJob | None:
    with jobs_lock:
        return jobs.get(job_id)


def fetch_reviews(source: ImportSource, identifier: str, app_name: str, limit: int) -> list[dict[str, Any]]:
    if source == "google_play":
        raw_reviews, _ = google_play_reviews(
            identifier,
            lang="en",
            country="us",
            sort=Sort.NEWEST,
            count=limit,
        )
        return [
            {
                "text": review.get("content", ""),
                "rating": review.get("score"),
                "date": review.get("at"),
                "source": source,
                "app_name": app_name,
            }
            for review in raw_reviews
            if isinstance(review.get("content"), str) and review["content"].strip()
        ][:limit]

    request = Request(
        "https://itunes.apple.com/us/rss/customerreviews/"
        f"id={identifier}/sortBy=mostRecent/page=1/json",
        headers={"User-Agent": "feedback-lens/0.1"},
    )
    with urlopen(request, timeout=30) as response:
        payload = json.load(response)
    entries = payload.get("feed", {}).get("entry", [])
    if not isinstance(entries, list):
        return []
    reviews: list[dict[str, Any]] = []
    for entry in entries:
        try:
            text = entry["content"]["label"]
            rating = int(entry["im:rating"]["label"])
            date = datetime.fromisoformat(entry["updated"]["label"])
        except (KeyError, TypeError, ValueError):
            continue
        if isinstance(text, str) and text.strip():
            reviews.append(
                {"text": text, "rating": rating, "date": date, "source": source, "app_name": app_name}
            )
        if len(reviews) == limit:
            break
    return reviews


def preview_reviews(
    source: ImportSource,
    identifier: str,
    app_name: str,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
) -> list[dict[str, Any]]:
    reviews = fetch_reviews(source, identifier, app_name, PREVIEW_REVIEW_LIMIT)
    if start_date is not None:
        reviews = [review for review in reviews if review["date"] >= start_date]
    if end_date is not None:
        reviews = [review for review in reviews if review["date"] <= end_date]
    return reviews


def review_id(review: dict[str, Any]) -> str:
    date = review["date"]
    date_value = date.isoformat() if isinstance(date, datetime) else str(date)
    payload = "|".join((review["text"], review["source"], review["app_name"], date_value))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def embed_text(text: str) -> str:
    global embedding_model
    with embedding_model_lock:
        if embedding_model is None:
            from sentence_transformers import SentenceTransformer

            embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        vector = embedding_model.encode(
            [text], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )[0]
    return "[" + ",".join(f"{float(value):.8f}" for value in vector) + "]"


def rebuild_clusters() -> None:
    with cluster_rebuild_lock:
        completed = subprocess.run(
            [sys.executable, "clustering/dedupe.py", "--threshold", "0.85"],
            cwd=os.getenv("PROJECT_ROOT", "/app"),
            env=os.environ.copy(),
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if completed.returncode:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"Clustering failed after import: {detail[-500:]}")


def run_import(job_id: str, selected_reviews: list[dict[str, Any]]) -> None:
    job = get_job(job_id)
    if job is None:
        return
    try:
        with import_lock:
            with job.lock:
                job.status = "running"
                job.fetched_reviews = len(selected_reviews)
            if not selected_reviews:
                raise RuntimeError("Select at least one review to import.")

            session = SessionLocal()
            try:
                for review in selected_reviews:
                    label = classifier.classify(str(review["text"]))
                    with job.lock:
                        job.labeled_reviews += 1
                    item_id = review_id(review)
                    item = session.get(FeedbackItem, item_id)
                    if item is None:
                        item = FeedbackItem(id=item_id)
                        session.add(item)
                    item.text = str(review["text"])
                    item.source = str(review["source"])
                    item.app_name = str(review["app_name"])
                    item.rating = int(review["rating"]) if review["rating"] is not None else None
                    item.date = review["date"]
                    item.category = label["category"]
                    item.severity = label["severity"]
                    item.justification = label["justification"]
                    item.embedding = None
                    session.commit()
                    embedding = embed_text(item.text)
                    session.execute(
                        text(
                            "UPDATE reviews SET embedding = CAST(:embedding AS vector) "
                            "WHERE id = :id"
                        ),
                        {"embedding": embedding, "id": item_id},
                    )
                    session.commit()
                    with job.lock:
                        job.saved_reviews += 1
            finally:
                session.close()
            rebuild_clusters()
            with job.lock:
                job.status = "completed"
    except Exception as error:
        logger.exception("import_failed job_id=%s app_name=%s", job.id, job.app_name)
        with job.lock:
            job.status = "failed"
            job.error = "Import processing failed. Review the backend logs for details."
