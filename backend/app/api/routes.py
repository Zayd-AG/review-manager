"""FastAPI routes for the Feedback Lens dashboard and live classifier."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from backend.app.db import get_db
from backend.app.models import Cluster, ClusterMember, FeedbackItem
from backend.app.schemas import (
    ClassificationResponse,
    AppSearchResult,
    ClassifyRequest,
    ClusterDetailResponse,
    DashboardClusterResponse,
    DashboardSummaryResponse,
    FeedbackItemResponse,
    ImportJobResponse,
    ImportPreviewRequest,
    ImportRequest,
    RecommendationRequest,
    RecommendationResponse,
)
from backend.app.services.classifier import classifier
from backend.app.services.app_search import search_apps
from backend.app.services.review_import import (
    create_job,
    get_job,
    preview_reviews,
    run_import,
)
from backend.app.services.recommendations import build_plan


router = APIRouter()
logger = logging.getLogger(__name__)
SEVERITY_WEIGHTS = {"high": 3, "medium": 2, "low": 1}
MODEL_INFERENCE_TIMEOUT_SECONDS = int(
    os.getenv("MODEL_INFERENCE_TIMEOUT_SECONDS", "60")
)


@router.get("/apps/search", response_model=list[AppSearchResult])
async def app_search(
    query: str = Query(min_length=2, max_length=100),
    source: Literal["google_play", "app_store"] = Query(),
) -> list[AppSearchResult]:
    try:
        results = await run_in_threadpool(search_apps, source, query)
    except Exception as error:
        logger.warning("app_search_failed source=%s error_type=%s", source, type(error).__name__)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The app store search is temporarily unavailable.",
        ) from error
    return [AppSearchResult(**result) for result in results]


@router.post("/imports/preview")
async def preview_import(payload: ImportPreviewRequest) -> list[dict[str, object]]:
    start = (
        datetime.combine(payload.start_date, datetime.min.time(), tzinfo=timezone.utc)
        if payload.start_date
        else None
    )
    end = (
        datetime.combine(payload.end_date, datetime.max.time(), tzinfo=timezone.utc)
        if payload.end_date
        else None
    )
    if start and end and start > end:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Start date must be before end date.")
    try:
        return await run_in_threadpool(
            preview_reviews,
            payload.source,
            payload.identifier,
            payload.app_name,
            start,
            end,
        )
    except Exception as error:
        logger.warning("import_preview_failed source=%s error_type=%s", payload.source, type(error).__name__)
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Could not fetch a review preview.") from error


@router.post("/imports", response_model=ImportJobResponse, status_code=status.HTTP_202_ACCEPTED)
def start_import(payload: ImportRequest, background_tasks: BackgroundTasks) -> ImportJobResponse:
    reviews = [review.model_dump() for review in payload.reviews]
    if any(review["source"] != payload.source or review["app_name"] != payload.app_name for review in reviews):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Selected reviews must match the chosen app and source.")
    try:
        job = create_job(payload.source, payload.app_name, len(reviews))
    except RuntimeError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
    background_tasks.add_task(run_import, job.id, reviews)
    return ImportJobResponse(**job.snapshot())


@router.get("/imports/{job_id}", response_model=ImportJobResponse)
def import_status(job_id: str) -> ImportJobResponse:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Import job not found")
    return ImportJobResponse(**job.snapshot())


@router.post("/recommendations", response_model=RecommendationResponse)
async def recommendations(
    payload: RecommendationRequest, db: Session = Depends(get_db)
) -> RecommendationResponse:
    if payload.provider == "anthropic" and not payload.confirm_paid_request:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Set confirm_paid_request to use Anthropic recommendations.",
        )
    try:
        plan = await run_in_threadpool(build_plan, db, payload.app_name, payload.provider)
        return RecommendationResponse(**plan)
    except (RuntimeError, ValueError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(error)) from error


@router.get("/summary", response_model=DashboardSummaryResponse)
def dashboard_summary(db: Session = Depends(get_db)) -> DashboardSummaryResponse:
    """Return live dataset counts and the recorded evaluation snapshot."""
    review_count = int(db.scalar(select(func.count(FeedbackItem.id))) or 0)
    cluster_count = int(db.scalar(select(func.count(Cluster.id))) or 0)
    return DashboardSummaryResponse(
        review_count=review_count,
        cluster_count=cluster_count,
        classifier_name="Qwen2.5-1.5B-Instruct + LoRA",
        embedding_model="all-MiniLM-L6-v2 (384 dimensions)",
        evaluation={
            "gold_set_reviews": 100,
            "base_category_accuracy": 0.44,
            "lora_category_accuracy": 0.72,
            "teacher_category_accuracy": 0.73,
        },
    )


@router.get("/dashboard", response_model=list[DashboardClusterResponse])
def dashboard(
    category: str | None = None,
    source: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_db),
) -> list[DashboardClusterResponse]:
    statement = select(Cluster)
    if category:
        statement = statement.where(Cluster.category == category)
    if source:
        statement = (
            statement.join(ClusterMember, ClusterMember.cluster_id == Cluster.id)
            .join(FeedbackItem, FeedbackItem.id == ClusterMember.review_id)
            .where(FeedbackItem.source == source)
            .distinct()
        )

    clusters = db.scalars(statement).all()
    ranked = sorted(
        clusters,
        key=lambda cluster: cluster.count * SEVERITY_WEIGHTS.get(cluster.severity or "", 1),
        reverse=True,
    )[:limit]
    cluster_ids = [cluster.id for cluster in ranked]
    source_rows = db.execute(
        select(
            ClusterMember.cluster_id,
            FeedbackItem.source,
            func.count(FeedbackItem.id),
        )
        .join(FeedbackItem, FeedbackItem.id == ClusterMember.review_id)
        .where(ClusterMember.cluster_id.in_(cluster_ids))
        .group_by(ClusterMember.cluster_id, FeedbackItem.source)
    ).all()
    source_breakdowns: dict[str, dict[str, int]] = {}
    for cluster_id, source_name, source_count in source_rows:
        source_breakdowns.setdefault(str(cluster_id), {})[str(source_name)] = int(
            source_count
        )
    return [
        DashboardClusterResponse(
            id=cluster.id,
            representative_text=cluster.representative_text,
            category=cluster.category,
            severity=cluster.severity,
            count=cluster.count,
            priority_score=cluster.count
            * SEVERITY_WEIGHTS.get(cluster.severity or "", 1),
            source_breakdown=source_breakdowns.get(cluster.id, {}),
        )
        for cluster in ranked
    ]


@router.get("/clusters/{cluster_id}", response_model=ClusterDetailResponse)
def cluster_detail(cluster_id: str, db: Session = Depends(get_db)) -> ClusterDetailResponse:
    cluster = db.get(Cluster, cluster_id)
    if cluster is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Cluster not found")

    reviews = db.scalars(
        select(FeedbackItem)
        .join(ClusterMember, ClusterMember.review_id == FeedbackItem.id)
        .where(ClusterMember.cluster_id == cluster_id)
        .order_by(FeedbackItem.date.desc())
    ).all()
    return ClusterDetailResponse(
        **cluster.__dict__,
        source_reviews=[FeedbackItemResponse.model_validate(review) for review in reviews],
    )


@router.post("/classify", response_model=ClassificationResponse)
async def classify(payload: ClassifyRequest) -> ClassificationResponse:
    if not payload.text.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="text must not be empty",
        )
    try:
        start_time = time.perf_counter()
        label = await asyncio.wait_for(
            run_in_threadpool(classifier.classify, payload.text),
            timeout=MODEL_INFERENCE_TIMEOUT_SECONDS,
        )
        logger.info(
            "classification_complete model=qwen2.5-1.5b-lora latency_ms=%.1f",
            (time.perf_counter() - start_time) * 1000,
        )
        return ClassificationResponse(**label)
    except TimeoutError as error:
        logger.warning(
            "classification_timeout timeout_seconds=%s",
            MODEL_INFERENCE_TIMEOUT_SECONDS,
        )
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Classification exceeded the model-inference timeout",
        ) from error
    except FileNotFoundError as error:
        logger.error("classification_unavailable adapter_missing=true")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)) from error
    except (ValueError, json.JSONDecodeError) as error:
        logger.warning("classification_invalid_output error_type=%s", type(error).__name__)
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(error)) from error
