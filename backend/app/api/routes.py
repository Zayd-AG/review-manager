"""FastAPI routes for the Feedback Lens dashboard and live classifier."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from backend.app.db import get_db
from backend.app.models import Cluster, ClusterMember, FeedbackItem
from backend.app.schemas import (
    ClassificationResponse,
    ClassifyRequest,
    ClusterDetailResponse,
    DashboardClusterResponse,
    FeedbackItemResponse,
)
from backend.app.services.classifier import classifier


router = APIRouter()
logger = logging.getLogger(__name__)
SEVERITY_WEIGHTS = {"high": 3, "medium": 2, "low": 1}
MODEL_INFERENCE_TIMEOUT_SECONDS = int(
    os.getenv("MODEL_INFERENCE_TIMEOUT_SECONDS", "60")
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
