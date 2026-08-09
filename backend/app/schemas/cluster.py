"""Pydantic response schemas for deduplicated review clusters."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from .feedback_item import Category, FeedbackItemResponse, Severity


class ClusterResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    similarity_threshold: float
    representative_review_id: str
    representative_text: str
    category: Category | None
    severity: Severity | None
    count: int = Field(ge=1)
    source_review_ids: list[str]
    created_at: datetime


class DashboardClusterResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    representative_text: str
    category: Category | None
    severity: Severity | None
    count: int = Field(ge=1)
    priority_score: int
    source_breakdown: dict[str, int]


class EvaluationSnapshotResponse(BaseModel):
    gold_set_reviews: int
    base_category_accuracy: float
    lora_category_accuracy: float
    teacher_category_accuracy: float


class DashboardSummaryResponse(BaseModel):
    review_count: int
    cluster_count: int
    classifier_name: str
    embedding_model: str
    evaluation: EvaluationSnapshotResponse


class ClusterDetailResponse(ClusterResponse):
    source_reviews: list[FeedbackItemResponse]
