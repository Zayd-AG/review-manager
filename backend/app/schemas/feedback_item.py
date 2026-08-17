"""Pydantic response schemas for feedback items."""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


Category = Literal[
    "bug",
    "feature_request",
    "praise",
    "churn_risk",
    "pricing_complaint",
    "usability_complaint",
    "other",
]
Severity = Literal["low", "medium", "high"]


class FeedbackItemResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    text: str
    source: str
    app_name: str
    rating: int | None
    date: datetime | None
    category: Category | None
    severity: Severity | None
    justification: str | None
    embedding_ref: str


class ClassifyRequest(BaseModel):
    text: str = Field(
        min_length=1,
        max_length=10_000,
        description="Review text to classify (maximum 10,000 characters).",
    )


class ClassificationResponse(BaseModel):
    category: Category
    severity: Severity
    justification: str


class AppSearchResult(BaseModel):
    name: str
    identifier: str
    developer: str | None
    icon_url: str | None
    store_url: str | None


class ImportPreviewRequest(BaseModel):
    source: Literal["google_play", "app_store"]
    app_name: str = Field(min_length=1, max_length=128)
    identifier: str = Field(min_length=1, max_length=255)
    start_date: date | None = None
    end_date: date | None = None


class ImportReview(BaseModel):
    text: str = Field(min_length=1, max_length=10_000)
    rating: int | None = Field(default=None, ge=1, le=5)
    date: datetime
    source: Literal["google_play", "app_store"]
    app_name: str = Field(min_length=1, max_length=128)


class ImportRequest(BaseModel):
    source: Literal["google_play", "app_store"]
    app_name: str = Field(min_length=1, max_length=128)
    reviews: list[ImportReview] = Field(min_length=1, max_length=20)


class ImportJobResponse(BaseModel):
    id: str
    source: Literal["google_play", "app_store"]
    app_name: str
    requested_reviews: int
    status: Literal["queued", "running", "completed", "failed"]
    fetched_reviews: int
    labeled_reviews: int
    saved_reviews: int
    error: str | None


class RecommendationRequest(BaseModel):
    app_name: str = Field(min_length=1, max_length=128)
    provider: Literal["local", "anthropic"] = "local"
    confirm_paid_request: bool = False


class RecommendationAction(BaseModel):
    priority: int
    title: str
    rationale: str
    evidence: str


class RecommendationResponse(BaseModel):
    provider: Literal["local", "anthropic"]
    summary: str
    actions: list[RecommendationAction]
