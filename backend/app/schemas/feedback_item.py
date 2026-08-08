"""Pydantic response schemas for feedback items."""

from __future__ import annotations

from datetime import datetime
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
