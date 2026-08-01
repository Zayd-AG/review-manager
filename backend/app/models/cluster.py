"""SQLAlchemy model for a deduplicated review cluster."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class Cluster(Base):
    __tablename__ = "clusters"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    similarity_threshold: Mapped[float] = mapped_column(Float, nullable=False)
    representative_review_id: Mapped[str] = mapped_column(
        ForeignKey("reviews.id"), nullable=False
    )
    representative_text: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str | None] = mapped_column(String(64))
    severity: Mapped[str | None] = mapped_column(String(16))
    count: Mapped[int] = mapped_column("review_count", Integer, nullable=False)
    source_review_ids: Mapped[list[str]] = mapped_column("review_ids", JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
