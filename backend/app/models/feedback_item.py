"""SQLAlchemy model for a review and its optional label/embedding."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, deferred, mapped_column
from sqlalchemy.types import UserDefinedType

from .base import Base


class Vector(UserDefinedType[Any]):
    """Minimal pgvector type mapping without requiring a separate Python package."""

    cache_ok = True

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    def get_col_spec(self, **_kwargs: Any) -> str:
        return f"vector({self.dimensions})"


class FeedbackItem(Base):
    """One source review, with teacher/manual labels and its pgvector embedding."""

    __tablename__ = "reviews"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    app_name: Mapped[str] = mapped_column(String(128), nullable=False)
    rating: Mapped[int | None] = mapped_column(Integer)
    date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # Labels are nullable until pseudo-labels or human labels are imported.
    category: Mapped[str | None] = mapped_column(String(64))
    severity: Mapped[str | None] = mapped_column(String(16))
    justification: Mapped[str | None] = mapped_column(Text)

    # Deferred keeps the large vector out of normal API reads.
    embedding: Mapped[Any | None] = deferred(mapped_column(Vector(384)))

    @property
    def embedding_ref(self) -> str:
        """Stable API reference; the vector itself is intentionally not serialized."""
        return f"reviews/{self.id}/embedding"
