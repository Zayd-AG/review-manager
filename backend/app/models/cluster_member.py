"""Internal mapping model for reviews belonging to a cluster."""

from sqlalchemy import ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class ClusterMember(Base):
    __tablename__ = "cluster_members"

    cluster_id: Mapped[str] = mapped_column(ForeignKey("clusters.id"), primary_key=True)
    review_id: Mapped[str] = mapped_column(ForeignKey("reviews.id"), primary_key=True)
