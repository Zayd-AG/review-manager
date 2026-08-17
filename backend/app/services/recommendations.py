"""Turn imported review clusters into a concise product-action plan."""

from __future__ import annotations

import json
import os
from typing import Any, Literal

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.models import Cluster, ClusterMember, FeedbackItem


def relevant_clusters(db: Session, app_name: str) -> list[Cluster]:
    return list(
        db.scalars(
            select(Cluster)
            .join(ClusterMember, ClusterMember.cluster_id == Cluster.id)
            .join(FeedbackItem, FeedbackItem.id == ClusterMember.review_id)
            .where(FeedbackItem.app_name == app_name)
            .distinct()
        )
    )


def local_plan(clusters: list[Cluster], reviews: list[FeedbackItem]) -> dict[str, Any]:
    ranked = sorted(
        clusters,
        key=lambda cluster: cluster.count * {"high": 3, "medium": 2, "low": 1}.get(cluster.severity or "", 1),
        reverse=True,
    )[:3]
    actions: list[dict[str, Any]] = [
        {
            "priority": index,
            "title": f"Address {cluster.category or 'unclassified'} feedback",
            "rationale": f"{cluster.count} related reviews; severity: {cluster.severity or 'unlabeled'}.",
            "evidence": cluster.representative_text,
        }
        for index, cluster in enumerate(ranked, start=1)
    ]
    if not actions:
        grouped: dict[tuple[str, str], list[FeedbackItem]] = {}
        for review in reviews:
            key = (review.category or "other", review.severity or "low")
            grouped.setdefault(key, []).append(review)
        for index, ((category, severity), matching_reviews) in enumerate(
            sorted(
                grouped.items(),
                key=lambda item: len(item[1]) * {"high": 3, "medium": 2, "low": 1}.get(item[0][1], 1),
                reverse=True,
            )[:3],
            start=1,
        ):
            actions.append(
                {
                    "priority": index,
                    "title": f"Review {category.replace('_', ' ')} feedback",
                    "rationale": f"{len(matching_reviews)} imported reviews; severity: {severity}.",
                    "evidence": matching_reviews[0].text,
                }
            )
    summary = (
        "No labeled reviews were available for this app."
        if not actions
        else f"Prioritized {len(actions)} recurring feedback themes by frequency and severity."
    )
    return {"provider": "local", "summary": summary, "actions": actions}


def anthropic_plan(clusters: list[Cluster]) -> dict[str, Any]:
    import anthropic

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not configured")
    evidence = [
        {
            "category": cluster.category,
            "severity": cluster.severity,
            "count": cluster.count,
            "example": cluster.representative_text,
        }
        for cluster in sorted(clusters, key=lambda item: item.count, reverse=True)[:10]
    ]
    prompt = (
        "You are a product manager. Using only this review-cluster evidence, return JSON "
        "with `summary` (one sentence) and `actions` (exactly 3 objects with `priority`, "
        "`title`, `rationale`, and `evidence`). Recommend concrete, evidence-backed product "
        "actions.\n\nEvidence:\n"
        + json.dumps(evidence, ensure_ascii=False)
    )
    response = anthropic.Anthropic(api_key=api_key).messages.create(
        model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5"),
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )
    content = response.content[0].text if response.content else ""
    plan = json.loads(content)
    if not isinstance(plan, dict) or not isinstance(plan.get("actions"), list):
        raise ValueError("Anthropic returned an invalid recommendation plan")
    return {"provider": "anthropic", **plan}


def build_plan(
    db: Session, app_name: str, provider: Literal["local", "anthropic"]
) -> dict[str, Any]:
    clusters = relevant_clusters(db, app_name)
    reviews = list(db.scalars(select(FeedbackItem).where(FeedbackItem.app_name == app_name)))
    if provider == "anthropic" and not clusters:
        return local_plan(clusters, reviews)
    return anthropic_plan(clusters) if provider == "anthropic" else local_plan(clusters, reviews)
