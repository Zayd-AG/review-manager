"""Store-search helpers for selecting an app before importing reviews."""

from __future__ import annotations

import json
from typing import Any, Literal
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from google_play_scraper import search as search_google_play


StoreSource = Literal["google_play", "app_store"]


def search_apps(source: StoreSource, query: str) -> list[dict[str, str | None]]:
    if source == "google_play":
        return search_google_play_apps(query)
    return search_app_store_apps(query)


def search_google_play_apps(query: str) -> list[dict[str, str | None]]:
    results = search_google_play(query, n_hits=10, lang="en", country="us")
    return [
        {
            "name": str(result.get("title") or result.get("appId") or "Unknown app"),
            "identifier": str(result.get("appId") or ""),
            "developer": str(result.get("developer") or "") or None,
            "icon_url": str(result.get("icon") or "") or None,
            "store_url": str(result.get("url") or "") or None,
        }
        for result in results
        if result.get("appId")
    ]


def search_app_store_apps(query: str) -> list[dict[str, str | None]]:
    parameters = urlencode(
        {"term": query, "entity": "software", "country": "us", "limit": 10}
    )
    request = Request(
        f"https://itunes.apple.com/search?{parameters}",
        headers={"User-Agent": "feedback-lens/0.1"},
    )
    with urlopen(request, timeout=15) as response:
        payload: dict[str, Any] = json.load(response)
    results = payload.get("results", [])
    if not isinstance(results, list):
        return []
    return [
        {
            "name": str(result.get("trackName") or result.get("trackId") or "Unknown app"),
            "identifier": str(result.get("trackId") or ""),
            "developer": str(result.get("sellerName") or "") or None,
            "icon_url": str(result.get("artworkUrl100") or "") or None,
            "store_url": str(result.get("trackViewUrl") or "") or None,
        }
        for result in results
        if result.get("trackId")
    ]
