"""Tavily Search API collector —— 覆盖近似浏览器搜索结果，弥补 Google News RSS 的盲区。

文档: https://docs.tavily.com/docs/rest-api/api-reference
免费档约 1000 次/月，按关键词数 × 每日触发次数计费。
"""
from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone

import httpx


TAVILY_URL = "https://api.tavily.com/search"


def _make_id(url: str, title: str) -> str:
    return hashlib.sha1(f"{url}|{title}".encode()).hexdigest()[:16]


def _normalize_published(raw: str | None) -> str:
    if not raw:
        return datetime.now(timezone.utc).isoformat()
    try:
        # Tavily 返回的 published_date 一般是 "2024-05-12T08:30:00Z" 或 "2024-05-12"
        s = raw.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()
    except ValueError:
        return datetime.now(timezone.utc).isoformat()


def fetch_tavily(
    query: str,
    source_name: str,
    category: str,
    api_key: str,
    days: int = 2,
    max_results: int = 10,
    topic: str = "news",
) -> list[dict]:
    """对单个关键词查一次 Tavily。返回与 RSS collector 同 schema 的 item 列表。"""
    if not api_key:
        return []
    payload = {
        "query": query,
        "topic": topic,
        "days": days,
        "max_results": max_results,
        "include_answer": False,
        "include_raw_content": False,
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        r = httpx.post(TAVILY_URL, json=payload, headers=headers, timeout=30)
        r.raise_for_status()
        data = r.json()
    except httpx.HTTPStatusError as e:
        print(f"    ! Tavily HTTP {e.response.status_code}: {e.response.text[:200]}")
        return []
    except Exception as e:
        print(f"    ! Tavily 请求失败: {e}")
        return []

    items: list[dict] = []
    for res in data.get("results", []):
        url = (res.get("url") or "").strip()
        title = (res.get("title") or "").strip()
        if not url or not title:
            continue
        items.append({
            "id": _make_id(url, title),
            "source": source_name,
            "category": category,
            "title": title,
            "url": url,
            "summary": res.get("content") or "",
            "published": _normalize_published(res.get("published_date")),
            "signals": {"tavily_score": res.get("score")} if res.get("score") is not None else {},
            "matched_keywords": [],
        })
    return items


def resolve_api_key(env_name: str = "TAVILY_API_KEY") -> str:
    return os.environ.get(env_name, "").strip()
